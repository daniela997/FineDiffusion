#!/bin/sh
# FineDiffusion (CLIP-conditioned) training pod. Paste into the launcher's Script box.
# Requires scripts/cluster_prep_ifcb.sh to have run once (stages IFCB + the DiT ckpt).
#
# POSIX sh (the launcher runs dash): no `set -o pipefail`, no `source`, absolute venv paths.
#   - venv on LOCAL disk (/root), NOT the network volume
#   - results to /mnt/resources (persistent) so a reaped pod doesn't take the checkpoints
#   - NCCL_P2P_DISABLE=0: P2P works on A6000 (the =1 in the local script is an A5000 workaround)
set -u

REPO_DIR=/mnt/resources/FineDiffusion
REPO_URL=https://github.com/daniela997/FineDiffusion.git
DATA=/mnt/datasets/ifcb_finediffusion
VENV=/root/fd-venv
PY="$VENV/bin/python"
RESULTS=/mnt/resources/finediffusion_clip_results
# Conditioning embeddings: ranked-dedup r=32 + participation level weighting (sweep xaj57jjj
# run 55xwqbpe, eval/seen/species_f1 0.7142 — above the whole uniform arm). Built with
# --morpho, so all 145 conditioning strings are unique. Override by exporting CLIP_NPZ.
# The uniform-encoder arm (ifcb_rd32_morpho_fixed.npz — identical strings, only the encoder
# differs) is NOT tracked; regenerate it with make_clip_embeddings.py --morpho from the
# daiqxa8h checkpoint if that comparison is wanted again.
CLIP_NPZ="${CLIP_NPZ:-$REPO_DIR/ifcb_rd32_participation_morpho.npz}"
# 0 = PURE CLIP conditioning: no trainable per-class code. The code is a free per-class
# lookup table — what CLIP conditioning is meant to replace — and with 145 classes even 32
# dims can encode identity directly and route around CLIP, flattening an encoder ablation
# for the wrong reason. Safe at 0 only because --morpho makes every string unique; set 8-16
# if the near-synonymous pairs (Guinardia_delicatula_single vs _single_double, cos 0.993)
# turn out to generate poorly.
CLIP_CODE_DIM="${CLIP_CODE_DIM:-0}"
# Text+image arm: set CLIP_IMAGE=1 and point CLIP_NPZ at an npz built with --images-embed
# (staged to $DATA by the prep script, since it is 100MB and not in git). P_MEAN is the
# fraction of training samples whose image embedding is swapped for their class mean, so
# that the mean — all that is available at sampling time — is an in-distribution query.
CLIP_IMAGE="${CLIP_IMAGE:-0}"
P_MEAN="${P_MEAN:-0.5}"
# GPUs on this pod, and the GLOBAL batch (train.py divides it by world size). 64/2 = 32 per
# GPU, which is what the 2x24GB workstation runs; 64/4 = 16 per GPU on the A6000 pod. Keeping
# the global batch fixed keeps the optimisation identical across pod shapes — same effective
# batch, same steps per epoch — at the cost of sublinear speedup on more GPUs.
NPROC="${NPROC:-4}"
GLOBAL_BS="${GLOBAL_BS:-64}"
# Dataloader workers PER RANK, so the pod total is NUM_WORKERS x NPROC. This MUST stay at or
# below the pod's CPU count — oversubscribing makes throughput WORSE, not better.
# pad_to_square costs ~23ms/image, so one worker sustains only ~44 img/s. Measured per rank
# at batch 32 (2 ranks, so double the proc count for the pod total):
#     1 worker/rank   38 img/s   1.17 steps/s
#     2 workers/rank  79 img/s   2.48 steps/s   <- the ceiling on a 4-CPU pod
#     3 workers/rank 112 img/s   3.49 steps/s
#     4 workers/rank 181 img/s   5.67 steps/s   <- needs an 8-CPU pod
# On a 4-CPU / 2-GPU pod the loader caps you near 2.5 steps/s, which is roughly what 4090s
# compute anyway — so that shape is dataloader-bound and an 8-CPU pod is worth preferring.
# Default 2 suits the 4-CPU pods; raise to 4 on an 8-CPU pod.
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH="${PREFETCH:-6}"

export IFCB_ANNS="$DATA/anns"
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=0
export HF_HOME=/root/hf-cache            # sd-vae-ft-mse (~300MB) caches here, off the network vol
export UV_INSTALL_DIR=/mnt/resources/uv-bin
export PATH="$UV_INSTALL_DIR:$PATH"
export UV_LINK_MODE=copy

log() { echo ">> [$(date +%H:%M:%S)] $*"; }

# ---- data must already be staged ----------------------------------------------------
[ -f "$DATA/anns/ifcb_train.csv" ] || { log "NO DATA at $DATA — run scripts/cluster_prep_ifcb.sh first"; sleep infinity; }
[ -f "$DATA/DiT-XL-2-256x256.pt" ] || { log "NO DiT ckpt at $DATA — run scripts/cluster_prep_ifcb.sh first"; sleep infinity; }

# ---- repo ----------------------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    log "updating repo"
    git -C "$REPO_DIR" pull --ff-only || log "git pull failed; using existing code"
else
    log "cloning $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR" || { log "clone FAILED"; sleep infinity; }
fi
cd "$REPO_DIR" || { log "no repo at $REPO_DIR"; sleep infinity; }

# ---- venv on LOCAL disk --------------------------------------------------------------
command -v uv >/dev/null 2>&1 || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
uv python install 3.12
# open_clip_torch + peft are for scripts/make_clip_embeddings.py, which imports
# hyperbolic_plankton to rebuild the LoRA-CLIP text encoder — one venv covers both steps.
if ! "$PY" -c "import torch,diffusers,timm,wandb,pandas,open_clip,peft" 2>/dev/null; then
    log "building $VENV (3.12) + GPU torch + deps"
    rm -rf "$VENV"
    uv venv --python 3.12 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    VIRTUAL_ENV="$VENV" uv pip install diffusers timm wandb pandas numpy pillow \
        open_clip_torch peft huggingface_hub
else
    log "venv ready at $VENV"
fi
log "python: $("$PY" --version) | torch: $("$PY" -c 'import torch;print(torch.__version__)')"

# ---- verify the CLIP-embedding path exists before burning GPU hours -------------------
if ! "$PY" train.py --help 2>&1 | grep -q "clip-embeddings"; then
    log "FATAL: --clip-embeddings missing — checkout predates the ClipEmbedder commit. Holding open."
    sleep infinity
fi
# The conditioning .npz is NOT in git — it is generated from a LoRA checkpoint by
# scripts/make_clip_embeddings.py. Fail here rather than after the VAE + DiT have loaded.
if [ ! -f "$CLIP_NPZ" ]; then
    log "FATAL: no conditioning embeddings at $CLIP_NPZ"
    log "  generate them first:"
    log "    PYTHONPATH=/mnt/resources/hyperbolic-plankton/src $PY scripts/make_clip_embeddings.py \\"
    log "        --ckpt <lora.pt> --records $DATA/anns/ifcb_records.csv \\"
    log "        --images $DATA/Images --out <out.npz> --name <tag>"
    log "  then re-run with CLIP_NPZ=<out.npz>. Holding pod open."
    sleep infinity
fi
IMAGE_FLAGS=""
if [ "$CLIP_IMAGE" = "1" ]; then
    IMAGE_FLAGS="--clip-image --clip-image-p-mean $P_MEAN"
    log "conditioning: TEXT + PER-IMAGE, p_mean=$P_MEAN"
else
    log "conditioning: TEXT only"
fi
log "  npz: $CLIP_NPZ (code_dim=$CLIP_CODE_DIM)"

if [ $((GLOBAL_BS % NPROC)) -ne 0 ]; then
    log "FATAL: GLOBAL_BS=$GLOBAL_BS is not divisible by NPROC=$NPROC. Holding pod open."
    sleep infinity
fi

mkdir -p "$RESULTS"

# ---- train ---------------------------------------------------------------------------
# Conditioned on rd_r32 (ranked-dedup LoRA-CLIP) taxonomy text embeddings instead of the
# learned lookup table: trains ClipEmbedder (projection MLP + per-class code) + biases +
# norms on top of the frozen pretrained DiT-XL/2. No --checkpoint: starts fresh from the
# pretrained DiT (old checkpoints have a LabelEmbedder, incompatible with ClipEmbedder).
#
# global-batch-size 64 is kept from the local 2-GPU runs so the optimisation is unchanged
# and results stay comparable; it is a GLOBAL size, so it splits 16/GPU across 4.
log "launching FineDiffusion-CLIP on $NPROC GPUs (global batch $GLOBAL_BS = $((GLOBAL_BS/NPROC))/GPU)"
log "  dataloader: $NUM_WORKERS workers/rank x $NPROC = $((NUM_WORKERS*NPROC)) procs (pod has $(nproc) CPUs)"
if [ $((NUM_WORKERS*NPROC)) -gt "$(nproc)" ]; then
    log "  WARNING: more workers than CPUs — they will contend. Lower NUM_WORKERS."
fi
"$VENV/bin/torchrun" --nnodes=1 --nproc_per_node="$NPROC" train.py \
    --model DiT-XL/2 \
    --resume "$DATA/DiT-XL-2-256x256.pt" \
    --data-path "$DATA/Images" \
    --anns-dir "$DATA/anns" \
    --num-classes 145 \
    --num-super-classes 12 \
    --clip-embeddings "$CLIP_NPZ" \
    --clip-code-dim "$CLIP_CODE_DIM" \
    $IMAGE_FLAGS \
    --epochs 150 \
    --global-batch-size "$GLOBAL_BS" \
    --image-size 256 \
    --results-dir "$RESULTS" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH" \
    --log-every 500 \
    --ckpt-every 5000
rc=$?

if [ "$rc" -eq 0 ]; then
    log "training finished OK — results in $RESULTS (persistent)"
else
    log "training exited $rc — HOLDING POD OPEN so a crash doesn't delete it before you look"
    sleep infinity
fi
