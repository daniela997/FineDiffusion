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
# Swap to ifcb_rd32_morpho_fixed.npz for the uniform-encoder arm: identical strings, so the
# only variable is the encoder.
CLIP_NPZ="${CLIP_NPZ:-$REPO_DIR/ifcb_rd32_participation_morpho.npz}"
# 0 = PURE CLIP conditioning: no trainable per-class code. The code is a free per-class
# lookup table — what CLIP conditioning is meant to replace — and with 145 classes even 32
# dims can encode identity directly and route around CLIP, flattening an encoder ablation
# for the wrong reason. Safe at 0 only because --morpho makes every string unique; set 8-16
# if the near-synonymous pairs (Guinardia_delicatula_single vs _single_double, cos 0.993)
# turn out to generate poorly.
CLIP_CODE_DIM="${CLIP_CODE_DIM:-0}"

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
log "conditioning: $CLIP_NPZ (code_dim=$CLIP_CODE_DIM)"

mkdir -p "$RESULTS"

# ---- train ---------------------------------------------------------------------------
# Conditioned on rd_r32 (ranked-dedup LoRA-CLIP) taxonomy text embeddings instead of the
# learned lookup table: trains ClipEmbedder (projection MLP + per-class code) + biases +
# norms on top of the frozen pretrained DiT-XL/2. No --checkpoint: starts fresh from the
# pretrained DiT (old checkpoints have a LabelEmbedder, incompatible with ClipEmbedder).
#
# global-batch-size 64 is kept from the local 2-GPU runs so the optimisation is unchanged
# and results stay comparable; it is a GLOBAL size, so it splits 16/GPU across 4.
log "launching FineDiffusion-CLIP on 4 GPUs (global batch 64)"
"$VENV/bin/torchrun" --nnodes=1 --nproc_per_node=4 train.py \
    --model DiT-XL/2 \
    --resume "$DATA/DiT-XL-2-256x256.pt" \
    --data-path "$DATA/Images" \
    --anns-dir "$DATA/anns" \
    --num-classes 145 \
    --num-super-classes 12 \
    --clip-embeddings "$CLIP_NPZ" \
    --clip-code-dim "$CLIP_CODE_DIM" \
    --epochs 150 \
    --global-batch-size 64 \
    --image-size 256 \
    --results-dir "$RESULTS" \
    --num-workers 2 \
    --log-every 500 \
    --ckpt-every 5000
rc=$?

if [ "$rc" -eq 0 ]; then
    log "training finished OK — results in $RESULTS (persistent)"
else
    log "training exited $rc — HOLDING POD OPEN so a crash doesn't delete it before you look"
    sleep infinity
fi
