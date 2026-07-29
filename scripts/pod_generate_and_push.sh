#!/bin/sh
# Post-training on the pod: publish the checkpoint to W&B (in the background, overlapping
# generation), generate the full synthetic set and the rare-class top-ups, tar both and push
# to the HF dataset repo. Images are pushed UNCROPPED — cropping and study sampling happen
# locally, so the uncropped originals stay available for the perceptual study.
#
# Both conditioning variants wrote into the same results parent (000-DiT-XL-2 = text-only,
# 001-DiT-XL-2 = text+image), so RESULTS must name the SPECIFIC run directory. Set it and TAG
# together — TAG propagates into the output dirs, the tar, the HF filename and the W&B
# artifact, so a mismatch is painful to unpick later.
#
#   TAG=clip_text_image RESULTS=/mnt/resources/finediffusion_clip_results/001-DiT-XL-2 \
#       nohup sh scripts/pod_generate_and_push.sh > /mnt/resources/gen_main.log 2>&1 &
#
# Needs HF_TOKEN and WANDB_API_KEY in the environment. Re-running resumes: generation skips
# classes that already have their full complement.
set -u

REPO=${REPO:-/mnt/resources/FineDiffusion}
DATA=${DATA:-/mnt/datasets/ifcb_finediffusion}
RESULTS=${RESULTS:-/mnt/resources/finediffusion_clip_results/001-DiT-XL-2}
VENV=${VENV:-/root/fd-venv}
TAG=${TAG:-clip_text_image}
HF_REPO=${HF_REPO:-danielaivanova/ifcb-finediffusion}
BATCH=${BATCH:-32}
STEPS=${STEPS:-250}
CFG=${CFG:-4.0}
NGPU=${NGPU:-2}
PY="$VENV/bin/python"

export PATH=/mnt/resources/uv-bin:$PATH
export UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/root/hf-cache}
export PYTHONUNBUFFERED=1

log() { echo ">> [$(date +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }

log "TAG=$TAG  RESULTS=$RESULTS"

# ---- code ----
git -C "$REPO" pull --ff-only || log "git pull failed; using existing checkout"
cd "$REPO" || die "no repo at $REPO"
log "at commit: $(git log --oneline -1)"
grep -q "uses_clip" generate_synthetic_dataset.py \
    || die "generators predate the CLIP-aware commit (4f76a5e)"

# ---- venv (usually already built; rebuilt only if missing, e.g. after a pod restart) ----
if ! "$PY" -c "import torch,diffusers,timm,wandb,pandas" 2>/dev/null; then
    log "building venv at $VENV"
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    rm -rf "$VENV"
    uv venv --python 3.12 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    VIRTUAL_ENV="$VENV" uv pip install diffusers timm wandb pandas numpy pillow huggingface_hub
else
    log "venv already present"
fi
log "torch: $("$PY" -c 'import torch;print(torch.__version__, torch.cuda.device_count(), "GPUs")')"

# ---- checkpoint ----
CKPT=$(ls -t "$RESULTS"/checkpoints/*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] || die "no checkpoint under $RESULTS/checkpoints"
log "checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"

# ---- publish to W&B in the BACKGROUND ----
# Network-bound, so it overlaps GPU generation for free. ~5.4GB: model + EMA + optimizer
# state; only the EMA is needed to generate, the rest matters only to resume training.
log "starting W&B checkpoint upload in background"
"$PY" - > "/mnt/resources/wandb_upload_$TAG.log" 2>&1 <<PYEOF &
import wandb
run = wandb.init(project="finediffusion-ifcb", job_type="export",
                 name="ckpt-$TAG", reinit=True)
art = wandb.Artifact("finediffusion-$TAG", type="model", metadata={"path": "$CKPT"})
art.add_file("$CKPT", name="checkpoint.pt")
run.log_artifact(art); art.wait()
print("uploaded:", art.name, art.version)
run.finish()
PYEOF
WANDB_PID=$!

# ---- generate, sharded over the GPUs ----
# Waits on its OWN shard PIDs; a bare `wait` would also block on the W&B upload and
# serialise the two, defeating the overlap.
gen() {
    script=$1; out=$2; extra=$3; label=$4
    log "$label -> $out"
    pids=""
    S=0
    while [ "$S" -lt "$NGPU" ]; do
        CUDA_VISIBLE_DEVICES=$S "$PY" "$script" \
            --ckpt "$CKPT" \
            --train_csv "$DATA/anns/ifcb_train.csv" \
            --data_path "$DATA/Images" \
            --output_dir "$out" \
            --batch_size "$BATCH" --num_sampling_steps "$STEPS" --cfg_scale "$CFG" \
            --shard "$S" --num_shards "$NGPU" --resume --device cuda:0 $extra \
            > "/mnt/resources/${label}_${TAG}_shard$S.log" 2>&1 &
        pids="$pids $!"
        S=$((S + 1))
    done
    for p in $pids; do wait "$p"; done
    log "$label done: $(find "$out" -name '*.png' | wc -l) images"
}

gen generate_synthetic_dataset.py "$DATA/synthetic_$TAG" "" "gen"
gen generate_synthetic_dataset_oversample.py "$DATA/oversample_$TAG" "--min_samples 100" "over"

# ---- verify BEFORE packaging ----
# --resume skips classes that are already complete, so a shard that died mid-run would
# otherwise produce a short tar that is only noticed at FID time.
NFULL=$(find "$DATA/synthetic_$TAG" -name '*.png' | wc -l)
NOVER=$(find "$DATA/oversample_$TAG" -name '*.png' | wc -l)
log "counts: full=$NFULL (expect 59344)  oversample=$NOVER (expect 3109)"
[ "$NFULL" -eq 59344 ] || log "WARNING: full set incomplete — re-run this script to resume"

# ---- package and push (uncropped) ----
log "packing tar"
cd "$DATA" || die "no $DATA"
tar cf "/root/synth_$TAG.tar" --owner=0 --group=0 \
    "synthetic_$TAG" "oversample_$TAG" || die "tar failed"
log "tar: $(du -h "/root/synth_$TAG.tar" | cut -f1)"

if "$PY" - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/root/synth_$TAG.tar",
                    path_in_repo="synthetic_$TAG.tar",
                    repo_id="$HF_REPO", repo_type="dataset")
print("uploaded synthetic_$TAG.tar")
PYEOF
then
    log "HF upload OK"
else
    log "HF upload FAILED — tar is at /root/synth_$TAG.tar, retry manually"
fi

wait "$WANDB_PID" 2>/dev/null \
    && log "W&B upload OK" \
    || log "W&B upload FAILED (see /mnt/resources/wandb_upload_$TAG.log)"

log "ALL DONE"
log "  HF:    $HF_REPO / synthetic_$TAG.tar"
log "  W&B:   finediffusion-$TAG"
