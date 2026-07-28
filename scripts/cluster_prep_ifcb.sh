#!/bin/sh
# CPU-ONLY prep pod: stage IFCB + the pretrained DiT onto the shared volume, once.
# Paste into the launcher's Script box with hardware cpu-lg (no GPU) and Timeout 2 hours.
#
# POSIX sh (the launcher runs dash): no `set -o pipefail`, no `source`, absolute venv paths.
# Same lessons as the planktonzilla prep:
#   - venv on LOCAL disk (/root), NOT the network volume (chokes on venv small files)
#   - everything that must survive the pod goes to /mnt/datasets (persistent, shared)
#   - CPU torch wheel: this pod only downloads, it never trains (a CUDA wheel is 2.5GB wasted)
#
# Idempotent: re-running skips whatever is already in place, so a died pod can just be
# restarted. Total ~5GB of downloads, expect roughly 20-40 min on a good link.
set -u

DEST=/mnt/datasets/ifcb_finediffusion       # <- where training pods will read from
HF_REPO=danielaivanova/ifcb-finediffusion       # the dataset repo pushed by scripts/push_ifcb_to_hf.py
DIT_CKPT_URL=https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
VENV=/root/prep-venv
PY="$VENV/bin/python"

export UV_INSTALL_DIR=/mnt/resources/uv-bin
export PATH="$UV_INSTALL_DIR:$PATH"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
# Keep the HF cache OFF the network volume: it writes many small files and would crawl.
export HF_HOME=/root/hf-cache

log() { echo ">> [$(date +%H:%M:%S)] $*"; }

mkdir -p "$DEST" || { log "cannot write $DEST"; sleep infinity; }

# ---- tiny venv on LOCAL disk (huggingface_hub only; no torch needed to download) ----
command -v uv >/dev/null 2>&1 || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
uv python install 3.12
if ! "$PY" -c "import huggingface_hub" 2>/dev/null; then
    log "building $VENV (3.12) + huggingface_hub"
    rm -rf "$VENV"
    uv venv --python 3.12 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install "huggingface_hub[hf_transfer]"
fi
# hf_transfer gives a much faster multi-threaded download for the big image tree.
export HF_HUB_ENABLE_HF_TRANSFER=1
log "python: $("$PY" --version)"

# ---- the dataset repo is PRIVATE, so a token is required --------------------------
# Set HF_TOKEN in the pod env, or run `huggingface-cli login` in a Terminal first.
if ! "$PY" -c "
from huggingface_hub import HfApi
import sys
try:
    HfApi().dataset_info('$HF_REPO')
except Exception as e:
    print('CANNOT ACCESS $HF_REPO:', type(e).__name__, e); sys.exit(1)
" ; then
    log "Set HF_TOKEN (a read token) in the pod environment and restart. Holding pod open."
    sleep infinity
fi

# ---- IFCB images + anns ------------------------------------------------------------
if [ -f "$DEST/anns/ifcb_train.csv" ] && [ -d "$DEST/Images" ]; then
    log "IFCB already staged at $DEST — skipping"
else
    log "downloading $HF_REPO -> $DEST (2.3GB: Images.tar + anns)"
    "$PY" - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="$HF_REPO", repo_type="dataset",
                      local_dir="$DEST", max_workers=8)
print("downloaded to", p)
PYEOF
    [ -f "$DEST/anns/ifcb_train.csv" ] || { log "DOWNLOAD INCOMPLETE — no anns/ifcb_train.csv"; sleep infinity; }
fi

# If the repo holds Images.tar (the --tar upload path: 1 file instead of 74181, far faster
# in both directions), unpack it into the layout the training code walks.
if [ -f "$DEST/Images.tar" ] && [ ! -d "$DEST/Images" ]; then
    log "unpacking Images.tar (74181 files; a few minutes on the shared volume)"
    # --no-same-owner: the archive records the uploading workstation's uid/gid (1008), which
    # the pod cannot chown to. Without this, tar emits "Cannot change ownership ... Operation
    # not permitted" per file and exits non-zero even though every file extracted fine.
    # --no-same-permissions likewise defers to the pod's umask instead of the archived mode.
    tar xf "$DEST/Images.tar" -C "$DEST" --no-same-owner --no-same-permissions \
        || { log "UNTAR FAILED"; sleep infinity; }
    # Drop the archive only after a successful extract — 2.3GB back, and re-running the
    # script then re-downloads rather than silently finding a half-extracted tree.
    rm -f "$DEST/Images.tar"
    log "unpacked; removed the archive"
fi

# ---- per-image conditioning embeddings (100MB; only needed for the text+image arm) ----
# Too large to track in git, so it lives in the dataset repo alongside the images.
# snapshot_download above already pulls it if present; this is the explicit check.
if [ -f "$DEST/ifcb_rd32_participation_morpho_img.npz" ]; then
    log "per-image embeddings present ($(du -h "$DEST/ifcb_rd32_participation_morpho_img.npz" | cut -f1))"
else
    log "NOTE: no per-image embeddings in the repo — the text+image arm will not run"
    log "      (the text-only arm is unaffected; its npz ships in the git repo)"
fi

# ---- pretrained DiT-XL/2 checkpoint (2.7GB, public, straight from FAIR) -------------
if [ -f "$DEST/DiT-XL-2-256x256.pt" ]; then
    log "DiT checkpoint already present — skipping"
else
    log "downloading DiT-XL-2-256x256.pt (2.7GB)"
    # -C resumes a partial file if the pod died mid-download.
    curl -fL -C - -o "$DEST/DiT-XL-2-256x256.pt.part" "$DIT_CKPT_URL" \
        && mv "$DEST/DiT-XL-2-256x256.pt.part" "$DEST/DiT-XL-2-256x256.pt" \
        || { log "DiT download FAILED"; sleep infinity; }
fi

# ---- repos onto the shared volume ---------------------------------------------------
# BOTH are needed later: FineDiffusion to train, hyperbolic-plankton because the LoRA
# checkpoint is adapters over an open_clip backbone — make_clip_embeddings.py imports
# HyperbolicCLIP/apply_lora to rebuild the encoder. Cloning here (CPU pod, cheap) means the
# GPU pod starts straight into work.
clone_or_pull() {
    _dir="$1"; _url="$2"
    if [ -d "$_dir/.git" ]; then
        log "updating $(basename "$_dir")"
        git -C "$_dir" pull --ff-only || log "  pull failed; keeping existing checkout"
    else
        log "cloning $(basename "$_dir")"
        git clone "$_url" "$_dir" || log "  clone FAILED for $_url"
    fi
}
clone_or_pull /mnt/resources/FineDiffusion        https://github.com/daniela997/FineDiffusion.git
clone_or_pull /mnt/resources/hyperbolic-plankton  https://github.com/daniela997/hyperbolic-plankton.git

# ---- report ------------------------------------------------------------------------
log "staged contents:"
du -sh "$DEST"/* 2>/dev/null
N=$(find "$DEST/Images" -name '*.png' 2>/dev/null | wc -l)
log "image count: $N (expected 74181)"
[ "$N" -eq 74181 ] || log "WARNING: image count does not match — check the upload"

log ""
log "PREP DONE. Training pods read from $DEST"
log "NEXT (on the GPU pod, once the planktonzilla sweep has a checkpoint):"
log "  1. pull the sweep's best ckpt:"
log "       cd /mnt/resources/hyperbolic-plankton"
log "       python scripts/pull_ckpt.py uofg/hyperbolic-plankton-sweep/<artifact>:latest \\"
log "           --out /mnt/resources/hyperbolic_plankton_ckpts/from_cluster"
log "  2. generate the conditioning embeddings from it:"
log "       cd /mnt/resources/FineDiffusion"
log "       PYTHONPATH=/mnt/resources/hyperbolic-plankton/src python scripts/make_clip_embeddings.py \\"
log "           --ckpt <the .pt> --records $DEST/anns/ifcb_records.csv \\"
log "           --images $DEST/Images --out ifcb_rd32_v5.npz --name rd_r32_participation"
log "  3. run scripts/train_pod_ifcb_clip.sh (point --clip-embeddings at that npz)"
log ""
log "holding pod open so you can inspect; stop it manually."
sleep infinity
