#!/bin/bash
# FineDiffusion conditioned on rd_r32 (ranked-dedup LoRA-CLIP) taxonomy text embeddings,
# instead of the learned lookup table. Trains the ClipEmbedder (projection MLP + per-class
# code) + biases + norms on top of the frozen pretrained DiT-XL/2.
#
# Isolated --results-dir so it never collides with the label-conditioned runs in
# /scratch/daniela/finediffusion_results. NO --checkpoint: this starts fresh from the
# pretrained DiT (the old checkpoints have a LabelEmbedder, incompatible with ClipEmbedder).
export NCCL_P2P_DISABLE=1

eval "$(conda shell.bash hook)"
conda activate DiT

cd /home/daniela/other/FineDiffusion
torchrun --nnodes=1 --nproc_per_node=2 train.py \
  --model DiT-XL/2 \
  --resume /scratch/daniela/.cache/finediffusion/DiT-XL-2-256x256.pt \
  --data-path /scratch/datasets/other/IFCB_FishNet_Format/Images \
  --num-classes 145 \
  --num-super-classes 12 \
  --clip-embeddings /home/daniela/other/FineDiffusion/ifcb_rd32_hierarchical_embeddings.npz \
  --clip-code-dim 32 \
  --epochs 150 \
  --global-batch-size 64 \
  --image-size 256 \
  --results-dir /scratch/daniela/finediffusion_clip_results \
  --log-every 500 \
  --ckpt-every 5000
