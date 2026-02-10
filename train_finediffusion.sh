#!/bin/bash
export NCCL_P2P_DISABLE=1

# Initialize conda properly
eval "$(conda shell.bash hook)"
conda activate DiT

cd /home/daniela/other/FineDiffusion
torchrun --nnodes=1 --nproc_per_node=2 train.py \
  --model DiT-XL/2 \
  --resume /scratch/daniela/.cache/finediffusion/DiT-XL-2-256x256.pt\
  --checkpoint /scratch/daniela/finediffusion_results/000-DiT-XL-2/checkpoints/0015000.pt \
  --data-path /scratch/datasets/other/IFCB_FishNet_Format/Images \
  --num-classes 145 \
  --num-super-classes 12 \
  --epochs 120 \
  --global-batch-size 64 \
  --image-size 256 \
  --results-dir /scratch/daniela/finediffusion_results \
  --log-every 500 \
  --ckpt-every 5000