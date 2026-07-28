#!/bin/bash

# Create a new tmux session
tmux new-session -d -s finediffusion -x 200 -y 50

# Shard 0 on cuda:0
tmux new-window -t finediffusion -n shard0
tmux send-keys -t finediffusion:shard0 "cd /home/daniela/other/FineDiffusion && conda activate DiT && python generate_synthetic_dataset.py \
  --ckpt /scratch/daniela/finediffusion_results/002-DiT-XL-2/checkpoints/0135000.pt \
  --train_csv /scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv \
  --data_path /scratch/datasets/other/IFCB_FishNet_Format/Images \
  --output_dir /scratch/datasets/other/IFCB_FishNet_Format/FineDiffusion_synthetic  \
  --batch_size 24 \
  --num_sampling_steps 50 \
  --device cuda:0 \
  --shard 0 \
  --num_shards 4 \
  --resume" C-m

# Shard 1 on cuda:0
tmux new-window -t finediffusion -n shard1
tmux send-keys -t finediffusion:shard1 "cd /home/daniela/other/FineDiffusion && conda activate DiT && python generate_synthetic_dataset.py \
  --ckpt /scratch/daniela/finediffusion_results/002-DiT-XL-2/checkpoints/0135000.pt \
  --train_csv /scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv \
  --data_path /scratch/datasets/other/IFCB_FishNet_Format/Images \
  --output_dir /scratch/datasets/other/IFCB_FishNet_Format/FineDiffusion_synthetic \
  --batch_size 24 \
  --num_sampling_steps 50 \
  --device cuda:0 \
  --shard 1 \
  --num_shards 4 \
  --resume" C-m

# Shard 2 on cuda:1
tmux new-window -t finediffusion -n shard2
tmux send-keys -t finediffusion:shard2 "cd /home/daniela/other/FineDiffusion && conda activate DiT && python generate_synthetic_dataset.py \
  --ckpt /scratch/daniela/finediffusion_results/002-DiT-XL-2/checkpoints/0135000.pt \
  --train_csv /scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv \
  --data_path /scratch/datasets/other/IFCB_FishNet_Format/Images \
  --output_dir /scratch/datasets/other/IFCB_FishNet_Format/FineDiffusion_synthetic \
  --batch_size 24 \
  --num_sampling_steps 50 \
  --device cuda:1 \
  --shard 2 \
  --num_shards 4 \
  --resume" C-m

# Shard 3 on cuda:1
tmux new-window -t finediffusion -n shard3
tmux send-keys -t finediffusion:shard3 "cd /home/daniela/other/FineDiffusion && conda activate DiT && python generate_synthetic_dataset.py \
  --ckpt /scratch/daniela/finediffusion_results/002-DiT-XL-2/checkpoints/0135000.pt \
  --train_csv /scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv \
  --data_path /scratch/datasets/other/IFCB_FishNet_Format/Images \
  --output_dir /scratch/datasets/other/IFCB_FishNet_Format/FineDiffusion_synthetic \
  --batch_size 24 \
  --num_sampling_steps 50 \
  --device cuda:1 \
  --shard 3 \
  --num_shards 4 \
  --resume" C-m

echo "✅ Started finediffusion tmux session with 4 shards"
echo "📺 View all windows: tmux list-windows -t finediffusion"
echo "🔍 Attach: tmux attach -t finediffusion"
echo "📊 Monitor: tmux list-windows -t finediffusion && watch -n 1 'tmux capture-pane -t finediffusion -p'"