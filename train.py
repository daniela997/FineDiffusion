# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from ast import arg
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import wandb
import torchvision
from torch.cuda.amp import autocast, GradScaler

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from ifcb_dataset import IFCBTrainDataset

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def pad_to_square(pil_image, target_size, strip=4):
    """
    Resize and pad PIL Image to square.
    - First resizes so the longer dimension = target_size (preserves aspect ratio)
    - Then pads with edge-tiled strips to make exactly square
    
    Args:
        pil_image: PIL Image
        target_size: target square size (e.g., 256)
        strip: number of edge pixels to use for tiling
    
    Returns:
        Padded square PIL Image of size (target_size, target_size)
    """
    import numpy as np
    
    image = np.array(pil_image)
    
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
        was_grayscale = True
    else:
        was_grayscale = False
    
    H, W = image.shape[:2]
    
    # Resize so longest dimension = target_size (preserves aspect ratio, no upscaling)
    if max(H, W) > target_size:
        scale = target_size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        image = np.array(Image.fromarray(image).resize((new_W, new_H), Image.BICUBIC))
        H, W = new_H, new_W
    
    # Now pad to make square
    pad_vertical = target_size - H
    pad_top = np.random.randint(0, pad_vertical + 1) if pad_vertical > 0 else 0
    pad_bottom = pad_vertical - pad_top
    
    pad_horizontal = target_size - W
    pad_left = np.random.randint(0, pad_horizontal + 1) if pad_horizontal > 0 else 0
    pad_right = pad_horizontal - pad_left
    
    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
    
    def tile_patch(patch, shape):
        reps = (
            (shape[0] + patch.shape[0] - 1) // patch.shape[0],
            (shape[1] + patch.shape[1] - 1) // patch.shape[1],
            1,
        )
        tiled = np.tile(patch, reps)
        return tiled[:shape[0], :shape[1]]
    
    # Replace padding with tiled edge strips
    if pad_top:
        patch = image[:strip, :, :]
        padded[:pad_top, pad_left:W + pad_left] = tile_patch(patch, (pad_top, W))
    if pad_bottom:
        patch = image[-strip:, :, :][::-1]
        padded[-pad_bottom:, pad_left:W + pad_left] = tile_patch(patch, (pad_bottom, W))
    if pad_left:
        patch = image[:, :strip, :]
        padded[pad_top:H + pad_top, :pad_left] = tile_patch(patch, (H, pad_left))
    if pad_right:
        patch = image[:, -strip:, :][:, ::-1, :]
        padded[pad_top:H + pad_top, -pad_right:] = tile_patch(patch, (H, pad_right))
    
    # Add noise to padded regions
    mask = np.zeros(padded.shape[:2], dtype=bool)
    if pad_top:
        mask[:pad_top, :] = True
    if pad_bottom:
        mask[-pad_bottom:, :] = True
    if pad_left:
        mask[:, :pad_left] = True
    if pad_right:
        mask[:, -pad_right:] = True
    
    mask3 = np.repeat(mask[:, :, None], padded.shape[2], axis=2)
    masked_pixels = padded[mask3].copy()
    masked_pixels = masked_pixels.reshape(-1, 3)
    np.random.shuffle(masked_pixels)
    padded[mask3] = masked_pixels.reshape(-1)
    padded = np.clip(padded, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(padded if not was_grayscale else padded[:, :, 0])
    return result

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# def generate_samples(ema_model, diffusion, vae, class_labels, num_timesteps, device, dataset, logger):
#     """
#     Generate samples using the EMA model for specified class labels.
    
#     Args:
#         class_labels: tensor of class indices to generate for
#     """
#     torch.set_grad_enabled(False)
    
#     num_samples = len(class_labels)
    
#     # Create noise
#     latent_size = 256 // 8
#     z = torch.randn(num_samples, 4, latent_size, latent_size, device=device)
    
#     # Setup classifier-free guidance
#     z = torch.cat([z, z], 0)
    
#     # Get superclass labels for guidance
#     superclass_labels = torch.tensor([dataset.get_superclass(c.item()) for c in class_labels], device=device)
#     y = torch.cat([class_labels, superclass_labels + dataset.num_classes], 0)
    
#     model_kwargs = dict(y=y, cfg_scale=4.0)
    
#     # Sample
#     samples = diffusion.p_sample_loop(
#         ema_model.forward_with_cfg, z.shape, z, clip_denoised=False, 
#         model_kwargs=model_kwargs, progress=False, device=device
#     )
#     samples, _ = samples.chunk(2, dim=0)
    
#     # Decode from latent space
#     samples = vae.decode(samples / 0.18215).sample
    
#     torch.set_grad_enabled(True)
    
#     return samples

def generate_samples(ema_model, diffusion, vae, class_labels, num_timesteps, device, dataset, logger):
    """
    Generate samples using the EMA model for specified class labels.
    """
    # try/finally: a failure here must NOT leave grad globally disabled, or the next
    # training-loop backward raises "does not require grad".
    torch.set_grad_enabled(False)
    try:
        # Use fp16 for faster inference
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            num_samples = len(class_labels)

            # Create noise
            latent_size = 256 // 8
            z = torch.randn(num_samples, 4, latent_size, latent_size, device=device)

            # Setup classifier-free guidance
            z = torch.cat([z, z], 0)

            # Build the doubled label tensor [conditional ; null]. With ClipEmbedder the null is
            # produced internally by forward_with_cfg (coarse Phylum CLIP), so both halves are the
            # plain class labels. With LabelEmbedder the second half is the superclass-row trick.
            if getattr(ema_model, "use_clip_embedder", False):
                y = torch.cat([class_labels, class_labels], 0)
            else:
                superclass_labels = torch.tensor([dataset.get_superclass(c.item()) for c in class_labels], device=device)
                y = torch.cat([class_labels, superclass_labels + dataset.num_classes], 0)

            model_kwargs = dict(y=y, cfg_scale=4.0)

            # Sample
            samples = diffusion.p_sample_loop(
                ema_model.forward_with_cfg, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)

            # Decode from latent space
            # Match the VAE's own dtype: it may now be fp16 (see --vae-fp32), and autocast
            # does not cast module *weights*, so a fp32 latent here would mismatch.
            samples = vae.decode((samples / 0.18215).to(next(vae.parameters()).dtype)).sample
    finally:
        torch.set_grad_enabled(True)
    
    return samples

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    # Optionally condition on precomputed LoRA-CLIP taxonomy embeddings (ClipEmbedder).
    clip_species = clip_coarse = clip_image_mean = None
    if args.clip_embeddings:
        z = np.load(args.clip_embeddings, allow_pickle=True)
        clip_species = z["clip_emb_species"]   # (C, clip_dim) full-lineage condition
        clip_coarse = z["clip_emb_coarse"]     # (C, clip_dim) Phylum-truncated CFG null
        if args.clip_image:
            if "clip_emb_image_mean" not in z:
                raise SystemExit(
                    f"--clip-image needs image embeddings, but {args.clip_embeddings} has none. "
                    f"Regenerate it with make_clip_embeddings.py --images-embed")
            clip_image_mean = z["clip_emb_image_mean"]   # (C, img_dim) per-class mean
        assert clip_species.shape[0] == args.num_classes, (
            f"clip table has {clip_species.shape[0]} classes but --num-classes={args.num_classes}")
        logger.info(f"Conditioning on CLIP embeddings from {args.clip_embeddings} "
                    f"(model={z['clip_model']}, dim={clip_species.shape[1]})")

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_super_classes=args.num_super_classes,
        clip_species=clip_species,
        clip_coarse=clip_coarse,
        clip_code_dim=args.clip_code_dim,
        clip_image_mean=clip_image_mean,
        clip_image_p_mean=args.clip_image_p_mean,
    )
        # Initialize wandb
    if rank == 0:
        wandb.init(
            project="finediffusion-ifcb",
            name=f"{experiment_index:03d}-{model_string_name}",
            config={
                "model": args.model,
                "image_size": args.image_size,
                "num_classes": args.num_classes,
                "num_super_classes": args.num_super_classes,
                "global_batch_size": args.global_batch_size,
                "epochs": args.epochs,
                "learning_rate": 1e-4,
            },
            dir=experiment_dir
        )

    logger.info("resume_model:")
    logger.info(args.resume)

    logger.info("img_size:"+str(args.image_size))
    
    if args.resume:
        state_dict = find_model(args.resume)
        # Drop the pretrained lookup-table row (incompatible with ClipEmbedder, and re-learned
        # anyway for LabelEmbedder since num_classes differs from the pretrained checkpoint).
        state_dict.pop('y_embedder.embedding_table.weight', None)
        model.load_state_dict(state_dict, strict=False)

    # Parameter-efficient fine-tune: train the conditioning embedder + biases + norms, freeze
    # the rest of the pretrained DiT. For ClipEmbedder this trains the projection MLP and the
    # per-class code (the frozen CLIP tables are buffers, so they never get requires_grad).
    def _is_trainable(name):
        return ('y_embedder.' in name) or ('bias' in name) or ('norm' in name)

    for name, param in model.named_parameters():
        param.requires_grad = _is_trainable(name)

    # for name, param in model.named_parameters():
    #     logger.info(name)
    #     logger.info(param.requires_grad)
    
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # The VAE is FROZEN and inference-only, so fp16 is safe and roughly halves its cost.
    # Measured at batch 32 on an A5000: encode takes 355ms in fp32 vs 207ms in fp16, against
    # ~396ms for the whole DiT forward+backward — i.e. the encode was ~47% of each step, and
    # it was running OUTSIDE the autocast block in full fp32.
    # --vae-fp32 restores the old behaviour if the latents ever look wrong.
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae_dtype = torch.float32 if args.vae_fp32 else torch.float16
    vae = vae.to(dtype=vae_dtype).eval()
    logger.info(f"VAE dtype: {vae_dtype}")
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")


    total_parameters = 0
    for p in model.parameters():
        if p.requires_grad:
            total_parameters += p.numel()

    logger.info(f"DiT Parameters with requires_grad=True: {total_parameters:,}")


    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Defaults for a fresh run (no --checkpoint); overwritten below if resuming a checkpoint.
    start_epoch = 0
    start_step = 0

    # If checkpoint provided, load it to resume training
    if args.checkpoint:
        logger.info(f"Loading training checkpoint from {args.checkpoint}")
        # weights_only=False: the checkpoint stores `args` as an argparse.Namespace, which
        # torch>=2.6 rejects under the new weights_only=True default.
        checkpoint_data = torch.load(args.checkpoint, map_location=lambda storage, loc: storage,
                                     weights_only=False)
        
        model.module.load_state_dict(checkpoint_data['model'])
        ema.load_state_dict(checkpoint_data['ema'])
        opt.load_state_dict(checkpoint_data['opt'])


        for name, param in model.named_parameters():
            param.requires_grad = _is_trainable(name)

        requires_grad(ema, False)

        for name, param in ema.named_parameters():
            logger.info(name)
            logger.info(param.requires_grad)

        # Move optimizer state to device
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logger.info(f"DiT Parameters with requires_grad=True: {total_parameters:,}")

        start_epoch = checkpoint_data.get('epoch', 0)
        start_step = checkpoint_data.get('train_steps', 0)
        logger.info(f"Resumed from epoch {start_epoch}, step {start_step}")

    # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: pad_to_square(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Annotation CSVs live next to the images. Default to the local workstation layout, but
    # allow IFCB_ANNS (or --anns-dir) to relocate the whole set — on the cluster the data sits
    # on a shared volume (/mnt/datasets/...), so hardcoded /scratch paths don't exist there.
    anns_dir = args.anns_dir or os.environ.get(
        "IFCB_ANNS", "/scratch/datasets/other/IFCB_FishNet_Format/anns")
    dataset = IFCBTrainDataset(
        data_path=args.data_path,
        train_csv_path=os.path.join(anns_dir, "ifcb_train.csv"),
        records_csv_path=os.path.join(anns_dir, "ifcb_records.csv"),
        transform=transform,
        # Only load per-image embeddings when the model will actually consume them.
        image_embeddings=args.clip_embeddings if args.clip_image else None,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        # Rebuilding worker processes every epoch is pure overhead here: an epoch is only
        # ~900 steps, and each respawn re-imports torch and re-opens the CSVs.
        persistent_workers=args.num_workers > 0,
        # The transform (pad_to_square: BICUBIC resize + edge-strip tiling + a per-pixel
        # shuffle of the padding) costs ~24ms/image, so a worker sustains only ~41 img/s.
        # Queue several batches ahead so a slow worker does not stall the step.
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    # Automatic Mixed Precision
    scaler = GradScaler()
    # Log dataset info
    stats = dataset.get_stats()
    logger.info(f"Dataset contains {stats['num_images']:,} images from training split ({args.data_path})")
    logger.info(f"Found {stats['num_superclasses']} superclasses (Phyla):")
    # for sc_idx, sc_name in enumerate(stats['superclass_names']):
    #     count = stats['superclass_counts'].get(sc_idx, 0)
    #     logger.info(f"  Superclass {sc_idx}: {sc_name} ({count} images)")
    
    # Set the class-to-superclass mapping in the model's label embedder. Only the
    # LabelEmbedder needs it (its CFG dropout maps a class to its Phylum superclass row).
    # ClipEmbedder carries the coarse (Phylum) null in its own clip_coarse table, so it has
    # no such method — skip in that case.
    if not model.module.use_clip_embedder:
        class_to_superclass_mapping = torch.zeros(dataset.num_classes, dtype=torch.long)
        for class_idx in range(dataset.num_classes):
            superclass_idx = dataset.get_superclass(class_idx)
            class_to_superclass_mapping[class_idx] = superclass_idx
        model.module.y_embedder.set_class_to_superclass_mapping(class_to_superclass_mapping)
    logger.info(f"Dataset contains {stats['num_classes']} classes in {stats['num_superclasses']} superclasses")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = start_step  # Will be 0 if not resuming, or the saved step count if resuming   
    log_steps = 0
    running_loss = 0
    start_time = time()
    # Pre-sample fixed class labels for periodic generation during training
    num_fixed_samples = 16
    fixed_sample_classes = torch.randint(0, dataset.num_classes, (num_fixed_samples,), device=device)
    logger.info(f"Fixed sample classes: {[dataset.get_class_name(c.item()) for c in fixed_sample_classes]}")
    if rank == 0:
        try:
            logger.info(f"Generating samples at step {train_steps}...")
            samples = generate_samples(
                ema, diffusion, vae, fixed_sample_classes,
                num_timesteps=250, device=device, 
                dataset=dataset, logger=logger
            )
            # Create grid and log
            sample_grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
            
            # Log to wandb with class info
            class_names = [dataset.get_class_name(c.item()) for c in fixed_sample_classes[:]]
            wandb.log({
                "samples": wandb.Image(sample_grid, caption=f"{class_names}"),
            }, step=train_steps)
            
            logger.info(f"Sample generation complete")
        except Exception as e:
            logger.warning(f"Failed to generate samples: {e}")
    dist.barrier()
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            # The dataset yields (img, class) or (img, class, image_emb) — the latter only
            # when per-image CLIP conditioning is enabled.
            if len(batch) == 3:
                x, y, img_emb = batch
                img_emb = img_emb.to(device)
            else:
                x, y = batch
                img_emb = None
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Cast the batch to the VAE's dtype, then bring the latent back to fp32 —
                # the diffusion loss and the DiT's own autocast expect fp32 inputs.
                x = vae.encode(x.to(vae_dtype)).latent_dist.sample().mul_(0.18215).float()
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y) if img_emb is None else dict(y=y, image_emb=img_emb)
            
            with autocast():
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                if rank == 0:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/steps_per_sec": steps_per_sec,
                        "train/step": train_steps,
                    })

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Keep only the last N checkpoints to save disk space
                    import glob as glob_module
                    checkpoints = sorted(glob_module.glob(f"{checkpoint_dir}/*.pt"))
                    num_to_keep = 1  # Keep the last checkpoint
                    if len(checkpoints) > num_to_keep:
                        for old_ckpt in checkpoints[:-num_to_keep]:
                            try:
                                os.remove(old_ckpt)
                                logger.info(f"Deleted old checkpoint: {old_ckpt}")
                            except Exception as e:
                                logger.warning(f"Failed to delete {old_ckpt}: {e}")
                
                    # Generate and log samples
                    try:
                        logger.info(f"Generating samples at step {train_steps}...")
                        samples = generate_samples(
                            ema, diffusion, vae, fixed_sample_classes,
                            num_timesteps=250, device=device, 
                            dataset=dataset, logger=logger
                        )
                        # Create grid and log
                        sample_grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
                        
                        # Log to wandb with class info
                        class_names = [dataset.get_class_name(c.item()) for c in fixed_sample_classes[:]]
                        wandb.log({
                            "samples": wandb.Image(sample_grid, caption=f"{class_names}"),
                        }, step=train_steps)
                        
                        logger.info(f"Sample generation complete")
                    except Exception as e:
                        logger.warning(f"Failed to generate samples: {e}")
                dist.barrier()
            

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--resume", type=str, default='DiT-XL-2-256x256.pt',
                        help="Pretrained model to initialize from")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Training checkpoint to resume from (overrides resume)")
    parser.add_argument("--data-path", type=str, default="datasets/train_mini")
    parser.add_argument("--anns-dir", type=str, default=None,
                        help="directory holding ifcb_train.csv / ifcb_records.csv. Defaults to "
                             "$IFCB_ANNS, else the local /scratch layout. Set this on the cluster, "
                             "where the data lives on a shared volume.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=10000)
    parser.add_argument("--num-super-classes", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=3)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--vae-fp32", action="store_true",
                        help="run the frozen VAE in fp32 (the old behaviour). Default is fp16, "
                             "which halves the encode cost — it was ~47%% of each training step.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="dataloader workers PER RANK (so N ranks x this many processes). "
                             "pad_to_square costs ~24ms/image = ~41 img/s per worker, so "
                             "sustaining S steps/s at B per GPU over N GPUs needs about "
                             "S*B*N/41 workers in total.")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="batches each worker queues ahead (torch default 2).")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--clip-embeddings", type=str, default=None,
                        help="Path to precomputed CLIP taxonomy embeddings .npz "
                             "(keys clip_emb_species, clip_emb_coarse). Enables ClipEmbedder "
                             "conditioning instead of the learned lookup table.")
    parser.add_argument("--clip-image", action="store_true",
                        help="also condition on each image's OWN LoRA-CLIP embedding, "
                             "concatenated with the text vector (needs a .npz built with "
                             "make_clip_embeddings.py --images-embed). Captures the "
                             "intra-class variance a class prototype discards.")
    parser.add_argument("--clip-image-p-mean", type=float, default=0.5,
                        help="fraction of TRAINING samples whose image embedding is replaced by "
                             "their class mean. Sampling has no image to encode, so the mean is "
                             "all that is available there; this makes it an in-distribution "
                             "query instead of an unseen centroid. 0 disables the substitution.")
    parser.add_argument("--clip-code-dim", type=int, default=32,
                        help="Dim of the trainable per-class hybrid code in ClipEmbedder. "
                             "0 = PURE CLIP conditioning (no code), which is the right setting "
                             "for comparing text encoders: the code is a free per-class lookup "
                             "table, and with 145 classes even 32 dims can encode identity "
                             "directly and route around CLIP, flattening the ablation for the "
                             "wrong reason. Only safe at 0 when every conditioning string is "
                             "unique — use embeddings built with make_clip_embeddings.py "
                             "--morpho (the plain lineage collides for 36 of 145 classes).")
    args = parser.parse_args()
    main(args)
