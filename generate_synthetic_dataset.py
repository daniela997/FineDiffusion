#!/usr/bin/env python3
"""
Generate synthetic plankton images using FineDiffusion model.
Usage:
    python generate_synthetic_dataset.py \
        --ckpt /path/to/checkpoint.pt \
        --train_csv /scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv \
        --data_path /scratch/datasets/other/IFCB_FishNet_Format/Images \
        --output_dir /scratch/datasets/other/IFCB_FishNet_Format/FineDiffusion_synthetic/ \
        --batch_size 16 \
        --resume
"""

import os
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
from ifcb_dataset import IFCBTrainDataset


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_models(ckpt_path, device, num_classes, num_super_classes, image_size, num_sampling_steps,
                clip_embeddings=None):
    """Load FineDiffusion model and VAE.

    Handles BOTH conditioning types. A ClipEmbedder checkpoint carries clip_species/clip_coarse
    buffers and no y_embedder.embedding_table, so it cannot be loaded into a LabelEmbedder model
    (and vice versa). The training args are stored in the checkpoint, so the conditioning is
    detected from there rather than having to be re-specified; --clip-embeddings overrides the
    recorded npz path if the file has since moved.
    """
    logging.info("Loading FineDiffusion model...")

    # weights_only=False: our own checkpoints store `args` as an argparse.Namespace,
    # which torch>=2.6 rejects under the new weights_only=True default.
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage,
                            weights_only=False)
    state = checkpoint['ema'] if isinstance(checkpoint, dict) and 'ema' in checkpoint else checkpoint

    # Detect the conditioning from the state dict itself (authoritative), falling back to the
    # recorded args for the npz path and code dim.
    uses_clip = any(k.startswith('y_embedder.clip_') for k in state)
    ck_args = checkpoint.get('args') if isinstance(checkpoint, dict) else None
    clip_species = clip_coarse = clip_image_mean = None
    clip_code_dim = 0
    if uses_clip:
        npz_path = clip_embeddings or (getattr(ck_args, 'clip_embeddings', None) if ck_args else None)
        if not npz_path or not os.path.exists(npz_path):
            raise SystemExit(
                f"checkpoint is CLIP-conditioned but its embeddings npz was not found "
                f"({npz_path!r}). Pass --clip-embeddings with the .npz used for training.")
        z = np.load(npz_path, allow_pickle=True)
        clip_species, clip_coarse = z['clip_emb_species'], z['clip_emb_coarse']
        # The trainable per-class code's width must match the checkpoint, not the args, so read
        # it off the saved tensor when present (code_dim=0 means there is no code at all).
        code_w = state.get('y_embedder.code.weight')
        clip_code_dim = int(code_w.shape[1]) if code_w is not None else 0
        # Image conditioning: take the per-class mean table straight from the checkpoint's
        # own buffer rather than the npz. It is authoritative (the npz may lack the key, or
        # be a different file), and its presence is exactly what distinguishes the
        # text+image variant from text-only.
        clip_image_mean = state.get('y_embedder.clip_image_mean')
        if clip_image_mean is not None:
            clip_image_mean = clip_image_mean.cpu().numpy()
        logging.info(f"CLIP conditioning: {npz_path} (model={z['clip_model']}, "
                     f"dim={clip_species.shape[1]}, code_dim={clip_code_dim}, "
                     f"image={'yes' if clip_image_mean is not None else 'no'})")
    else:
        logging.info("Label conditioning (learned embedding table)")

    latent_size = image_size // 8
    model = DiT_models["DiT-XL/2"](
        input_size=latent_size,
        num_classes=num_classes,
        num_super_classes=num_super_classes,
        clip_species=clip_species,
        clip_coarse=clip_coarse,
        clip_code_dim=clip_code_dim,
        clip_image_mean=clip_image_mean,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    # Create diffusion
    diffusion = create_diffusion(str(num_sampling_steps))
    
    return model, vae, diffusion


def generate_images_for_class(
    class_idx,
    ml_class,
    target_count,
    model,
    vae,
    diffusion,
    dataset,
    device,
    output_dir,
    batch_size,
    num_classes,
    num_super_classes,
    cfg_scale,
    num_sampling_steps,
    resume,
    seed
):
    """Generate target_count images for a single class."""
    class_dir = output_dir / ml_class
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing images
    existing_images = list(class_dir.glob("*.png"))
    if resume and len(existing_images) >= target_count:
        logging.info(f"Skipping {ml_class} - already has {len(existing_images)} images")
        return
    
    start_idx = len(existing_images) if resume else 0
    if not resume and existing_images:
        logging.info(f"Clearing {len(existing_images)} existing images for {ml_class}")
        for img in existing_images:
            img.unlink()
        start_idx = 0
    
    logging.info(f"Generating {target_count - start_idx} images for {ml_class}")
    
    # Get superclass for this class
    superclass_idx = dataset.get_superclass(class_idx)
    
    samples_generated = start_idx
    pbar = tqdm(total=target_count - start_idx, desc=f"{ml_class[:30]:30s}")
    
    with torch.no_grad():
        while samples_generated < target_count:
            current_batch_size = min(batch_size, target_count - samples_generated)
            
            # Set seed for reproducibility
            torch.manual_seed(seed + samples_generated)
            
            # Create noise
            latent_size = 32  # 256 // 8
            z = torch.randn(current_batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_idx] * current_batch_size, device=device, dtype=torch.long)
            
            # Setup classifier-free guidance. With ClipEmbedder the null half is produced
            # INTERNALLY by forward_with_cfg (it swaps the species CLIP vector for the coarse
            # Phylum one), so both halves are the plain class labels. Only LabelEmbedder needs
            # the "superclass row = num_classes + superclass_idx" trick.
            z = torch.cat([z, z], 0)
            if getattr(model, "use_clip_embedder", False):
                y = torch.cat([y, y], 0)
            else:
                y_super = torch.tensor([superclass_idx + num_classes] * current_batch_size,
                                       device=device, dtype=torch.long)
                y = torch.cat([y, y_super], 0)
            
            model_kwargs = dict(y=y, cfg_scale=cfg_scale)
            
            # Use autocast for mixed precision (like in training)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Sample images
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
            
            # Save images
            for i, sample in enumerate(samples):
                img_idx = samples_generated + i
                save_path = class_dir / f"synthetic_{img_idx:05d}.png"
                torchvision.transforms.functional.to_pil_image(
                    sample.clamp(-1, 1).add(1).div(2)
                ).save(save_path)
            
            samples_generated += current_batch_size
            pbar.update(current_batch_size)
    
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic plankton images with FineDiffusion")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to FineDiffusion checkpoint")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--data_path", type=str, required=True, help="Path to IFCB images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--resume", action="store_true", help="Resume from existing images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--num_sampling_steps", type=int, default=250, help="Number of sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--clip_embeddings", "--clip-embeddings", type=str, default=None,
                        help="Conditioning .npz for a CLIP-conditioned checkpoint. Only needed "
                             "if the path recorded in the checkpoint's args has moved; the "
                             "conditioning type itself is detected from the checkpoint.")
    args = parser.parse_args()
    
    setup_logging()
    
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device)
    
    # Load dataset for class mappings
    dataset = IFCBTrainDataset(
        data_path=args.data_path,
        train_csv_path=args.train_csv,
        records_csv_path=args.train_csv.replace("ifcb_train.csv", "ifcb_records.csv"),
        transform=None
    )
    
    # Set class-to-superclass mapping
    class_to_superclass_mapping = torch.zeros(dataset.num_classes, dtype=torch.long)
    for class_idx in range(dataset.num_classes):
        class_to_superclass_mapping[class_idx] = dataset.get_superclass(class_idx)
    
    # Load model
    model, vae, diffusion = load_models(
        args.ckpt, device, 
        dataset.num_classes, dataset.num_superclasses,
        256, args.num_sampling_steps,
        clip_embeddings=args.clip_embeddings,
    )
    # ClipEmbedder carries the coarse (Phylum) null in its own clip_coarse table and has no
    # class->superclass mapping to set.
    if not getattr(model, "use_clip_embedder", False):
        model.y_embedder.set_class_to_superclass_mapping(class_to_superclass_mapping)
    
    # Load class counts
    train_df = pd.read_csv(args.train_csv)
    class_counts = train_df['Folder'].value_counts().to_dict()
    logging.info(f"Loaded {len(class_counts)} classes from training set")
    
    # Shard the class list
    all_classes = sorted(class_counts.keys())
    shard_classes = [c for i, c in enumerate(all_classes) if i % args.num_shards == args.shard]
    class_counts = {k: v for k, v in class_counts.items() if k in shard_classes}
    
    logging.info(f"Shard {args.shard}/{args.num_shards}: processing {len(class_counts)} classes")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate for each class
    total_classes = len(class_counts)
    for idx, (ml_class, count) in enumerate(class_counts.items(), 1):
        class_idx = dataset.class_names.index(ml_class) if ml_class in dataset.class_names else None
        if class_idx is None:
            logging.warning(f"[{idx}/{total_classes}] Skipping {ml_class} - not found in dataset")
            continue
        
        logging.info(f"[{idx}/{total_classes}] Processing {ml_class} ({count} images)")
        
        try:
            generate_images_for_class(
                class_idx=class_idx,
                ml_class=ml_class,
                target_count=count,
                model=model,
                vae=vae,
                diffusion=diffusion,
                dataset=dataset,
                device=device,
                output_dir=output_dir,
                batch_size=args.batch_size,
                num_classes=dataset.num_classes,
                num_super_classes=dataset.num_superclasses,
                cfg_scale=args.cfg_scale,
                num_sampling_steps=args.num_sampling_steps,
                resume=args.resume,
                seed=args.seed
            )
        except Exception as e:
            logging.error(f"Error generating {ml_class}: {e}")
            continue
        
        # Clear cache periodically
        if idx % 10 == 0:
            torch.cuda.empty_cache()
    
    logging.info("Generation complete!")


if __name__ == "__main__":
    main()