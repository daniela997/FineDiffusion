# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample new images from IFCB FineDiffusion model.
Generates 10 samples per class (145 classes = 1,450 total samples).
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from ifcb_dataset import IFCBTrainDataset
import argparse
import pandas as pd

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset to get class-to-superclass mapping
    print("Loading dataset for class mappings...")
    dataset = IFCBTrainDataset(
        data_path=args.data_path,
        train_csv_path="/scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv",
        records_csv_path="/scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_records.csv",
        transform=None
    )
    
    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_super_classes=args.num_super_classes
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Setup diffusion and VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Set class-to-superclass mapping in model
    class_to_superclass_mapping = torch.zeros(dataset.num_classes, dtype=torch.long)
    for class_idx in range(dataset.num_classes):
        superclass_idx = dataset.get_superclass(class_idx)
        class_to_superclass_mapping[class_idx] = superclass_idx
    model.y_embedder.set_class_to_superclass_mapping(class_to_superclass_mapping)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample for each class
    print(f"Generating {args.samples_per_class} samples for each of {args.num_classes} classes...")
    
    with torch.no_grad():
        for class_idx in range(args.num_classes):
            class_name = dataset.get_class_name(class_idx)
            superclass_name = dataset.get_superclass_name(dataset.get_superclass(class_idx))
            
            print(f"[{class_idx+1}/{args.num_classes}] {class_name} ({superclass_name})")
            
            # Generate samples_per_class images
            for sample_num in range(args.samples_per_class):
                torch.manual_seed(args.seed + class_idx * 1000 + sample_num)
                torch.set_grad_enabled(False)
                
                # Create noise
                z = torch.randn(1, 4, latent_size, latent_size, device=device)
                y = torch.tensor([class_idx], device=device)
                
                # Setup classifier-free guidance
                z = torch.cat([z, z], 0)
                y_super = torch.tensor([dataset.get_superclass(class_idx) + args.num_classes], device=device)
                y = torch.cat([y, y_super], 0)
                
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                
                # Sample image
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
                
                # Save image in output_dir/class_name/subcategory_B/
                class_dir = os.path.join(output_dir, class_name, "subcategory_B")
                os.makedirs(class_dir, exist_ok=True)
                
                file_path = os.path.join(class_dir, f"sample_{sample_num:02d}.png")
                save_image(samples, file_path, normalize=True, value_range=(-1, 1))
    
    print(f"Done! Samples saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=145)
    parser.add_argument("--num-super-classes", type=int, default=12)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained IFCB FineDiffusion checkpoint")
    parser.add_argument("--data-path", type=str, default="/scratch/datasets/other/IFCB_FishNet_Format/Images",
                        help="Path to IFCB dataset (for class mappings)")
    parser.add_argument("--output-dir", type=str, default="ifcb_samples",
                        help="Output directory for generated samples")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--samples-per-class", type=int, default=10,
                        help="Number of samples to generate per class")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)