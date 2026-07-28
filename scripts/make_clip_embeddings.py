"""Encode the IFCB class taxonomy with a LoRA-CLIP checkpoint -> the .npz `train.py` conditions on.

This is the offline step between the two repos: `hyperbolic-plankton` trains a LoRA-CLIP
(e.g. ranked-dedup r=32), and FineDiffusion conditions on the *text embeddings* that encoder
produces for the 145 IFCB classes. The checkpoint itself is never needed at diffusion-training
time — only this .npz — so this script is the only place the two repos meet.

Reproduces the schema of the existing files (verified against ifcb_rd32_hierarchical_embeddings.npz):
  folder          (145,)      class dir name, the join key
  species_string  (145,)      cumulative lineage, space-joined: "Kingdom Phylum ... Genus species"
  coarse_string   (145,)      "Kingdom Phylum" (the superclass condition)
  clip_emb_species(145, 512)  L2-normalised float32
  clip_emb_coarse (145, 512)  L2-normalised float32
  clip_model      ()          provenance string

  # on the pod, with the LoRA checkpoint staged:
  python scripts/make_clip_embeddings.py \
      --ckpt /mnt/resources/ckpts/bioclip_lora_best.pt \
      --records /mnt/datasets/ifcb_finediffusion/anns/ifcb_records.csv \
      --images  /mnt/datasets/ifcb_finediffusion/Images \
      --out ifcb_rd32_hierarchical_embeddings.npz \
      --name rd_r32_ranked_dedup

Needs the `hyperbolic_plankton` package importable (for HyperbolicCLIP + apply_lora), since the
checkpoint stores LoRA adapters over an open_clip backbone, not a standalone model.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "species"]


def build_strings(records_csv: str, images_dir: str):
    """Per-class (folder, species_string, coarse_string), ordered by folder name.

    Order matters: it must match the class indices `train.py` derives from
    `sorted(data_path.iterdir())`, which is what IFCBTrainDataset uses.
    """
    df = pd.read_csv(records_csv)
    folders = sorted(d.name for d in os.scandir(images_dir) if d.is_dir())

    first = df.drop_duplicates("Folder").set_index("Folder")
    rows = []
    for f in folders:
        if f not in first.index:
            sys.exit(f"folder {f!r} has no row in {records_csv}")
        r = first.loc[f]
        # Cumulative lineage, skipping blanks — a truncated lineage simply stops early
        # (matches the existing files, e.g. "Chromista Radiozoa Acantharia").
        parts = [str(r[c]).strip() for c in RANKS
                 if c in r.index and pd.notna(r[c]) and str(r[c]).strip() != ""]
        coarse = [str(r[c]).strip() for c in ("Kingdom", "Phylum")
                  if c in r.index and pd.notna(r[c]) and str(r[c]).strip() != ""]
        # The non-taxonomic classes (Bead, Detritus, Faecal_pellet, Undet_small) have an
        # entirely empty lineage. Fall back to the folder name, as the original embeddings
        # did — an empty string would encode all four to the same meaningless vector and
        # make them mutually indistinguishable as conditions.
        rows.append((f, " ".join(parts) or f, " ".join(coarse) or f))
    return rows


def load_encoder(ckpt_path: str, device: str):
    """The LoRA-CLIP text encoder from a hyperbolic-plankton checkpoint."""
    from hyperbolic_plankton.lora import apply_lora
    from hyperbolic_plankton.model import HyperbolicCLIP

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    print(f"checkpoint: contrastive={a.get('contrastive')} lora_r={a.get('lora_r')} "
          f"alpha={a.get('lora_alpha')} geometry={a.get('geometry')} no_proj={a.get('no_proj')}")
    model = HyperbolicCLIP(backbone=a["backbone"], use_proj=not a["no_proj"])
    apply_lora(model, r=a["lora_r"], alpha=a["lora_alpha"],
               adapt_visual_blocks=a["lora_visual_blocks"],
               adapt_text_blocks=a["lora_text_blocks"],
               reinit_final_ln=not a["no_reinit_final_ln"],
               include_mlp=a["lora_mlp"])
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    n_lora = sum(1 for k in ck["model"] if "lora" in k.lower())
    print(f"  loaded {n_lora} LoRA tensors ({len(missing)} frozen-backbone keys left at "
          f"pretrained values, {len(unexpected)} unexpected)")
    if n_lora == 0:
        sys.exit("checkpoint contains no LoRA tensors — wrong file?")
    if unexpected:
        sys.exit(f"unexpected keys in checkpoint: {unexpected[:5]}")
    # project=False -> raw CLIP text features, which is what the euclidean/no_proj runs use
    # and what the existing 512-dim npz files contain.
    return model.eval().to(device)


@torch.no_grad()
def encode(model, strings, device, batch=128):
    out = []
    for i in range(0, len(strings), batch):
        f = model.encode_text(list(strings[i:i + batch]), project=False).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="hyperbolic-plankton LoRA checkpoint (.pt)")
    ap.add_argument("--records", default="/scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_records.csv")
    ap.add_argument("--images", default="/scratch/datasets/other/IFCB_FishNet_Format/Images")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--name", default=None,
                    help="provenance string stored as clip_model (default: the ckpt basename)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the strings, load no model, write nothing")
    args = ap.parse_args()

    rows = build_strings(args.records, args.images)
    folders = [r[0] for r in rows]
    species = [r[1] for r in rows]
    coarse = [r[2] for r in rows]
    print(f"{len(rows)} classes from {args.images}")
    for f, s, c in rows[:3]:
        print(f"  {f:32s} species={s!r}\n  {'':32s} coarse ={c!r}")

    if args.dry_run:
        print("\n(dry run - no model loaded, nothing written)")
        return

    model = load_encoder(args.ckpt, args.device)
    emb_s = encode(model, species, args.device)
    emb_c = encode(model, coarse, args.device)
    print(f"encoded: species {emb_s.shape}, coarse {emb_c.shape}")

    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists — refusing to overwrite. Delete it or pick another --out.")
    np.savez(args.out,
             folder=np.array(folders, dtype=object),
             species_string=np.array(species),
             coarse_string=np.array(coarse),
             clip_emb_species=emb_s,
             clip_emb_coarse=emb_c,
             clip_model=args.name or os.path.basename(args.ckpt).replace(".pt", ""))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
