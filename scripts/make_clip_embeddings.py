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


def build_strings(records_csv: str, images_dir: str, morpho: bool = False):
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
        # A class with NO lineage at all already falls back to its folder name, which is
        # fully distinguishing — appending a suffix would just repeat one of its own tokens
        # ("Faecal_pellet pellet"), so morpho only applies to real lineages.
        species = " ".join(parts)
        if morpho and species:
            species = _append_morpho(species, f)
        species = species or f
        rows.append((f, species, " ".join(coarse) or f))
    return rows


def _append_morpho(lineage: str, folder: str) -> str:
    """Append the folder's distinguishing suffix to the lineage string.

    25% of IFCB classes (36/145) share a lineage string with another class, so on the plain
    lineage their CLIP vectors are identical and the classes are literally unconditionable —
    e.g. Cerautulina_pelagica_chain vs _single_double are visually distinct morphologies with
    the same taxonomy. The suffix is whatever the folder name carries beyond the taxonomy:
    tokenise the folder, drop the tokens already present in the lineage, keep the rest.

      Cerautulina_pelagica_chain  -> "... Cerataulina pelagica chain"
      Chaetoceros_morphotype1     -> "... Chaetoceros morphotype1"
      Flagellate_clump            -> "Protozoa Flagellates clump"

    Matching by token (not by prefix length) is what makes the Flagellates work: their
    lineage stops at Phylum, so there is no species token to strip, yet "Flagellate" is
    already represented by "Flagellates".

    A folder token is dropped when it is *near* a lineage token, not only when equal: the
    source data spells the same taxon differently in the folder and the records CSV
    (Cerautulina/Cerataulina, Dinobyron/Dinobryon, Eutriptiella/Eutreptiella, Asterompalus/
    Asteromphalus). Appending those would duplicate the genus as if it were a morphotype,
    so they are treated as redundant rather than distinguishing.
    """
    have = {t.lower() for t in lineage.replace("-", " ").split()}
    extra = []
    for tok in folder.replace("-", "_").split("_"):
        t = tok.lower()
        if not t or t in have:
            continue
        if any(_near(t, h) for h in have):
            continue
        extra.append(tok)
    return " ".join([lineage, *extra]) if extra else lineage


def _near(a: str, b: str) -> bool:
    """True when two tokens denote the same taxon despite differing spelling.

    Prefix relation (Flagellate/Flagellates, pellet/pellets) or a small edit distance on
    long tokens (Cerautulina/Cerataulina). Deliberately conservative: real morphotype
    markers ('chain', 'single', 'clump', 'morphotype1') are short and share no stem with
    any lineage token, so they survive.
    """
    if len(a) >= 5 and len(b) >= 5 and (a.startswith(b) or b.startswith(a)):
        return True
    if len(a) < 6 or len(b) < 6 or abs(len(a) - len(b)) > 2:
        return False
    # Levenshtein, capped — same-length-ish long tokens differing by <=2 edits are the
    # misspelling case, not a distinct morphotype.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 2


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


@torch.no_grad()
def encode_images(model, images_dir, folders, device, batch=256, workers=8):
    """Per-image LoRA-CLIP embeddings for every training image, plus per-class means.

    Returns (keys, emb, class_mean) where `keys` is a list of "<Folder>/<file>.png" in the
    same order as `emb` [N, D], and `class_mean` is [C, D] aligned to `folders`.

    Per-image (not just per-class-mean) because the model is trained on each image's OWN
    embedding: mean-pooling would discard the intra-class variance that distinguishes e.g.
    Chaetoceros_didymus chains from singles. The mean is stored alongside because sampling
    has no image to encode, and because training substitutes it for a fraction of samples
    (see ClipEmbedder's p_mean) so that it is an in-distribution query at generation time.
    """
    import torch.utils.data as tud
    from PIL import Image

    pre = model.preprocess
    paths, keys = [], []
    for f in folders:
        d = os.path.join(images_dir, f)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".png"):
                paths.append(os.path.join(d, fn))
                keys.append(f"{f}/{fn}")

    class _DS(tud.Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i): return pre(Image.open(paths[i]).convert("RGB"))

    dl = tud.DataLoader(_DS(), batch_size=batch, num_workers=workers, pin_memory=True)
    out = []
    for i, px in enumerate(dl):
        f = model.encode_image(px.to(device), project=False).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        if i % 20 == 0:
            print(f"  encoded {min((i + 1) * batch, len(paths))}/{len(paths)}", flush=True)
    emb = torch.cat(out).numpy().astype(np.float32)

    idx = {f: i for i, f in enumerate(folders)}
    cls = np.array([idx[k.split("/")[0]] for k in keys])
    mean = np.zeros((len(folders), emb.shape[1]), dtype=np.float32)
    for c in range(len(folders)):
        m = emb[cls == c].mean(0)
        mean[c] = m / (np.linalg.norm(m) + 1e-8)
    return keys, emb, mean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="hyperbolic-plankton LoRA checkpoint (.pt)")
    ap.add_argument("--records", default="/scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_records.csv")
    ap.add_argument("--images", default="/scratch/datasets/other/IFCB_FishNet_Format/Images")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--morpho", action="store_true",
                    help="append each folder's distinguishing suffix to the lineage string "
                         "(e.g. '... Cerataulina pelagica chain'). Without it, 36 of 145 "
                         "classes share a string with another class and are indistinguishable "
                         "as conditions. Use this unless you specifically want plain lineages.")
    ap.add_argument("--name", default=None,
                    help="provenance string stored as clip_model (default: the ckpt basename)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--images-embed", action="store_true",
                    help="also encode every training image with the LoRA IMAGE tower and store "
                         "per-image embeddings + per-class means (clip_emb_image_* keys). Needed "
                         "for the text+image conditioning arm; ~10 min on a GPU for 74k images.")
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--embed-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the strings, load no model, write nothing")
    args = ap.parse_args()

    rows = build_strings(args.records, args.images, morpho=args.morpho)
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
    tables = dict(
        folder=np.array(folders, dtype=object),
        species_string=np.array(species),
        coarse_string=np.array(coarse),
        clip_emb_species=emb_s,
        clip_emb_coarse=emb_c,
        clip_model=args.name or os.path.basename(args.ckpt).replace(".pt", ""),
    )
    if args.images_embed:
        print(f"encoding images with the LoRA image tower ...")
        keys, img_emb, img_mean = encode_images(model, args.images, folders, args.device,
                                                batch=args.embed_batch, workers=args.embed_workers)
        tables.update(
            clip_emb_image=img_emb.astype(np.float16),   # [N, D] per-image; fp16 halves the file
            clip_emb_image_keys=np.array(keys),          # "<Folder>/<file>.png", aligned to rows
            clip_emb_image_mean=img_mean,                # [C, D] per-class mean, fp32
        )
        print(f"  per-image {img_emb.shape} + class-mean {img_mean.shape}")

    np.savez(args.out, **tables)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
