#!/usr/bin/env python3
"""Visualise CFG-Rejection: same conditioning image, N candidates, ranked by ASD.

Layout per row (one row per conditioning image), matching sample_peek/cond_*.png:

    [ real conditioning image ] | [ cand 0 ] [ cand 1 ] ... [ cand N-1 ]

Candidates are shown in ASD order, highest first, with each one's ASD printed. The leftmost
candidate is what CFG-Rejection would KEEP; the rightmost is what it would discard. Candidate
0 in generation order (i.e. what plain sampling returns) is marked "plain" so you can see
whether selection actually changed anything for that slot.

The point of looking: on a 3-slot calibration ASD appeared to track organism SIZE/CONTRAST
rather than validity, because our generator produces few outright failures. If that is what is
happening, the kept column will simply be the biggest/darkest specimen rather than the best
one, and selection would shift the distribution (probably hurting FID) instead of improving
quality. This grid makes that visible before committing GPU hours.

  python peek_cfg_rejection.py --ckpt <text+image ckpt> --npz <npz> \
      --classes Tripos_lineatus Lessardia Bacillaria \
      --n_candidates 4 --n_rows 4 --out sample_peek/cfgrej_
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw

from generate_synthetic_dataset import load_models
from generate_with_cfg_rejection import asd_prefix, sample_full

REAL_LIB = "/scratch/datasets/DEAL/Plankton/IFCB/IFCB_Annotated_Training_Library_FINAL"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--data_path", default=REAL_LIB)
    ap.add_argument("--train_csv",
                    default="/scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--n_rows", type=int, default=4, help="conditioning images per class")
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--num_sampling_steps", type=int, default=250)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="sample_peek/cfgrej_")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    classes = sorted(d.name for d in Path(args.data_path).iterdir() if d.is_dir())
    train = pd.read_csv(args.train_csv)
    per_class = train.groupby("Folder")["image"].apply(list).to_dict()

    z = np.load(args.npz, allow_pickle=True)
    rows_idx = {str(k): i for i, k in enumerate(z["clip_emb_image_keys"])}
    emb_all = z["clip_emb_image"].astype(np.float32)

    _l = load_models(args.ckpt, args.device, len(classes), 12, 256,
                     args.num_sampling_steps, clip_embeddings=args.npz)
    model, vae, diffusion = _l[0], _l[1], _l[2]
    if not hasattr(model.y_embedder, "clip_image_mean"):
        raise SystemExit("checkpoint has no image conditioning")

    W, PAD = 256, 6
    for cls in args.classes:
        if cls not in per_class:
            logging.warning(f"{cls} not in train CSV, skipping")
            continue
        srcs = sorted(per_class[cls])[: args.n_rows]
        ci = classes.index(cls)
        N = args.n_candidates

        grid = Image.new("RGB", (W * (N + 1) + PAD * (N + 1), W * len(srcs) + PAD * len(srcs)),
                         (32, 32, 32))
        dr = ImageDraw.Draw(grid)

        for r, src in enumerate(srcs):
            key = f"{cls}/{src}"
            if key not in rows_idx:
                logging.warning(f"no embedding for {key}")
                continue
            e = torch.from_numpy(emb_all[rows_idx[key]][None]).to(args.device)

            # the real conditioning image, left column
            real = Image.open(Path(args.data_path) / cls / src).convert("RGB")
            real.thumbnail((W, W))
            grid.paste(real, (0, r * (W + PAD) + (W - real.height) // 2))

            cand_z = []
            for j in range(N):
                torch.manual_seed(args.seed + r * 97 + j * 1_000_003)
                cand_z.append(torch.randn(1, 4, 32, 32, device=args.device))
            y = torch.full((1,), ci, dtype=torch.long, device=args.device)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                scores = [float(asd_prefix(model, diffusion, cz, y, e, args.cfg_scale,
                                           args.tau, args.device)[0]) for cz in cand_z]
                # full denoise every candidate -- this is a diagnostic, not the fast path
                imgs = [sample_full(model, vae, diffusion, cz, y, e,
                                    args.cfg_scale, args.device)[0] for cz in cand_z]

            order = sorted(range(N), key=lambda j: -scores[j])   # highest ASD first = kept
            for c, j in enumerate(order):
                pil = torchvision.transforms.functional.to_pil_image(
                    imgs[j].clamp(-1, 1).add(1).div(2))
                x = (c + 1) * (W + PAD)
                grid.paste(pil, (x, r * (W + PAD)))
                tag = f"ASD {scores[j]:.3f}"
                if j == 0:
                    tag += "  (plain)"
                if c == 0:
                    tag += "  <-KEPT"
                dr.text((x + 5, r * (W + PAD) + 5), tag, fill=(255, 230, 0))
            logging.info(f"{cls} row {r}: ASD {[round(s,3) for s in scores]} "
                         f"kept cand {order[0]}")

        out = Path(f"{args.out}{cls}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        grid.save(out)
        logging.info(f"wrote {out}")


if __name__ == "__main__":
    main()
