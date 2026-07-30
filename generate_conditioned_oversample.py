#!/usr/bin/env python3
"""Generate rare-class top-ups conditioned on the real images naive oversampling would copy.

`syke-pic`'s oversample_class walks a class's training images in order, wrapping around,
appending duplicates until the class reaches `until` (100):

    while len(x) + len(over_x) < until:
        over_x.append(x[i]); i = (i + 1) % len(x)

We follow the same schedule, but instead of copying image i we generate a new image
conditioned on image i's own LoRA-CLIP embedding. Each duplicate therefore becomes a
variation of that specific specimen rather than a repeat of it, while the class reaches the
same size and draws on the same source images in the same order.

This requires a checkpoint trained with --clip-image: the model must have learned
p(x | image embedding), which is what the p_mean training substitution provides.

The traversal operates on the TRAIN SPLIT, not on ifcb_train.csv or the unsplit library,
because that is what apply_oversampling sees (62 classes below 100, 3652 top-ups, versus
53/3109 for the CSV).

  python generate_conditioned_oversample.py \
      --ckpt <text+image checkpoint> \
      --npz  <npz built with make_clip_embeddings.py --images-embed> \
      --output_dir .../oversample_conditioned \
      --shard 0 --num_shards 2 --device cuda:0
"""

import argparse
import io
import logging
import os
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from generate_synthetic_dataset import load_models

DINO_SCRIPTS = "/home/daniela/mine/dino/dino_classification/scripts"


def train_split_paths(dataset_path, records_csv, random_seed=24, schedule_json=None):
    """The training split exactly as the classifier pipeline derives it.

    `schedule_json` supplies a precomputed {schedule, train_counts} instead, for machines
    without dino_classification (the pods). The schedule is deterministic given the split, so
    shipping it is equivalent to recomputing it -- and it keeps the two from ever diverging.
    Note the JSON stores the SCHEDULE (which sources to condition on, in cyclic order), so it
    is used directly rather than re-derived from per-class file lists.
    """
    sys.path.insert(0, DINO_SCRIPTS)
    from data import load_and_split_dataset

    rec = pd.read_csv(records_csv)
    with redirect_stdout(io.StringIO()):          # the loader is chatty
        # val_split=0.1 -- what prepare_training (and so every published arm) actually ran at.
        # At 0.2 the split is 47,477 and the top-up is 3,652 across 62 classes; at 0.1 it is
        # 52,749 and 3,388 across 58. Getting this wrong does not just change the count: the
        # cyclic traversal wraps at different points, so 46 of 58 classes condition on a
        # DIFFERENT set of source images.
        d = load_and_split_dataset(dataset_path, 0.6, 0.1, 0.2, random_seed=random_seed,
                                   exclude=["Unclassifiable"], records_df=rec)
    by_class = defaultdict(list)
    for p in d["train_paths"]:
        by_class[Path(p).parent.name].append(Path(p).name)
    # sorted so the traversal is deterministic and reproducible
    return {c: sorted(v) for c, v in by_class.items()}


def cyclic_schedule(files, until):
    """The images oversample_class would duplicate, in the order it would duplicate them."""
    if len(files) >= until:
        return []
    out, i = [], 0
    while len(files) + len(out) < until:
        out.append(files[i])
        i = (i + 1) % len(files)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True, help="npz with clip_emb_image + keys")
    ap.add_argument("--dataset_path",
                    default="/scratch/datasets/DEAL/Plankton/IFCB/IFCB_Annotated_Training_Library_FINAL")
    ap.add_argument("--records_csv",
                    default="/scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_records.csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--until", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_sampling_steps", type=int, default=250)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--schedule_json", default=None,
                    help="precomputed {schedule: {class: [src, ...]}} for machines without "
                         "dino_classification; the schedule is deterministic given the split")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule, generate nothing")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    # Class indices must span the FULL label set the model was trained on, independent of which
    # classes need top-up, so they come from the image tree rather than from the schedule.
    classes = sorted(d.name for d in Path(args.dataset_path).iterdir() if d.is_dir())
    if args.schedule_json:
        import json
        schedule = json.load(open(args.schedule_json))["schedule"]
        logging.info(f"schedule from {args.schedule_json}")
    else:
        by_class = train_split_paths(args.dataset_path, args.records_csv)
        schedule = {c: cyclic_schedule(by_class[c], args.until) for c in classes if c in by_class}
    # real-image counts are only for logging, and are unavailable from the schedule alone
    n_real = {c: len(v) for c, v in by_class.items()} if not args.schedule_json else {}
    rare = {c: s for c, s in schedule.items() if s}
    logging.info(f"{len(rare)} classes below {args.until}; "
                 f"{sum(len(s) for s in rare.values())} images to generate")

    mine = sorted(rare)[args.shard::args.num_shards]
    logging.info(f"shard {args.shard}/{args.num_shards}: {len(mine)} classes")

    if args.dry_run:
        for c in mine[:10]:
            logging.info(f"  {c}: {n_real.get(c, '?')} real -> +{len(schedule[c])} "
                         f"(first: {schedule[c][:3]})")
        return

    z = np.load(args.npz, allow_pickle=True)
    if "clip_emb_image" not in z:
        sys.exit(f"{args.npz} has no per-image embeddings; rebuild with --images-embed")
    rows = {str(k): i for i, k in enumerate(z["clip_emb_image_keys"])}
    emb = z["clip_emb_image"].astype(np.float32)

    # load_models returns 3 values in older revisions, 4 in newer ones; accept either.
    _l = load_models(args.ckpt, args.device, len(classes), 12, 256,
                     args.num_sampling_steps, clip_embeddings=args.npz)
    model, vae, diffusion = _l[0], _l[1], _l[2]
    if not getattr(model, "use_clip_embedder", False):
        sys.exit("checkpoint is not CLIP-conditioned")
    if not hasattr(model.y_embedder, "clip_image_mean"):
        sys.exit("checkpoint has no image conditioning; needs one trained with --clip-image")

    for n, cls in enumerate(mine, 1):
        todo = schedule[cls]
        out_dir = Path(args.output_dir) / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        done = len(list(out_dir.glob("*.png"))) if args.resume else 0
        if done >= len(todo):
            logging.info(f"[{n}/{len(mine)}] {cls}: complete ({done})")
            continue
        logging.info(f"[{n}/{len(mine)}] {cls}: {n_real.get(cls, '?')} real -> "
                     f"generating {len(todo) - done}")

        ci = classes.index(cls)
        i = done
        while i < len(todo):
            bs = min(args.batch_size, len(todo) - i)
            # One conditioning embedding per output, following the cyclic order.
            e = []
            for j in range(bs):
                key = f"{cls}/{todo[i + j]}"
                if key not in rows:
                    sys.exit(f"no embedding for {key}")
                e.append(emb[rows[key]])
            e = torch.from_numpy(np.stack(e)).to(args.device)

            torch.manual_seed(args.seed + i)
            zn = torch.randn(bs, 4, 32, 32, device=args.device)
            y = torch.full((bs,), ci, dtype=torch.long, device=args.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                s = diffusion.p_sample_loop(
                    model.forward_with_cfg, (2 * bs, 4, 32, 32), torch.cat([zn, zn], 0),
                    clip_denoised=False,
                    model_kwargs=dict(y=torch.cat([y, y], 0), cfg_scale=args.cfg_scale,
                                      image_emb=torch.cat([e, e], 0)),
                    progress=False, device=args.device)
                s, _ = s.chunk(2, 0)
                imgs = vae.decode((s / 0.18215).to(next(vae.parameters()).dtype)).sample

            for j in range(bs):
                # Name records the source image, so the correspondence to what naive
                # oversampling would have copied stays inspectable.
                src = Path(todo[i + j]).stem
                p = out_dir / f"cond_{i + j:05d}_{src}.png"
                torchvision.transforms.functional.to_pil_image(
                    imgs[j].clamp(-1, 1).add(1).div(2)).save(p)
            i += bs
        torch.cuda.empty_cache()

    logging.info("done")


if __name__ == "__main__":
    main()
