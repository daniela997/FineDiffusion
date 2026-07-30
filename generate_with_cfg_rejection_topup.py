#!/usr/bin/env python3
"""Generate the rare-class TOP-UP with CFG-Rejection best-of-N sample selection.

Same selection mechanism as generate_with_cfg_rejection.py, but the slots come from syke-pic's
cyclic oversampling schedule (which real image each top-up is a variation of) rather than one
per row of the training CSV. See generate_conditioned_oversample.py for why the schedule, not
just the count, has to be followed.

Implements CFG-Rejection (Diffusion Sampling Path Tells More,
/home/daniela/other/CFG-Rejection) for our DiT: each output slot draws N candidate noise seeds,
denoises all of them to a cheap prefix, scores each by Accumulated Score Differences

    ASD = mean_over_first_tau_steps ||eps_cond - eps_uncond||

keeps the highest-ASD candidate, and finishes ONLY that one. Both terms are already computed by
classifier-free guidance, so the score costs one norm per step and no extra forward passes.

Cost, per slot, at tau=10 of 250 steps and N=4:
    N partial runs (N * tau steps) + 1 full run (250 steps) = 1 + (N-1)*tau/250 ~= 1.12x
which is what makes best-of-N affordable on a 59,344-image set at all.

TWO SETS ARE WRITTEN FROM ONE PASS so the comparison is controlled:
    <output_dir>_base/<class>/*.png      candidate 0 -- what plain generation would produce
    <output_dir>_sel/<class>/*.png       the best-ASD candidate
Candidate 0's noise seed is identical to what the plain generator uses for that slot, so the
only difference between the two sets is the selection policy -- not the seeds, not the count,
not the conditioning.

CAVEAT worth measuring rather than assuming: on a 3-slot calibration of this model, ASD appeared
to rank candidates by organism SIZE/CONTRAST rather than by validity (all samples were already
valid organisms -- our generator does not produce the prompt-following failures the paper's
gains come from). Selecting on it may therefore shift the distribution and make FID WORSE.
Compute FID on both output sets before adopting either.

  python generate_with_cfg_rejection.py \
      --ckpt <text+image ckpt> --npz <npz with clip_emb_image> \
      --train_csv .../anns/ifcb_train.csv --data_path .../Images \
      --output_dir .../synthetic_clip_text_image_cfgrej \
      --n_candidates 4 --tau 10 --shard 0 --num_shards 4 --device cuda:0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from generate_synthetic_dataset import load_models


def asd_prefix(model, diffusion, z, y, image_emb, cfg_scale, tau, device):
    """Denoise `tau` steps and return each candidate's mean ||eps_cond - eps_uncond||.

    Runs the same guided loop as full sampling but stops early: p_sample_loop_progressive
    yields per-step, so we break after tau steps and never pay for the remaining 240.
    """
    b = z.shape[0]
    gaps = []

    def hook(_m, _i, output):
        # Guidance is applied to the FIRST 3 CHANNELS only (models.py), so the score
        # difference lives there.
        eps = output[:, :3]
        cond, uncond = eps[: len(eps) // 2], eps[len(eps) // 2:]
        gaps.append((cond - uncond).flatten(1).norm(dim=-1).detach().float().cpu())

    handle = model.final_layer.register_forward_hook(hook)
    zz, yy = torch.cat([z, z], 0), torch.cat([y, y], 0)
    kw = dict(y=yy, cfg_scale=cfg_scale)
    if image_emb is not None:
        kw["image_emb"] = torch.cat([image_emb, image_emb], 0)
    try:
        for i, _ in enumerate(diffusion.p_sample_loop_progressive(
                model.forward_with_cfg, zz.shape, zz, clip_denoised=False,
                model_kwargs=kw, progress=False, device=device)):
            if i + 1 >= tau:
                break
    finally:
        handle.remove()
    if not gaps:
        return torch.zeros(b)
    return torch.stack(gaps, dim=1).mean(dim=1)          # [b]


def sample_full(model, vae, diffusion, z, y, image_emb, cfg_scale, device):
    """Ordinary full generation for a single chosen candidate set."""
    zz, yy = torch.cat([z, z], 0), torch.cat([y, y], 0)
    kw = dict(y=yy, cfg_scale=cfg_scale)
    if image_emb is not None:
        kw["image_emb"] = torch.cat([image_emb, image_emb], 0)
    s = diffusion.p_sample_loop(model.forward_with_cfg, zz.shape, zz, clip_denoised=False,
                                model_kwargs=kw, progress=False, device=device)
    s, _ = s.chunk(2, 0)
    return vae.decode((s / 0.18215).to(next(vae.parameters()).dtype)).sample


def save(imgs, paths):
    for img, p in zip(imgs, paths):
        p.parent.mkdir(parents=True, exist_ok=True)
        torchvision.transforms.functional.to_pil_image(
            img.clamp(-1, 1).add(1).div(2)).save(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--data_path", required=True, help="real image root, for the class list")
    ap.add_argument("--output_dir", required=True,
                    help="two dirs are written: <output_dir>_base and <output_dir>_sel")
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--tau", type=int, default=10, help="prefix steps used to score candidates")
    ap.add_argument("--num_sampling_steps", type=int, default=250)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="SLOTS per batch; each slot expands to n_candidates during scoring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit_per_class", type=int, default=None,
                    help="cap slots per class, for a quick FID-comparable subsample")
    ap.add_argument("--schedule_json", default=None,
                    help="cyclic top-up schedule {schedule: {class: [src, ...]}}; slots follow "
                         "it in order so each output conditions on the image naive duplication "
                         "would have copied")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)
    base_dir = Path(f"{args.output_dir}_base")
    sel_dir = Path(f"{args.output_dir}_sel")

    classes = sorted(d.name for d in Path(args.data_path).iterdir() if d.is_dir())
    if args.schedule_json:
        import json
        per_class = json.load(open(args.schedule_json))["schedule"]
    else:
        train = pd.read_csv(args.train_csv)
        per_class = train.groupby("Folder")["image"].apply(list).to_dict()

    mine = [c for i, c in enumerate(sorted(per_class)) if i % args.num_shards == args.shard]
    logging.info(f"shard {args.shard}/{args.num_shards}: {len(mine)} classes, "
                 f"N={args.n_candidates} tau={args.tau}")

    z = np.load(args.npz, allow_pickle=True)
    has_img = "clip_emb_image" in z
    rows = {str(k): i for i, k in enumerate(z["clip_emb_image_keys"])} if has_img else {}
    emb_all = z["clip_emb_image"].astype(np.float32) if has_img else None

    _l = load_models(args.ckpt, args.device, len(classes), 12, 256,
                     args.num_sampling_steps, clip_embeddings=args.npz)
    model, vae, diffusion = _l[0], _l[1], _l[2]
    use_img = hasattr(model.y_embedder, "clip_image_mean")
    logging.info(f"image conditioning: {use_img}")

    for n, cls in enumerate(mine, 1):
        srcs = sorted(per_class[cls])
        if args.limit_per_class:
            srcs = srcs[: args.limit_per_class]
        ci = classes.index(cls)
        done = len(list((sel_dir / cls).glob("*.png"))) if args.resume else 0
        if done >= len(srcs):
            logging.info(f"[{n}/{len(mine)}] {cls}: complete ({done})")
            continue
        logging.info(f"[{n}/{len(mine)}] {cls}: {len(srcs) - done} slots")

        i = done
        while i < len(srcs):
            bs = min(args.batch_size, len(srcs) - i)
            slots = srcs[i:i + bs]

            # One conditioning embedding per slot (mean when the class has no per-image entry).
            if use_img:
                e = []
                for s in slots:
                    k = f"{cls}/{s}"
                    e.append(emb_all[rows[k]] if k in rows
                             else model.y_embedder.clip_image_mean[ci].cpu().numpy())
                e_slot = torch.from_numpy(np.stack(e)).to(args.device)
            else:
                e_slot = None

            # Candidate noise. Candidate 0 is the no-selection control: it is drawn first, from
            # the same distribution, and is written to _base regardless of its score. It is NOT
            # bit-identical to the existing plain set (that one seeds per accumulated-sample
            # index over a different batch layout), so compare _base vs _sel -- both produced
            # here -- rather than _sel vs the earlier synthetic_clip_text_image set.
            cand_z = []
            for j in range(args.n_candidates):
                torch.manual_seed(args.seed + i + j * 1_000_003)
                cand_z.append(torch.randn(bs, 4, 32, 32, device=args.device))

            y = torch.full((bs,), ci, dtype=torch.long, device=args.device)
            with torch.autocast("cuda", dtype=torch.float16):
                scores = torch.stack([
                    asd_prefix(model, diffusion, cz, y, e_slot, args.cfg_scale,
                               args.tau, args.device)
                    for cz in cand_z])                       # [N, bs]
                best = scores.argmax(0)                       # [bs]

                # Full denoise for candidate 0 (baseline) and for the winner (selected).
                base_imgs = sample_full(model, vae, diffusion, cand_z[0], y, e_slot,
                                        args.cfg_scale, args.device)
                sel_z = torch.stack([cand_z[best[k]][k] for k in range(bs)])
                sel_imgs = sample_full(model, vae, diffusion, sel_z, y, e_slot,
                                       args.cfg_scale, args.device)

            names = [f"synthetic_{i + k:05d}.png" for k in range(bs)]
            save(base_imgs, [base_dir / cls / nm for nm in names])
            save(sel_imgs, [sel_dir / cls / nm for nm in names])
            i += bs
        torch.cuda.empty_cache()

    logging.info(f"done: base={sum(1 for _ in base_dir.rglob('*.png'))} "
                 f"sel={sum(1 for _ in sel_dir.rglob('*.png'))}")


if __name__ == "__main__":
    main()
