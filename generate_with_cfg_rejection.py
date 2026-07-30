#!/usr/bin/env python3
"""Generate the main synthetic set with CFG-Rejection best-of-N sample selection.

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
Candidate 0 is the no-selection control: drawn first, from the same distribution, written to
_base whatever its score. It is NOT bit-identical to the earlier plain sets (those seed per
accumulated-sample index over a different batch layout), so compare _base vs _sel -- both from
this run -- rather than _sel against the earlier set's FID.

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

    The score difference is taken from _forward_clip_cfg's OUTPUT, not from a hook on
    final_layer: final_layer emits patch tokens of shape [B, num_patches, patch_dim] BEFORE
    unpatchify, so slicing [:, :3] there takes three PATCHES, not three image channels, and
    the resulting "ASD" is an arbitrary corner of the latent. (That bug produced scores ~6x
    too small and uncorrelated with the true score gap.)

    Runs the same guided loop as full sampling but stops early: p_sample_loop_progressive
    yields per-step, so we break after tau steps and never pay for the remaining steps.
    """
    b = z.shape[0]
    gaps = []

    def fn(x_t, t, **kw):
        # Reproduce forward_with_cfg's own null construction, then keep both halves so the
        # true cond/uncond split is available. models.py applies guidance to the first 3
        # channels only, which is where the score difference lives.
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        n = combined.shape[0]
        force_drop = torch.zeros(n, dtype=torch.long, device=combined.device)
        force_drop[n // 2:] = 1
        out = model._forward_clip_cfg(combined, t, kw["y"], force_drop,
                                      image_emb=kw.get("image_emb"))
        eps, rest = out[:, :3], out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        gaps.append((cond_eps - uncond_eps).flatten(1).norm(dim=-1).detach().float().cpu())
        half_eps = uncond_eps + kw["cfg_scale"] * (cond_eps - uncond_eps)
        return torch.cat([torch.cat([half_eps, half_eps], dim=0), rest], dim=1)

    zz, yy = torch.cat([z, z], 0), torch.cat([y, y], 0)
    kw = dict(y=yy, cfg_scale=cfg_scale)
    if image_emb is not None:
        kw["image_emb"] = torch.cat([image_emb, image_emb], 0)
    for i, _ in enumerate(diffusion.p_sample_loop_progressive(
            fn, zz.shape, zz, clip_denoised=False,
            model_kwargs=kw, progress=False, device=device)):
        if i + 1 >= tau:
            break
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
    ap.add_argument("--batch_size", type=int, default=4,
                    help="SLOTS per batch. Each slot is scored as n_candidates separate passes "
                         "over a 2*batch_size batch (CFG doubles it), so peak memory scales with "
                         "batch_size, not with n_candidates. 4 fits a 24GB card at N=4; 8 OOMs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit_per_class", type=int, default=None,
                    help="cap slots per class, for a quick FID-comparable subsample")
    ap.add_argument("--image_sampling", "--image-sampling", default="mean",
                    choices=("mean", "real"),
                    help="conditioning at sampling time. 'mean' uses the class-mean embedding, "
                         "which is what the EXISTING main sets used (generate_synthetic_dataset "
                         "also defaults to mean and the pod runs passed no override), so it is "
                         "the only setting that keeps a FID comparison against them valid. "
                         "'real' conditions on each slot's own source-image embedding -- closer "
                         "to how the model was trained, and an untried lever, but it changes two "
                         "things at once if combined with selection.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)
    base_dir = Path(f"{args.output_dir}_base")
    sel_dir = Path(f"{args.output_dir}_sel")

    classes = sorted(d.name for d in Path(args.data_path).iterdir() if d.is_dir())
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
    logging.info(f"image conditioning: {use_img}, sampling={args.image_sampling}")

    for n, cls in enumerate(mine, 1):
        srcs = sorted(per_class[cls])
        if args.limit_per_class:
            srcs = srcs[: args.limit_per_class]
        ci = classes.index(cls)
        # Resume on the MINIMUM of the two sets. _base and _sel are written together per
        # batch and must correspond slot-for-slot, so a class is complete only when both
        # halves are. This also means purging one half correctly forces the other to redo:
        # a fixed-metric _sel must not be paired with a stale _base.
        done = min(len(list((base_dir / cls).glob("*.png"))),
                   len(list((sel_dir / cls).glob("*.png")))) if args.resume else 0
        if done >= len(srcs):
            logging.info(f"[{n}/{len(mine)}] {cls}: complete ({done})")
            continue
        logging.info(f"[{n}/{len(mine)}] {cls}: {len(srcs) - done} slots")

        i = done
        while i < len(srcs):
            bs = min(args.batch_size, len(srcs) - i)
            slots = srcs[i:i + bs]

            # Conditioning per slot. Default 'mean' matches the existing main sets, so the
            # only variable versus them is the selection policy.
            if not use_img:
                e_slot = None
            elif args.image_sampling == "mean":
                m = model.y_embedder.clip_image_mean[ci].cpu().numpy()
                e_slot = torch.from_numpy(np.stack([m] * bs)).to(args.device)
            else:
                e = []
                for s in slots:
                    k = f"{cls}/{s}"
                    e.append(emb_all[rows[k]] if k in rows
                             else model.y_embedder.clip_image_mean[ci].cpu().numpy())
                e_slot = torch.from_numpy(np.stack(e)).to(args.device)

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
            names = [f"synthetic_{i + k:05d}.png" for k in range(bs)]

            # Score the candidates, freeing each prefix run before the next: the scoring loop
            # is what makes this memory-hungry (N passes over a 2*bs batch), and holding all N
            # plus two decoded image tensors at once OOMs a 24GB card.
            with torch.autocast("cuda", dtype=torch.float16):
                scores = []
                for cz in cand_z:
                    scores.append(asd_prefix(model, diffusion, cz, y, e_slot,
                                             args.cfg_scale, args.tau, args.device))
                    torch.cuda.empty_cache()
                best = torch.stack(scores).argmax(0)          # [bs]

            # Decode and WRITE each set before starting the next, so only one batch of decoded
            # images is resident at a time.
            sel_z = torch.stack([cand_z[best[k]][k] for k in range(bs)])
            for z_use, out_dir in ((cand_z[0], base_dir), (sel_z, sel_dir)):
                with torch.autocast("cuda", dtype=torch.float16):
                    imgs = sample_full(model, vae, diffusion, z_use, y, e_slot,
                                       args.cfg_scale, args.device)
                save(imgs, [out_dir / cls / nm for nm in names])
                del imgs
                torch.cuda.empty_cache()
            del cand_z, sel_z, scores
            i += bs
        torch.cuda.empty_cache()

    logging.info(f"done: base={sum(1 for _ in base_dir.rglob('*.png'))} "
                 f"sel={sum(1 for _ in sel_dir.rglob('*.png'))}")


if __name__ == "__main__":
    main()
