#!/usr/bin/env python3
"""Calibrate CFG-Rejection (ASD) sample filtering on our DiT plankton generator.

CFG-Rejection (Diffusion Sampling Path Tells More, /home/daniela/other/CFG-Rejection) scores a
sample by the Accumulated Score Differences along its denoising path:

    ASD = mean_over_steps ||eps_cond - eps_uncond||

High ASD correlates with landing in a high-density, class-consistent region of the learned
manifold; low ASD with sparse regions and weak semantic alignment. Both terms are already
computed by classifier-free guidance, so tracking ASD costs one norm per step and NO extra
forward passes.

Before committing to best-of-N generation we have to establish that the signal is real for OUR
model -- a DiT-XL/2 on 256^2 plankton is not SDv1.5 on ImageNet. This script generates N
candidates per conditioning image, keeps ALL of them plus their per-step ASD traces, and lets
an independent judge (Grounding DINO detection confidence, which is not part of the downstream
classifier) test whether ASD ranking agrees with sample quality.

Outputs, under --output_dir:
    <class>/cand_<slot>_<n>.png      every candidate
    asd_traces.pt                    {key: tensor[N, steps]} full per-step traces
    asd_summary.csv                  one row per candidate: class, slot, n, asd_full, asd_10, ...

Then run analyse_cfg_rejection.py to correlate ASD against detection confidence.

  python calibrate_cfg_rejection.py \
      --ckpt /scratch/daniela/hyperbolic_plankton_ckpts/from_cluster/finediffusion_clip_text_image.pt \
      --npz  ifcb_rd32_participation_morpho_img.npz \
      --output_dir /scratch/daniela/cfg_calib --n_candidates 8 --n_slots 25
"""

import argparse
import collections
import contextlib
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from generate_synthetic_dataset import load_models

DINO_SCRIPTS = "/home/daniela/mine/dino/dino_classification/scripts"
# Prefixes at which the paper selects (cells 6-9 of sort_100_gen_sd1.5_guidance5.0.ipynb):
# early exit is the whole point -- rejecting at step 10 of 250 costs 4% of full generation.
SELECT_STEPS = (10, 20, 30, 40, 50)


def rare_class_slots(dataset_path, records_csv, until=100, seed=24, schedule_json=None):
    """Rare classes and their cyclic top-up schedule at val_split=0.1 (the real requirement).

    `schedule_json` short-circuits the computation with a precomputed {class: [src, ...]} map.
    The pods do not have the dino_classification package, and the schedule is deterministic, so
    it is cheaper to compute it once on the machine that has the splitter and ship the result
    than to replicate the dependency chain.
    """
    if schedule_json:
        import json
        d = json.load(open(schedule_json))
        sched = d["schedule"]
        return sched, d["train_counts"]

    sys.path.insert(0, DINO_SCRIPTS)
    from data import load_and_split_dataset

    rec = pd.read_csv(records_csv)
    with contextlib.redirect_stdout(io.StringIO()):
        d = load_and_split_dataset(dataset_path, 0.6, 0.1, 0.2, random_seed=seed,
                                   exclude=["Unclassifiable"], records_df=rec)
    by = collections.defaultdict(list)
    for p in d["train_paths"]:
        by[Path(p).parent.name].append(Path(p).name)

    sched = {}
    for c, v in by.items():
        files = sorted(v)
        if len(files) >= until:
            continue
        out, i = [], 0
        while len(files) + len(out) < until:
            out.append(files[i])
            i = (i + 1) % len(files)
        sched[c] = out
    return sched, {c: len(v) for c, v in by.items()}


@torch.no_grad()
def sample_with_asd(model, vae, diffusion, z, y, image_emb, cfg_scale, device):
    """p_sample_loop with CFG, recording ||eps_cond - eps_uncond|| at every step.

    Mirrors generate_conditioned_oversample.py's call exactly so the images are the ones the
    real generator would produce; the only addition is the per-step norm.
    """
    b = z.shape[0]
    gaps = []

    # forward_with_cfg computes cond/uncond internally and returns only the guided result, so
    # take the split off the model's own output via a forward hook -- no second forward pass.
    # Guidance is applied to the FIRST 3 CHANNELS only (models.py: eps, rest = out[:, :3],
    # out[:, 3:]), so the score difference lives there.
    def hook(_mod, _inp, output):
        eps = output[:, :3]
        cond, uncond = eps[: len(eps) // 2], eps[len(eps) // 2:]
        gaps.append((cond - uncond).flatten(1).norm(dim=-1).detach().float().cpu())

    handle = model.final_layer.register_forward_hook(hook)

    def fn(x_t, t, **kw):
        return model.forward_with_cfg(x_t, t, **kw)

    zz = torch.cat([z, z], 0)
    yy = torch.cat([y, y], 0)
    kw = dict(y=yy, cfg_scale=cfg_scale)
    if image_emb is not None:
        kw["image_emb"] = torch.cat([image_emb, image_emb], 0)

    try:
        s = diffusion.p_sample_loop(fn, zz.shape, zz, clip_denoised=False,
                                    model_kwargs=kw, progress=False, device=device)
    finally:
        handle.remove()
    s, _ = s.chunk(2, 0)
    imgs = vae.decode((s / 0.18215).to(next(vae.parameters()).dtype)).sample
    # p_sample_loop runs T..0, so gaps[0] is the FIRST (noisiest) step -- matching the paper's
    # "accumulate from the early steps" prefix selection.
    trace = torch.stack(gaps, dim=1) if gaps else torch.zeros(b, 0)   # [b, steps]
    return imgs, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--dataset_path",
                    default="/scratch/datasets/DEAL/Plankton/IFCB/IFCB_Annotated_Training_Library_FINAL")
    ap.add_argument("--records_csv",
                    default="/scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_records.csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--schedule_json", default=None,
                    help="precomputed {schedule, train_counts} from a machine with the splitter")
    ap.add_argument("--n_candidates", type=int, default=8)
    ap.add_argument("--n_slots", type=int, default=25,
                    help="conditioning slots to calibrate on, spread across rare classes")
    ap.add_argument("--num_sampling_steps", type=int, default=250)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sched, train_counts = rare_class_slots(args.dataset_path, args.records_csv,
                                           schedule_json=args.schedule_json)
    classes = sorted(d.name for d in Path(args.dataset_path).iterdir() if d.is_dir())

    # Spread the calibration slots over the rarest classes -- that is where quality varies most
    # and where selection would matter.
    rare_sorted = sorted(sched, key=lambda c: train_counts[c])
    picks = []
    ci = 0
    while len(picks) < args.n_slots and rare_sorted:
        c = rare_sorted[ci % len(rare_sorted)]
        k = len([p for p in picks if p[0] == c])
        if k < len(sched[c]):
            picks.append((c, k, sched[c][k]))
        ci += 1
        if ci > 10000:
            break
    logging.info(f"calibrating on {len(picks)} slots across "
                 f"{len(set(p[0] for p in picks))} rare classes, "
                 f"{args.n_candidates} candidates each "
                 f"= {len(picks)*args.n_candidates} images")

    z = np.load(args.npz, allow_pickle=True)
    rows_idx = {str(k): i for i, k in enumerate(z["clip_emb_image_keys"])}
    emb_all = z["clip_emb_image"].astype(np.float32)

    # load_models returns 3 values in older revisions and 4 (with image_bank_data) in newer
    # ones; the pod checkout may lag this one, so accept either.
    _loaded = load_models(args.ckpt, args.device, len(classes), 12, 256,
                          args.num_sampling_steps, clip_embeddings=args.npz)
    model, vae, diffusion = _loaded[0], _loaded[1], _loaded[2]
    if not hasattr(model.y_embedder, "clip_image_mean"):
        sys.exit("checkpoint has no image conditioning; needs one trained with --clip-image")

    traces, records = {}, []
    for n, (cls, slot, src) in enumerate(picks, 1):
        key = f"{cls}/{src}"
        if key not in rows_idx:
            logging.warning(f"no embedding for {key}, skipping")
            continue
        e = torch.from_numpy(emb_all[rows_idx[key]][None]).to(args.device)
        e = e.repeat(args.n_candidates, 1)
        ci_ = classes.index(cls)

        # One noise seed per candidate, so a candidate is reproducible from (seed, slot, n).
        torch.manual_seed(args.seed + 1000 * slot + n)
        zz = torch.randn(args.n_candidates, 4, 32, 32, device=args.device)
        y = torch.full((args.n_candidates,), ci_, dtype=torch.long, device=args.device)

        with torch.autocast("cuda", dtype=torch.float16):
            imgs, trace = sample_with_asd(model, vae, diffusion, zz, y, e,
                                          args.cfg_scale, args.device)

        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        for j in range(args.n_candidates):
            p = d / f"cand_{slot:03d}_{j}.png"
            torchvision.transforms.functional.to_pil_image(
                imgs[j].clamp(-1, 1).add(1).div(2)).save(p)
            t = trace[j].float()
            rec = dict(cls=cls, slot=slot, cand=j, src=src, path=str(p),
                       train_count=train_counts[cls],
                       asd_full=t.mean().item())
            for k in SELECT_STEPS:
                rec[f"asd_{k}"] = t[:k].mean().item() if t.numel() >= k else float("nan")
            records.append(rec)
        traces[f"{cls}/{slot}"] = trace
        logging.info(f"[{n}/{len(picks)}] {cls} slot {slot} ({train_counts[cls]} real): "
                     f"ASD range {trace.mean(1).min():.2f}-{trace.mean(1).max():.2f}")
        torch.cuda.empty_cache()

    torch.save(traces, out / "asd_traces.pt")
    pd.DataFrame(records).to_csv(out / "asd_summary.csv", index=False)
    logging.info(f"wrote {len(records)} candidates, traces + asd_summary.csv to {out}")


if __name__ == "__main__":
    main()
