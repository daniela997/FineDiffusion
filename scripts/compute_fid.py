"""FID between real IFCB images and one or more synthetic sets, globally and per class.

Every set goes through ONE preprocessing function. FID is notoriously sensitive to the
resize path — if real and synthetic are resampled differently you measure interpolation
rather than generative quality — so the only difference between sets here is their pixels.

  python scripts/compute_fid.py \
      --real /scratch/datasets/other/IFCB_FishNet_Format/Images \
      --train-csv /scratch/datasets/other/IFCB_FishNet_Format/anns/ifcb_train.csv \
      --synthetic finediffusion=/path/FineDiffusion_synthetic_cropped \
                  taxadiffusion=/path/taxadiffusion_synthetic_cropped \
      --out fid_results.csv

Per-class FID is reported alongside the global number: with 145 classes and a long tail, a
global FID can hide exactly the fine-grained behaviour the conditioning work targets. Note
that per-class FID on a few hundred images is badly biased upward (the covariance estimate
is noisy at small N) — it is comparable BETWEEN methods on the same class, but should not be
read as an absolute distance. Classes below --min-per-class are skipped for that reason.

A caveat that applies to every method equally: images are generated at 256x256, so their
aspect ratios are compressed relative to real (measured 1.19 vs 1.40). That inflates all
absolute FIDs but leaves the ranking between methods intact.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class _Images(Dataset):
    """Loads uint8 CHW tensors at a fixed size — the single shared preprocessing path."""

    def __init__(self, paths, size):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BICUBIC)
        # torchmetrics' FID wants uint8 CHW when normalize=False.
        x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return x.view(self.size, self.size, 3).permute(2, 0, 1).contiguous()


def list_images(root, restrict=None):
    """{class: [paths]} for root/<class>/*.png, optionally restricted to given filenames."""
    out = defaultdict(list)
    for d in sorted(os.scandir(root), key=lambda e: e.name):
        if not d.is_dir():
            continue
        for f in sorted(os.scandir(d.path), key=lambda e: e.name):
            if not f.name.endswith(".png"):
                continue
            if restrict is not None and (d.name, f.name) not in restrict:
                continue
            out[d.name].append(f.path)
    return out


@torch.no_grad()
def features(paths, metric, size, batch, workers, device, real):
    dl = DataLoader(_Images(paths, size), batch_size=batch, num_workers=workers,
                    pin_memory=True)
    for x in dl:
        metric.update(x.to(device), real=real)


def fid_for(real_paths, fake_paths, size, batch, workers, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    features(real_paths, m, size, batch, workers, device, real=True)
    features(fake_paths, m, size, batch, workers, device, real=False)
    v = float(m.compute())
    del m
    torch.cuda.empty_cache()
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="real image root (<class>/*.png)")
    ap.add_argument("--train-csv", default=None,
                    help="restrict the real set to the TRAINING split, so FID is not computed "
                         "against images the generators never saw. Strongly recommended.")
    ap.add_argument("--synthetic", nargs="+", required=True,
                    help="one or more name=path entries")
    ap.add_argument("--size", type=int, default=299,
                    help="images are resized to size x size before Inception (default: its "
                         "native 299, so nothing is resampled twice)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--per-class", action="store_true", help="also compute FID per class")
    ap.add_argument("--min-per-class", type=int, default=100,
                    help="skip per-class FID below this many images in either set — the "
                         "covariance estimate is too noisy to mean anything (default: 100)")
    ap.add_argument("--out", default=None, help="write results to this CSV")
    args = ap.parse_args()

    restrict = None
    if args.train_csv:
        import pandas as pd
        df = pd.read_csv(args.train_csv)
        restrict = set(zip(df["Folder"], df["image"]))
        print(f"restricting real to the {len(restrict)} training rows in {args.train_csv}")

    real = list_images(args.real, restrict)
    real_flat = [p for ps in real.values() for p in ps]
    print(f"real: {len(real_flat)} images in {len(real)} classes")

    sets = {}
    for spec in args.synthetic:
        if "=" not in spec:
            sys.exit(f"--synthetic entries must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.isdir(path):
            sys.exit(f"missing {path}")
        sets[name] = list_images(path)
        n = sum(len(v) for v in sets[name].values())
        print(f"{name}: {n} images in {len(sets[name])} classes")

    rows = []
    print("\n=== GLOBAL FID ===")
    for name, imgs in sets.items():
        flat = [p for ps in imgs.values() for p in ps]
        v = fid_for(real_flat, flat, args.size, args.batch, args.workers, args.device)
        print(f"  {name:20s} {v:8.2f}   (n_real={len(real_flat)}, n_fake={len(flat)})")
        rows.append({"scope": "global", "class": "", "method": name, "fid": v,
                     "n_real": len(real_flat), "n_fake": len(flat)})

    if args.per_class:
        print(f"\n=== PER-CLASS FID (classes with >= {args.min_per_class} images in every set) ===")
        classes = [c for c in sorted(real)
                   if len(real[c]) >= args.min_per_class
                   and all(len(s.get(c, [])) >= args.min_per_class for s in sets.values())]
        print(f"{len(classes)} of {len(real)} classes qualify\n")
        header = f"{'class':40s}" + "".join(f"{n:>16s}" for n in sets)
        print(header)
        for c in classes:
            line = f"{c[:39]:40s}"
            for name, imgs in sets.items():
                v = fid_for(real[c], imgs[c], args.size, args.batch, args.workers, args.device)
                line += f"{v:16.2f}"
                rows.append({"scope": "class", "class": c, "method": name, "fid": v,
                             "n_real": len(real[c]), "n_fake": len(imgs[c])})
            print(line, flush=True)

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["scope", "class", "method", "fid",
                                               "n_real", "n_fake"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
