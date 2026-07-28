"""Upload the IFCB training data to the HF Hub so a cluster pod can pull it.

Mirrors how Planktonzilla reached the cluster: the data lives in a Hub repo, and a CPU-only
prep pod downloads it once onto the shared volume, where every later training pod reads it.

Uploads the images tree and the annotation CSVs verbatim (no HF `datasets` conversion) — the
training code walks `Images/<Class>/*.png` directly and reads the CSVs with pandas, so keeping
the on-disk layout identical means nothing in the loader has to change.

  # inspect what would be uploaded, upload nothing:
  python scripts/push_ifcb_to_hf.py --dry-run

  # upload (private by default):
  python scripts/push_ifcb_to_hf.py

The repo is PRIVATE unless --public is passed. IFCB is a third-party dataset; publishing a
copy is a licensing decision, so it is never the default.
"""

import argparse
import os
import sys

DEFAULT_SRC = "/scratch/datasets/other/IFCB_FishNet_Format"
DEFAULT_REPO = "daniela997/ifcb-finediffusion"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="local IFCB root (default: %(default)s)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hub dataset repo (default: %(default)s)")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public. Default is PRIVATE — IFCB is third-party data.")
    ap.add_argument("--dry-run", action="store_true", help="report what would upload, then stop")
    args = ap.parse_args()

    images = os.path.join(args.src, "Images")
    anns = os.path.join(args.src, "anns")
    for p in (images, anns):
        if not os.path.isdir(p):
            sys.exit(f"missing {p}")

    n_png = sum(len([f for f in fs if f.endswith(".png")]) for _, _, fs in os.walk(images))
    n_cls = sum(1 for d in os.scandir(images) if d.is_dir())
    csvs = sorted(f for f in os.listdir(anns) if f.endswith((".csv", ".json", ".txt")))
    print(f"source  : {args.src}")
    print(f"  images: {n_png} png across {n_cls} class dirs")
    print(f"  anns  : {len(csvs)} files -> {', '.join(csvs)}")
    print(f"target  : {args.repo} ({'PUBLIC' if args.public else 'private'})")

    if args.dry_run:
        print("\n(dry run - nothing uploaded)")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    # Two calls so a failure part-way leaves a diagnosable state (anns are tiny and go first,
    # so a broken run is obvious from the repo contents rather than silently half-populated).
    print("\nuploading anns/ ...")
    api.upload_folder(folder_path=anns, path_in_repo="anns",
                      repo_id=args.repo, repo_type="dataset")
    print("uploading Images/ (2.3GB, this is the slow part) ...")
    api.upload_folder(folder_path=images, path_in_repo="Images",
                      repo_id=args.repo, repo_type="dataset")
    print(f"\ndone: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
