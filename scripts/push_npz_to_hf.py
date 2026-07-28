"""Upload a conditioning .npz to the IFCB HF dataset repo so cluster pods can fetch it.

The per-image embeddings file is ~100MB (74181 x 512), too large to track in git, so it
travels with the data rather than the code.

  python scripts/push_npz_to_hf.py ifcb_rd32_participation_morpho_img.npz
"""

import argparse
import os
import sys

DEFAULT_REPO = "danielaivanova/ifcb-finediffusion"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="local .npz to upload")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--path-in-repo", default=None,
                    help="destination name (default: the file's basename)")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        sys.exit(f"missing {args.npz}")
    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()
    ns = args.repo.split("/")[0]
    if ns != me.get("name") and ns not in [o.get("name") for o in (me.get("orgs") or [])]:
        sys.exit(f"namespace {ns!r} is not yours ({me.get('name')!r})")
    dest = args.path_in_repo or os.path.basename(args.npz)
    print(f"uploading {args.npz} ({os.path.getsize(args.npz) / 1e6:.1f} MB) -> {args.repo}/{dest}")
    api.upload_file(path_or_fileobj=args.npz, path_in_repo=dest,
                    repo_id=args.repo, repo_type="dataset")
    print("done")


if __name__ == "__main__":
    main()
