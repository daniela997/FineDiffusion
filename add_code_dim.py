#!/usr/bin/env python3
"""Warm-start a pure-CLIP checkpoint into the CLIP+learned-code hybrid (code_dim > 0).

ClipEmbedder concatenates [clip_text, code, clip_image] before its projection MLP, so adding a
trainable per-class code changes exactly two tensors:

    y_embedder.code.weight     NEW      (num_classes, code_dim)
    y_embedder.mlp.0.weight    RESIZED  (proj_hidden, clip+code+img) from (proj_hidden, clip+img)

Everything else -- all 28 DiT blocks, the second MLP layer, both CLIP buffers -- is untouched.

The new columns are ZERO-initialised, so at step 0 the model computes bit-identically to the
checkpoint it came from: the code contributes nothing until gradients move those weights. That
makes this a true warm start rather than a perturbation, and it means any change in behaviour is
attributable to the code learning something, not to the surgery.

Column ORDER matters: forward() builds [clip_vec] + [code] + [img], so the code's columns go
BETWEEN the text and image blocks, not at the end. Splicing them at the end would silently
misroute the image half into the code slot.

WHY code_dim was 0 to begin with (see models.py): a free per-class lookup table is exactly what
CLIP conditioning is meant to replace, and with 145 classes even 32 dims can encode identity and
route around CLIP -- which would flatten an encoder ablation. That argument applies to comparing
ENCODERS; it does not apply to asking whether a learned code recovers structure CLIP cannot
represent (e.g. Carchesium's stalked colony, Akashiwo's cingulum). This script is for the latter.

CONFOUND to state in any writeup: continuing from a trained checkpoint adds both the code AND
extra training steps. A from-scratch code_dim run is the clean comparison; this is the cheap one.

  python add_code_dim.py --in ckpt.pt --out ckpt_code32.pt --code-dim 32
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--code-dim", type=int, default=32)
    ap.add_argument("--clip-dim", type=int, default=512)
    ap.add_argument("--num-classes", type=int, default=145)
    ap.add_argument("--col-init", type=float, default=0.0,
                    help="std for the NEW MLP columns. 0 makes the patched model bit-identical "
                         "to the source, which is a clean warm start but starves the code of "
                         "gradient: measured after 18.5k steps, the code block reached only "
                         "||W||=0.96 against 23.9 (text) and 27.6 (image), i.e. ~0.03%% of the "
                         "conditioning signal, and the code table SHRANK from its 0.02 init. "
                         "Set this to the scale of the existing columns (~0.03) so the code "
                         "starts on comparable footing and can actually compete.")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    # Checkpoints hold {"model", "ema", "opt", "args"}; patch every weight dict present so the
    # EMA (which is what generation actually uses) is not left at the old shape.
    keys = [k for k in ("model", "ema") if isinstance(ck, dict) and k in ck]
    if not keys:
        raise SystemExit("no 'model'/'ema' state dict in this checkpoint")

    for key in keys:
        sd = ck[key]
        w = sd.get("y_embedder.mlp.0.weight")
        if w is None:
            raise SystemExit(f"{key}: no y_embedder.mlp.0.weight -- not a ClipEmbedder checkpoint")
        if "y_embedder.code.weight" in sd:
            raise SystemExit(f"{key}: already has a code table; nothing to do")

        out_dim, in_dim = w.shape
        img_dim = in_dim - args.clip_dim
        if img_dim < 0:
            raise SystemExit(f"{key}: input {in_dim} smaller than clip_dim {args.clip_dim}")

        # [text | code | image] -- splice the zeroed code columns between the two blocks.
        new = torch.zeros(out_dim, in_dim + args.code_dim, dtype=w.dtype)
        new[:, :args.clip_dim] = w[:, :args.clip_dim]
        if img_dim:
            new[:, args.clip_dim + args.code_dim:] = w[:, args.clip_dim:]
        if args.col_init > 0:
            new[:, args.clip_dim:args.clip_dim + args.code_dim] = (
                torch.randn(out_dim, args.code_dim, dtype=w.dtype) * args.col_init)
        sd["y_embedder.mlp.0.weight"] = new

        # Small random init, matching ClipEmbedder.__init__. The zeroed MLP columns mean this
        # has no effect on the output until training moves them.
        sd["y_embedder.code.weight"] = torch.randn(args.num_classes, args.code_dim) * 0.02
        print(f"{key}: mlp.0.weight {tuple(w.shape)} -> {tuple(new.shape)} "
              f"(text {args.clip_dim} + code {args.code_dim} + image {img_dim})")

    if isinstance(ck.get("args"), object) and hasattr(ck.get("args"), "__dict__"):
        setattr(ck["args"], "clip_code_dim", args.code_dim)
        print(f"args.clip_code_dim -> {args.code_dim}")

    torch.save(ck, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
