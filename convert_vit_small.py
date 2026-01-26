import argparse
import os
import torch
import timm
import jittor as jt
import models_jt as models
import sys


# Convert timm vit_small_patch16_224 PyTorch weights to Jittor format
# Usage:
# python convert_vit_small.py --dst Pretrained_Models/vit_small_jittor.pkl


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ViT-Small timm weights to Jittor")
    parser.add_argument(
        "--dst",
        type=str,
        default="Pretrained_Models/vit_small_jittor.pkl",
        help="Output Jittor weight file",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Optional local PyTorch weight file (.pth/.pt). If provided, no download is attempted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load timm pretrained model (PyTorch)
    if args.src:
        if not os.path.isfile(args.src):
            print(f"[Error] Provided --src not found: {args.src}")
            sys.exit(1)
        print(f"Loading PyTorch weights from {args.src}")
        state_dict = torch.load(args.src, map_location="cpu")
    else:
        try:
            model = timm.create_model("vit_small_patch16_224", pretrained=True)
            state_dict = model.state_dict()
        except Exception as exc:
            print("[Error] Failed to download timm pretrained weights. Provide --src pointing to a local .pth file.")
            print(f"Detail: {exc}")
            sys.exit(1)

    # Drop classification head
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    # 2) Build Jittor ViT-S model
    jt_model = models.ViTSmall(pretrained=False)
    jt_state = jt_model.state_dict()

    converted = {}
    missing = []
    extra = []

    for k in jt_state.keys():
        if k in state_dict:
            converted[k] = jt.array(state_dict[k].cpu().numpy())
        else:
            missing.append(k)

    for k in state_dict.keys():
        if k not in jt_state:
            extra.append(k)

    if missing:
        print("[Warning] Missing keys:", missing)
    if extra:
        print("[Info] Ignored extra keys:", extra)

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    jt.save(converted, args.dst)
    print(f"Converted weights saved to {args.dst}")


if __name__ == "__main__":
    main()
