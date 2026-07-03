#!/usr/bin/env python3
"""Create the MMOCR 1.x-compatible SAR_CN checkpoint used by this benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("weights/mmocr/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "weights/mmocr/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214_mmocr1_compat.pth"
        ),
    )
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    weight = state_dict["decoder.prediction.weight"]
    bias = state_dict["decoder.prediction.bias"]
    if tuple(weight.shape) == (11379, 1536):
        state_dict["decoder.prediction.weight"] = torch.cat(
            [weight, weight.new_zeros((1, weight.shape[1]))], dim=0
        )
    if tuple(bias.shape) == (11379,):
        state_dict["decoder.prediction.bias"] = torch.cat(
            [bias, bias.new_full((1,), -20.0)], dim=0
        )

    if "state_dict" in ckpt:
        ckpt["state_dict"] = state_dict

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
