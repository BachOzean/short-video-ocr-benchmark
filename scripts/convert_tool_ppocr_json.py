#!/usr/bin/env python3
"""Convert tool PPOCR JSON outputs to unified jsonl_frames format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(engine: str, src: Path, out: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for video_id in sorted(data):
            frame_texts = []
            for item in data[video_id]:
                frame_texts.append(
                    {
                        "t": float(item.get("t", 0.0)),
                        "text": str(item.get("text") or ""),
                    }
                )
            row = {
                "engine": engine,
                "video_id": str(video_id),
                "frames": len(frame_texts),
                "frame_texts": frame_texts,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert(args.engine, args.src, args.out)


if __name__ == "__main__":
    main()
