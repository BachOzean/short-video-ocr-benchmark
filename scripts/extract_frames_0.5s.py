#!/usr/bin/env python3
"""Extract fixed-interval frames from videos into video_id/*.jpg folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_video(video_path: Path, out_dir: Path, interval: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    video_out = out_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    count = 0
    t = 0.0
    while t <= duration + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        cv2.imwrite(str(video_out / f"{count:06d}.jpg"), frame)
        t += interval
    cap.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=Path("data/formal_testset/videos"))
    parser.add_argument("--out", type=Path, default=Path("frames_0.5s"))
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for video_path in sorted(args.videos.glob("*.mp4")):
        frames = extract_video(video_path, args.out, args.interval)
        print(f"{video_path.stem}: {frames}")


if __name__ == "__main__":
    main()
