"""Pre-processes a Pine real2sim LeRobot-v2.1 dataset for fast training.

Every video is center-cropped to 480x480, resized to 224x224 and re-encoded as H.264 with *every frame a keyframe*
(all-intra), so a random frame lookup decodes exactly one frame instead of up to a whole GOP. Parquet files and meta are
copied; `info.json` gets the new video shapes and `meta/episodes_stats.jsonl` (numeric + image stats) is generated,
so the output loads with plain LeRobotDataset (no hub access, no overlay needed).

    python scripts/prepare_pine_sim_fast.py <src_root> <dst_root> [--workers 16]
    # <src_root> holds one LeRobot dataset per task folder (e.g. pine_wm_real2sim_v0/)
"""

import argparse
import concurrent.futures as futures
import json
import pathlib
import shutil
import subprocess

import av
import numpy as np
import pandas as pd

CROP = 480  # center crop of the 640x480 frames: x offset (640 - 480) / 2 = 80
SIZE = 224
IMAGE_SAMPLE_STRIDE = 10  # image stats are computed on every 10th frame
VIDEO_INFO = {"video.width": SIZE, "video.height": SIZE, "video.fps": 15.0, "video.codec": "h264",
              "video.pix_fmt": "yuv420p", "video.channels": 3, "video.is_depth_map": False, "has_audio": False,
              "video.gop_size": 1}  # fmt: skip


def _frame_count(path: pathlib.Path) -> int | None:
    try:
        with av.open(str(path)) as container:
            return container.streams.video[0].frames
    except Exception:
        return None


def encode(src: pathlib.Path, dst: pathlib.Path) -> str:
    if dst.exists() and _frame_count(dst) == _frame_count(src):
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src),
         "-vf", f"crop={CROP}:{CROP}:(iw-{CROP})/2:(ih-{CROP})/2,scale={SIZE}:{SIZE}:flags=lanczos",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p",
         "-an", "-vsync", "passthrough", "-threads", "2", "-movflags", "+faststart", str(tmp)],
        check=True,
    )  # fmt: skip
    n_src, n_dst = _frame_count(src), _frame_count(tmp)
    if n_src != n_dst:
        raise RuntimeError(f"frame count changed for {src}: {n_src} -> {n_dst}")
    tmp.rename(dst)
    return "ok"


def _stats(values: np.ndarray) -> dict:
    values = values.astype(np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(), "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(), "count": [len(values)]}  # fmt: skip


def _image_stats(video: pathlib.Path, count: int) -> dict:
    with av.open(str(video)) as container:
        frames = [f.to_ndarray(format="rgb24") for i, f in enumerate(container.decode(video=0))
                  if i % IMAGE_SAMPLE_STRIDE == 0]  # fmt: skip
    pixels = np.stack(frames).astype(np.float64) / 255.0  # N,H,W,3
    per_channel = lambda fn: fn(pixels, axis=(0, 1, 2)).reshape(3, 1, 1).tolist()  # noqa: E731
    return {"min": per_channel(np.min), "max": per_channel(np.max), "mean": per_channel(np.mean),
            "std": per_channel(np.std), "count": [count]}  # fmt: skip


def episode_stats(args: tuple) -> str:
    dst_task, episode, numeric_keys, video_keys, chunks_size = args
    chunk = episode // chunks_size
    frame = pd.read_parquet(dst_task / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet")
    stats = {k: _stats(np.stack(frame[k].to_numpy())) for k in numeric_keys}
    for key in video_keys:
        video = dst_task / f"videos/chunk-{chunk:03d}/{key}/episode_{episode:06d}.mp4"
        stats[key] = _image_stats(video, len(frame))
    return json.dumps({"episode_index": episode, "stats": stats})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=pathlib.Path)
    parser.add_argument("dst", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tasks = sorted(p for p in args.src.iterdir() if (p / "meta/info.json").exists())
    infos, jobs = {}, []
    for task in tasks:
        dst_task = args.dst / task.name
        info = json.loads((task / "meta/info.json").read_text())
        infos[task.name] = info
        shutil.copytree(task / "data", dst_task / "data", dirs_exist_ok=True)
        shutil.copytree(task / "meta", dst_task / "meta", dirs_exist_ok=True)
        jobs.extend((video, dst_task / video.relative_to(task)) for video in sorted((task / "videos").rglob("*.mp4")))
    print(f"{len(jobs)} videos to encode", flush=True)

    with futures.ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(encode, *zip(*jobs, strict=True), chunksize=1)):
            if i % 25 == 0:
                print(f"  encoded {i + 1}/{len(jobs)} ({result})", flush=True)

    for task in tasks:
        dst_task, info = args.dst / task.name, infos[task.name]
        video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        numeric_keys = [k for k, v in info["features"].items() if v["dtype"] not in ("video", "image", "string")]
        for key in video_keys:
            info["features"][key]["shape"] = [SIZE, SIZE, 3]
            info["features"][key]["video_info"] = dict(VIDEO_INFO)
        (dst_task / "meta/info.json").write_text(json.dumps(info, indent=4))
        episodes = [
            json.loads(line)["episode_index"] for line in (task / "meta/episodes.jsonl").read_text().splitlines()
        ]
        work = [(dst_task, e, numeric_keys, video_keys, info["chunks_size"]) for e in episodes]
        with futures.ProcessPoolExecutor(args.workers) as pool:
            lines = list(pool.map(episode_stats, work, chunksize=2))
        (dst_task / "meta/episodes_stats.jsonl").write_text("\n".join(lines) + "\n")
        print(f"{task.name}: {len(episodes)} episodes, stats written", flush=True)

    for name in ("README.md", ".gitattributes"):
        if (args.src / name).exists():
            shutil.copy2(args.src / name, args.dst / name)


if __name__ == "__main__":
    main()
