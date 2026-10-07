"""Builds a LeRobot-v2.1-loadable overlay of datasets that lack `meta/episodes_stats.jsonl`.

LeRobot v2.1 requires per-episode stats and falls back to the HF hub when they are missing. The overlay leaves the
source dataset untouched: `data/` and `videos/` are symlinks, `meta/` is copied and extended with the stats
(numeric features only; video stats are not used by openpi, which computes its own norm stats).

    python scripts/prepare_lerobot_overlay.py <src_dataset_root> <dst_root>
    # <src_dataset_root> contains one LeRobot dataset per task folder (e.g. pine_wm_real2sim_v0/)
"""

import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd


def _feature_stats(values: np.ndarray) -> dict:
    values = values.astype(np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def build(task_src: pathlib.Path, task_dst: pathlib.Path) -> None:
    info = json.loads((task_src / "meta/info.json").read_text())
    keys = [k for k, v in info["features"].items() if v["dtype"] not in ("video", "image", "string")]
    task_dst.mkdir(parents=True, exist_ok=True)
    for name in ("data", "videos"):
        link = task_dst / name
        if not link.exists():
            link.symlink_to((task_src / name).resolve())
    shutil.copytree(task_src / "meta", task_dst / "meta", dirs_exist_ok=True)

    lines = []
    for ep in (json.loads(line) for line in (task_src / "meta/episodes.jsonl").read_text().splitlines()):
        i = ep["episode_index"]
        frame = pd.read_parquet(task_src / f"data/chunk-{i // info['chunks_size']:03d}/episode_{i:06d}.parquet")
        stats = {k: _feature_stats(np.stack(frame[k].to_numpy())) for k in keys}
        lines.append(json.dumps({"episode_index": i, "stats": stats}))
    (task_dst / "meta/episodes_stats.jsonl").write_text("\n".join(lines) + "\n")
    print(f"{task_src.name}: {len(lines)} episodes -> {task_dst}")


if __name__ == "__main__":
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for task in sorted(p for p in src.iterdir() if (p / "meta/info.json").exists()):
        build(task, dst / task.name)
