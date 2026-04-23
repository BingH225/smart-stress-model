from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare assets for stepwise StressID remote evaluation")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def binary_labels(lbl: np.ndarray) -> np.ndarray:
    return (lbl == 2).astype(np.int64)


def stratified_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    pos = idx[y == 1]
    neg = idx[y == 0]
    n_pos = max(1, int(round(len(pos) * fraction)))
    n_neg = max(1, int(round(len(neg) * fraction)))
    pos_sel = rng.choice(pos, size=n_pos, replace=False) if len(pos) > 0 else np.array([], dtype=np.int64)
    neg_sel = rng.choice(neg, size=n_neg, replace=False) if len(neg) > 0 else np.array([], dtype=np.int64)
    out = np.concatenate([pos_sel, neg_sel], axis=0)
    out.sort()
    return out


def compute_source_stats(wesad_json_paths: List[Path], keep_labels: Tuple[int, ...] | None) -> Dict:
    feats: List[np.ndarray] = []
    for path in wesad_json_paths:
        obj = read_json(path)
        x = np.asarray(obj["features"], dtype=np.float64)
        lbl = np.asarray(obj["label"], dtype=np.int64)
        if keep_labels is not None:
            mask = np.isin(lbl, np.asarray(keep_labels, dtype=np.int64))
            x = x[mask]
        if len(x) > 0:
            feats.append(x)
    if not feats:
        raise RuntimeError("No source features found for requested label filter")
    all_x = np.concatenate(feats, axis=0)
    mean = np.mean(all_x, axis=0)
    cov = np.cov(all_x, rowvar=False)
    return {
        "feature_dim": int(all_x.shape[1]),
        "samples": int(all_x.shape[0]),
        "mean": mean.tolist(),
        "cov": cov.tolist(),
        "labels_used": list(keep_labels) if keep_labels is not None else [1, 2, 3, 4],
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    runtime = args.runtime_dir.resolve()

    stressid_merged = repo / "Data_Processed_StressID" / "WESADECG_S17.json"
    wesad_dir = repo / "Data_Processed"

    merged = read_json(stressid_merged)
    x = np.asarray(merged["features"], dtype=np.float64)
    lbl = np.asarray(merged["label"], dtype=np.int64)
    y = binary_labels(lbl)
    calib_idx = stratified_indices(y, args.calib_fraction, args.seed)

    calib = {
        "features": x[calib_idx].astype(float).tolist(),
        "label": lbl[calib_idx].astype(int).tolist(),
    }
    test_copy = {
        "features": x.astype(float).tolist(),
        "label": lbl.astype(int).tolist(),
    }

    data_dir = runtime / "data"
    write_json(data_dir / "STRESSID_CALIB.json", calib)
    write_json(data_dir / "STRESSID_TEST.json", test_copy)

    wesad_jsons = sorted(wesad_dir.glob("WESADECG_S*.json"))
    stats_all = compute_source_stats(wesad_jsons, keep_labels=None)
    stats_neutral_stress = compute_source_stats(wesad_jsons, keep_labels=(1, 2))

    cfg_dir = runtime / "config"
    write_json(cfg_dir / "source_stats_all.json", stats_all)
    write_json(cfg_dir / "source_stats_neutral_stress.json", stats_neutral_stress)

    summary = {
        "stressid_total_samples": int(len(lbl)),
        "stressid_calib_samples": int(len(calib_idx)),
        "stressid_calib_fraction": float(args.calib_fraction),
        "wesad_source_files": [str(p) for p in wesad_jsons],
        "source_stats_all_samples": int(stats_all["samples"]),
        "source_stats_neutral_stress_samples": int(stats_neutral_stress["samples"]),
    }
    write_json(runtime / "asset_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
