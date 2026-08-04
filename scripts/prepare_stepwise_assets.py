from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare assets for stepwise StressID remote evaluation")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=("window", "subject"),
        default="window",
        help="window keeps the existing random calibration split; subject creates adaptation/eval splits by StressID subject.",
    )
    parser.add_argument(
        "--subject-fraction",
        type=float,
        default=0.1,
        help="Fraction of StressID subjects to allocate to the adaptation subset when split-mode=subject.",
    )
    parser.add_argument(
        "--subject-selection",
        type=str,
        choices=("random", "stratified"),
        default="random",
        help="How to choose adaptation subjects when split-mode=subject.",
    )
    parser.add_argument(
        "--subject-bins",
        type=int,
        default=5,
        help="Number of stress-ratio bins used by stratified subject selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
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


def subject_split_count(total_subjects: int, fraction: float) -> int:
    if total_subjects <= 1:
        return total_subjects
    requested = int(round(total_subjects * fraction))
    requested = max(1, requested)
    return min(total_subjects - 1, requested)


def label_dist(lbl: np.ndarray) -> Dict[str, int]:
    return {str(int(v)): int(np.sum(lbl == v)) for v in sorted(np.unique(lbl))}


def subject_stress_ratio(lbl: np.ndarray) -> float:
    if len(lbl) == 0:
        return 0.0
    return float(np.mean(lbl == 2))


def load_subject_files(stressid_dir: Path) -> List[Tuple[str, Path, Dict[str, Any]]]:
    out: List[Tuple[str, Path, Dict[str, Any]]] = []
    for path in sorted(stressid_dir.glob("STRESSIDECG_*.json")):
        subject = path.stem.split("STRESSIDECG_", 1)[1]
        out.append((subject, path, read_json(path)))
    if not out:
        raise RuntimeError(f"No StressID subject JSON files found in {stressid_dir}")
    return out


def build_subject_records(subject_entries: List[Tuple[str, Path, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for subject, path, payload in subject_entries:
        lbl = np.asarray(payload["label"], dtype=np.int64)
        records.append(
            {
                "subject": subject,
                "path": path,
                "payload": payload,
                "samples": int(len(lbl)),
                "label_dist": label_dist(lbl),
                "stress_ratio": subject_stress_ratio(lbl),
            }
        )
    return records


def allocate_bin_counts(bin_sizes: List[int], target_count: int) -> List[int]:
    active_bins = [idx for idx, size in enumerate(bin_sizes) if size > 0]
    if not active_bins:
        return [0 for _ in bin_sizes]
    if target_count < len(active_bins):
        raise ValueError("Target count must be >= number of active bins for stratified sampling")

    allocation = [0 for _ in bin_sizes]
    for idx in active_bins:
        allocation[idx] = 1
    remaining = target_count - len(active_bins)

    while remaining > 0:
        candidates = [
            (bin_sizes[idx] - allocation[idx], bin_sizes[idx], -idx)
            for idx in active_bins
            if allocation[idx] < bin_sizes[idx]
        ]
        if not candidates:
            break
        candidates.sort(reverse=True)
        for _, _, neg_idx in candidates:
            idx = -neg_idx
            if allocation[idx] >= bin_sizes[idx]:
                continue
            allocation[idx] += 1
            remaining -= 1
            if remaining == 0:
                break
    return allocation


def choose_subjects(
    records: List[Dict[str, Any]],
    fraction: float,
    seed: int,
    selection_mode: str,
    subject_bins: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    subjects_total = len(records)
    n_adapt = subject_split_count(subjects_total, fraction)
    rng = np.random.default_rng(seed)

    if selection_mode == "random":
        subjects = sorted(record["subject"] for record in records)
        adapt_subjects = sorted(rng.choice(np.asarray(subjects), size=n_adapt, replace=False).tolist())
        selection_bins = []
    elif selection_mode == "stratified":
        sorted_records = sorted(records, key=lambda row: (row["stress_ratio"], row["subject"]))
        n_bins = min(max(1, subject_bins), len(sorted_records), n_adapt)
        split_bins = [list(bin_rows) for bin_rows in np.array_split(np.asarray(sorted_records, dtype=object), n_bins)]
        allocation = allocate_bin_counts([len(bin_rows) for bin_rows in split_bins], target_count=n_adapt)
        adapt_subjects = []
        selection_bins = []
        for bin_idx, (bin_rows, bin_take) in enumerate(zip(split_bins, allocation)):
            bin_list = list(bin_rows)
            if not bin_list:
                continue
            if bin_take <= 0:
                chosen = []
            elif bin_take >= len(bin_list):
                chosen = list(bin_list)
            else:
                chosen_idx = rng.choice(np.arange(len(bin_list)), size=bin_take, replace=False)
                chosen = [bin_list[int(idx)] for idx in sorted(chosen_idx.tolist())]
            adapt_subjects.extend(row["subject"] for row in chosen)
            selection_bins.append(
                {
                    "bin_index": int(bin_idx),
                    "stress_ratio_min": float(min(row["stress_ratio"] for row in bin_list)),
                    "stress_ratio_max": float(max(row["stress_ratio"] for row in bin_list)),
                    "subjects": [row["subject"] for row in bin_list],
                    "selected_subjects": [row["subject"] for row in chosen],
                    "bin_size": int(len(bin_list)),
                    "selected_count": int(len(chosen)),
                }
            )
        adapt_subjects = sorted(adapt_subjects)
    else:
        raise ValueError(f"Unsupported subject selection mode: {selection_mode}")

    return adapt_subjects, selection_bins


def merge_subject_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    features: List[List[float]] = []
    labels: List[int] = []
    for payload in payloads:
        features.extend(payload["features"])
        labels.extend(payload["label"])
    return {
        "features": features,
        "label": labels,
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    runtime = args.runtime_dir.resolve()

    stressid_merged = repo / "Data_Processed_StressID" / "WESADECG_S17.json"
    wesad_dir = repo / "Data_Processed"

    data_dir = runtime / "data"

    if args.split_mode == "window":
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

        write_json(data_dir / "STRESSID_CALIB.json", calib)
        write_json(data_dir / "STRESSID_TEST.json", test_copy)
        split_summary: Dict[str, Any] = {
            "split_mode": "window",
            "stressid_total_samples": int(len(lbl)),
            "stressid_calib_samples": int(len(calib_idx)),
            "stressid_calib_fraction": float(args.calib_fraction),
        }
    else:
        subject_entries = load_subject_files(repo / "Data_Processed_StressID")
        subject_records = build_subject_records(subject_entries)
        subjects = [record["subject"] for record in subject_records]
        adapt_subjects, selection_bins = choose_subjects(
            subject_records,
            fraction=args.subject_fraction,
            seed=args.seed,
            selection_mode=args.subject_selection,
            subject_bins=args.subject_bins,
        )
        adapt_set = set(adapt_subjects)
        eval_subjects = [subject for subject in subjects if subject not in adapt_set]

        subject_stats = []
        adapt_payloads: List[Dict[str, Any]] = []
        eval_payloads: List[Dict[str, Any]] = []

        for record in subject_records:
            subject = str(record["subject"])
            path = Path(record["path"])
            payload = record["payload"]
            samples = int(record["samples"])
            stat = {
                "subject": subject,
                "path": str(path),
                "samples": samples,
                "label_dist": record["label_dist"],
                "stress_ratio": float(record["stress_ratio"]),
            }
            subject_stats.append(stat)
            if subject in adapt_set:
                adapt_payloads.append(payload)
            else:
                eval_payloads.append(payload)

        adapt = merge_subject_payloads(adapt_payloads)
        eval_payload = merge_subject_payloads(eval_payloads)
        write_json(data_dir / "STRESSID_ADAPT.json", adapt)
        write_json(data_dir / "STRESSID_EVAL.json", eval_payload)

        manifest = {
            "split_mode": "subject",
            "seed": int(args.seed),
            "subject_fraction": float(args.subject_fraction),
            "subject_selection": args.subject_selection,
            "subject_bins": int(args.subject_bins),
            "subjects_total": int(len(subjects)),
            "adapt_subjects_count": int(len(adapt_subjects)),
            "eval_subjects_count": int(len(eval_subjects)),
            "adapt_subjects": adapt_subjects,
            "eval_subjects": eval_subjects,
            "adapt_samples": int(len(adapt["label"])),
            "eval_samples": int(len(eval_payload["label"])),
            "adapt_stress_ratio": subject_stress_ratio(np.asarray(adapt["label"], dtype=np.int64)),
            "eval_stress_ratio": subject_stress_ratio(np.asarray(eval_payload["label"], dtype=np.int64)),
            "selection_bins": selection_bins,
            "subject_stats": subject_stats,
        }
        write_json(runtime / "subject_split_manifest.json", manifest)
        split_summary = manifest

    wesad_jsons = sorted(wesad_dir.glob("WESADECG_S*.json"))
    stats_all = compute_source_stats(wesad_jsons, keep_labels=None)
    stats_neutral_stress = compute_source_stats(wesad_jsons, keep_labels=(1, 2))

    cfg_dir = runtime / "config"
    write_json(cfg_dir / "source_stats_all.json", stats_all)
    write_json(cfg_dir / "source_stats_neutral_stress.json", stats_neutral_stress)

    summary = dict(split_summary)
    summary.update(
        {
            "wesad_source_files": [str(p) for p in wesad_jsons],
            "source_stats_all_samples": int(stats_all["samples"]),
            "source_stats_neutral_stress_samples": int(stats_neutral_stress["samples"]),
        }
    )
    write_json(runtime / "asset_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
