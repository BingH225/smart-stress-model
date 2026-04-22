"""
Approximate SWELL->WESAD12 mapping and external evaluation.

This script builds a proxy 12-dimensional feature vector from SWELL HRV
engineered features, applies subject-level neutral normalization, and
evaluates existing WESAD-trained DNN checkpoints.

The 12 output features follow the exact order used by the current DNN:
1. Mean Freq
2. Std Freq
3. TINN
4. HRV Index
5. NN50
6. pNN50
7. Mean HRV
8. Std HRV
9. RMSSD
10. FFT Mean
11. FFT Std
12. Sum PSD
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except Exception as ex:  # pragma: no cover - runtime environment specific
    torch = None
    TORCH_IMPORT_ERROR = ex
else:
    TORCH_IMPORT_ERROR = None


FEATURE_NAMES: List[str] = [
    "Mean Freq",
    "Std Freq",
    "TINN",
    "HRV Index",
    "NN50",
    "pNN50",
    "Mean HRV",
    "Std HRV",
    "RMSSD",
    "FFT Mean",
    "FFT Std",
    "Sum PSD",
]

STRESS_CONDITIONS = {"interruption", "time pressure"}
NON_STRESS_CONDITION = "no stress"
EPS = 1e-8


def build_dnn_model():
    """Build DNN architecture aligned with Cross_validation_ablation.py."""
    if torch is None:
        raise RuntimeError(
            f"PyTorch import failed: {TORCH_IMPORT_ERROR}"
        )
    import torch.nn as nn

    class _ClassifierDNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Keep layer container name aligned with training code checkpoints.
            self.nnECG = nn.Sequential(
                nn.Linear(12, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64, bias=True),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 16, bias=True),
                nn.BatchNorm1d(16),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                nn.Linear(16, 4, bias=True),
                nn.BatchNorm1d(4),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                nn.Linear(4, 1, bias=True),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.nnECG(x)

    return _ClassifierDNN()


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    total: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "total_samples": self.total,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate SWELL feature mapping and DNN evaluation"
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("D:/NUS/BMI5101/other_dataset/combined-swell-classification-hrv-train-dataset.csv"),
        help="Path to SWELL train CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("D:/NUS/BMI5101/other_dataset/combined-swell-classification-hrv-test-dataset.csv"),
        help="Path to SWELL test CSV",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("Results_CrossVal_Full/cross_val_results.json"),
        help="Path to CrossVal results JSON containing best DNN epochs",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("Models_CrossVal_Full"),
        help="Directory containing fold_<k>_DNN/epoch_<n>.pth checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results_SWELL_Approx"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for fast smoke tests; 0 means all rows",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Binary decision threshold for stress probability",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only generate mapped JSON files and skip checkpoint evaluation",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: Sequence[str], source_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


def load_csv(path: Path, max_rows: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    nrows = max_rows if max_rows > 0 else None
    df = pd.read_csv(path, nrows=nrows)
    return df


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def map_swell_to_proxy_12(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build proxy 12-D features from SWELL engineered HRV columns.

    Approximation strategy:
    - Convert ms-domain timing features to seconds where needed
    - Approximate frequency spread from RR variability via delta method
    - Use SD2 as a proxy for TINN width
    - Use SDRR/RMSSD as proxy for HRV Index
    - Approximate NN50 from pNN50 and estimated beat count in 20-second window
    - Use TP as total spectral power proxy
    - Approximate FFT mean/std from RR-derived Nyquist relations
    """
    required_cols = [
        "MEAN_RR",
        "SDRR",
        "RMSSD",
        "pNN50",
        "HR",
        "SD2",
        "TP",
    ]
    ensure_columns(df, required_cols, "SWELL CSV")

    rr_ms = _safe_numeric(df["MEAN_RR"])
    sdrr_ms = _safe_numeric(df["SDRR"])
    rmssd_ms = _safe_numeric(df["RMSSD"])
    pnn50_pct = _safe_numeric(df["pNN50"])
    hr_bpm = _safe_numeric(df["HR"])
    sd2 = _safe_numeric(df["SD2"])
    tp = _safe_numeric(df["TP"])

    rr_s = rr_ms / 1000.0
    sdrr_s = sdrr_ms / 1000.0
    rmssd_s = rmssd_ms / 1000.0
    pnn50_ratio = pnn50_pct / 100.0

    mean_freq = hr_bpm / 60.0
    std_freq = (1000.0 * sdrr_ms) / np.maximum(rr_ms * rr_ms, EPS)
    tinn_proxy = 2.0 * sd2
    hrv_index_proxy = sdrr_ms / np.maximum(rmssd_ms, EPS)
    beats_20s = 20.0 / np.maximum(rr_s, EPS)
    nn50_proxy = pnn50_ratio * beats_20s

    nyquist = 1.0 / np.maximum(2.0 * rr_s, EPS)
    fft_mean_proxy = nyquist / 2.0
    fft_std_proxy = nyquist / math.sqrt(12.0)
    sum_psd_proxy = tp

    mapped = pd.DataFrame(
        {
            "Mean Freq": mean_freq,
            "Std Freq": std_freq,
            "TINN": tinn_proxy,
            "HRV Index": hrv_index_proxy,
            "NN50": nn50_proxy,
            "pNN50": pnn50_ratio,
            "Mean HRV": rr_s,
            "Std HRV": sdrr_s,
            "RMSSD": rmssd_s,
            "FFT Mean": fft_mean_proxy,
            "FFT Std": fft_std_proxy,
            "Sum PSD": sum_psd_proxy,
        }
    )
    mapped = mapped.replace([np.inf, -np.inf], np.nan)
    return mapped


def normalize_by_subject_neutral(mapped: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate WESAD-style baseline normalization per subject:
    each row feature is divided by subject neutral median.
    """
    ensure_columns(meta_df, ["subject_id", "condition"], "SWELL CSV metadata")

    cond = meta_df["condition"].astype(str).str.strip().str.lower()
    subj = meta_df["subject_id"].astype(str).str.strip()

    out = mapped.copy()
    out["__subject_id"] = subj.values
    out["__condition"] = cond.values

    for sid, group_idx in out.groupby("__subject_id").groups.items():
        idx = list(group_idx)
        group = out.loc[idx]
        neutral = group[group["__condition"] == NON_STRESS_CONDITION]
        base = neutral[FEATURE_NAMES].median(axis=0, skipna=True)
        if base.isna().all():
            base = group[FEATURE_NAMES].median(axis=0, skipna=True)
        base = base.fillna(1.0)
        base = np.maximum(base.values.astype(np.float64), EPS)
        vals = out.loc[idx, FEATURE_NAMES].values.astype(np.float64)
        out.loc[idx, FEATURE_NAMES] = vals / base

    out = out.drop(columns=["__subject_id", "__condition"])
    out = out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return out


def map_labels(df: pd.DataFrame) -> np.ndarray:
    ensure_columns(df, ["condition"], "SWELL CSV labels")
    cond = df["condition"].astype(str).str.strip().str.lower()
    labels = np.where(cond == NON_STRESS_CONDITION, 1, np.where(cond.isin(STRESS_CONDITIONS), 2, 0))
    return labels.astype(np.int64)


def build_json_dict(features: np.ndarray, labels: np.ndarray) -> Dict[str, List]:
    keep = labels > 0
    features = features[keep]
    labels = labels[keep]
    return {
        "features": features.tolist(),
        "label": labels.tolist(),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return EvalMetrics(acc, precision, recall, f1, tp, tn, fp, fn, total)


def load_best_dnn_checkpoints(results_json: Path, models_dir: Path) -> List[Tuple[str, Path]]:
    if not results_json.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_json}")
    with results_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    items: List[Tuple[str, Path]] = []
    for row in rows:
        fold = int(row["fold"])
        best_epoch = int(row["DNN_best_epoch"])
        subject = str(row["test_subject"])
        ckpt = models_dir / f"fold_{fold - 1}_DNN" / f"epoch_{best_epoch}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint missing for fold {fold}: {ckpt}")
        items.append((subject, ckpt))
    return items


def eval_checkpoint(
    checkpoint: Path,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    device,
) -> EvalMetrics:
    model = build_dnn_model().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        score = model(x).view(-1).detach().cpu().numpy()
    pred = (score > threshold).astype(np.int64)
    y = (labels == 2).astype(np.int64)
    return compute_metrics(y, pred)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def merge_preserving_columns(features_df: pd.DataFrame, src_df: pd.DataFrame) -> pd.DataFrame:
    out = features_df.copy()
    out["condition"] = src_df["condition"].astype(str).values
    out["subject_id"] = src_df["subject_id"].astype(str).values
    return out


def print_mapping_notes() -> None:
    print("Approximation notes:")
    print("- Mean Freq ~= HR/60")
    print("- Std Freq ~= 1000*SDRR/(MEAN_RR^2)")
    print("- TINN ~= 2*SD2")
    print("- HRV Index ~= SDRR/RMSSD")
    print("- NN50 ~= pNN50 * (20 / RR)")
    print("- Mean/Std HRV from MEAN_RR/SDRR (ms->s)")
    print("- RMSSD from RMSSD (ms->s)")
    print("- FFT Mean/Std from RR-based Nyquist approximation")
    print("- Sum PSD ~= TP")


def run() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_mapping_notes()
    print(f"Loading train CSV: {args.train_csv}")
    train_df = load_csv(args.train_csv, max_rows=args.max_rows)
    print(f"Loading test CSV: {args.test_csv}")
    test_df = load_csv(args.test_csv, max_rows=args.max_rows)

    train_map_raw = map_swell_to_proxy_12(train_df)
    test_map_raw = map_swell_to_proxy_12(test_df)

    train_map_norm = normalize_by_subject_neutral(train_map_raw, train_df)
    test_map_norm = normalize_by_subject_neutral(test_map_raw, test_df)

    # Re-align labels after row drops from NaN filtering
    train_meta = merge_preserving_columns(train_map_norm, train_df.loc[train_map_norm.index].reset_index(drop=True))
    test_meta = merge_preserving_columns(test_map_norm, test_df.loc[test_map_norm.index].reset_index(drop=True))

    train_labels = map_labels(train_meta)
    test_labels = map_labels(test_meta)
    train_feats = train_map_norm[FEATURE_NAMES].values.astype(np.float32)
    test_feats = test_map_norm[FEATURE_NAMES].values.astype(np.float32)

    train_json = build_json_dict(train_feats, train_labels)
    test_json = build_json_dict(test_feats, test_labels)

    train_json_path = output_dir / "SWELL_APPROX_TRAIN.json"
    test_json_path = output_dir / "SWELL_APPROX_TEST.json"
    save_json(train_json_path, train_json)
    save_json(test_json_path, test_json)
    print(f"Saved mapped train JSON: {train_json_path}")
    print(f"Saved mapped test JSON: {test_json_path}")

    if args.skip_eval:
        print("Skip eval requested. Mapping outputs are ready.")
        return

    if torch is None:
        raise RuntimeError(
            "PyTorch is not available in this Python environment. "
            "Use --skip-eval for mapping only, or run in a PyTorch-ready environment. "
            f"Import error: {TORCH_IMPORT_ERROR}"
        )

    eval_features = np.array(test_json["features"], dtype=np.float32)
    eval_labels = np.array(test_json["label"], dtype=np.int64)

    checkpoints = load_best_dnn_checkpoints(args.results_json, args.models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    fold_results: List[Dict] = []
    for fold_idx, (subject, ckpt) in enumerate(checkpoints, start=1):
        metrics = eval_checkpoint(
            checkpoint=ckpt,
            features=eval_features,
            labels=eval_labels,
            threshold=args.score_threshold,
            device=device,
        )
        row = {
            "fold": fold_idx,
            "wesad_test_subject": subject,
            "checkpoint": str(ckpt),
            "metrics": metrics.to_dict(),
        }
        fold_results.append(row)
        m = metrics.to_dict()
        print(
            f"Fold {fold_idx:02d} ({subject}) | "
            f"Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} "
            f"Rec={m['recall']:.4f} F1={m['f1']:.4f}"
        )

    avg = {
        "accuracy": float(np.mean([r["metrics"]["accuracy"] for r in fold_results])),
        "precision": float(np.mean([r["metrics"]["precision"] for r in fold_results])),
        "recall": float(np.mean([r["metrics"]["recall"] for r in fold_results])),
        "f1": float(np.mean([r["metrics"]["f1"] for r in fold_results])),
    }
    print(
        "Average over folds | "
        f"Acc={avg['accuracy']:.4f} Prec={avg['precision']:.4f} "
        f"Rec={avg['recall']:.4f} F1={avg['f1']:.4f}"
    )

    out_report = {
        "feature_order": FEATURE_NAMES,
        "mapping_type": "approximate",
        "normalization": "subject neutral median ratio",
        "train_rows": len(train_json["label"]),
        "test_rows": len(test_json["label"]),
        "threshold": args.score_threshold,
        "fold_results": fold_results,
        "average_metrics": avg,
    }
    report_path = output_dir / "swell_approx_eval_report.json"
    save_json(report_path, out_report)
    print(f"Saved evaluation report: {report_path}")


if __name__ == "__main__":
    run()
