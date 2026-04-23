"""
Stepwise StressID transfer evaluation with optional adaptation/calibration knobs.

This script keeps the same checkpoint loading protocol as evaluate_mapped_json.py
but adds a configurable preprocessing + calibration pipeline so each step in the
repair loop can be tested remotely with one PBS job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class ClassifierDNN(nn.Module):
    """DNN architecture aligned with training checkpoints."""

    def __init__(self) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nnECG(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate StressID with stepwise options")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--source-stats-json", type=Path, default=None)
    parser.add_argument("--calib-json", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--step-name", type=str, default="unspecified")
    return parser.parse_args()


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    x = np.asarray(data["features"], dtype=np.float32)
    lbl = np.asarray(data["label"], dtype=np.int64)
    y_bin = (lbl == 2).astype(np.int64)
    return x, y_bin, lbl


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total_samples": total,
    }


def load_best_checkpoints(results_json: Path, models_dir: Path) -> List[Tuple[int, str, Path]]:
    with results_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out: List[Tuple[int, str, Path]] = []
    for row in rows:
        fold = int(row["fold"])
        subj = str(row["test_subject"])
        epoch = int(row["DNN_best_epoch"])
        ckpt = models_dir / f"fold_{fold - 1}_DNN" / f"epoch_{epoch}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        out.append((fold, subj, ckpt))
    return out


def load_source_stats(path: Path | None) -> Tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    mean = np.asarray(obj["mean"], dtype=np.float64)
    cov = np.asarray(obj["cov"], dtype=np.float64)
    return mean, cov


def eig_sqrt(cov: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov + np.eye(cov.shape[0], dtype=np.float64) * eps)
    vals = np.clip(vals, eps, None)
    sqrt_cov = (vecs * np.sqrt(vals)) @ vecs.T
    inv_sqrt_cov = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    return sqrt_cov, inv_sqrt_cov


def build_coral_transform(
    source_mean: np.ndarray,
    source_cov: np.ndarray,
    target_fit_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    target_mean = np.mean(target_fit_x, axis=0).astype(np.float64)
    target_cov = np.cov(target_fit_x.astype(np.float64), rowvar=False)
    src_sqrt, _ = eig_sqrt(source_cov)
    _, tgt_inv_sqrt = eig_sqrt(target_cov)
    transform = tgt_inv_sqrt @ src_sqrt
    return target_mean, transform


def apply_coral(
    x: np.ndarray,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    centered = x.astype(np.float64) - target_mean.reshape(1, -1)
    aligned = centered @ transform + source_mean.reshape(1, -1)
    return aligned.astype(np.float32)


def robust_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(x, axis=0).astype(np.float64)
    q75 = np.quantile(x, 0.75, axis=0).astype(np.float64)
    q25 = np.quantile(x, 0.25, axis=0).astype(np.float64)
    iqr = np.clip(q75 - q25, 1e-6, None)
    return med, iqr


def robust_normalize(x: np.ndarray, med: np.ndarray, iqr: np.ndarray, clip: float) -> np.ndarray:
    z = (x.astype(np.float64) - med.reshape(1, -1)) / iqr.reshape(1, -1)
    z = np.clip(z, -clip, clip)
    return z.astype(np.float32)


def find_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5
    best_m = compute_metrics(y_true, (scores > best_t).astype(np.int64))
    best_f1 = best_m["f1"]
    best_acc = best_m["accuracy"]
    for t in np.linspace(0.0, 1.0, 1001):
        m = compute_metrics(y_true, (scores > t).astype(np.int64))
        f1 = m["f1"]
        acc = m["accuracy"]
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-12 and acc > best_acc):
            best_t = float(t)
            best_m = m
            best_f1 = f1
            best_acc = acc
    return best_t, best_m


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = safe_logit(scores)
    scaled = logits / max(temperature, 1e-3)
    out = 1.0 / (1.0 + np.exp(-scaled))
    return out.astype(np.float32)


def fit_temperature(scores: np.ndarray, y_true: np.ndarray) -> float:
    logits = torch.tensor(safe_logit(scores), dtype=torch.float32)
    labels = torch.tensor(y_true, dtype=torch.float32)
    log_t = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_t], lr=0.2, max_iter=100, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        t = torch.exp(log_t) + 1e-3
        scaled_logits = logits / t
        loss = loss_fn(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).item() + 1e-3)
    return temperature


def score_model(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            sb = model(xb).view(-1).detach().cpu().numpy()
            scores.append(sb)
    return np.concatenate(scores, axis=0).astype(np.float32)


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "step_name": args.step_name,
        "threshold": args.threshold,
        "enable_coral": False,
        "enable_robust_norm": False,
        "robust_clip": 8.0,
        "enable_temp_scaling": False,
        "enable_threshold_search": False,
        "enable_quality_filter": False,
        "quality_z_threshold": 6.0,
    }
    if args.config_json is not None:
        with args.config_json.open("r", encoding="utf-8-sig") as f:
            loaded = json.load(f)
        cfg.update(loaded)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    source_stats = load_source_stats(args.source_stats_json)
    x_test_raw, y_test, lbl_test = load_dataset(args.test_json)

    if args.calib_json is not None:
        x_calib_raw, y_calib, _ = load_dataset(args.calib_json)
    else:
        x_calib_raw, y_calib = x_test_raw, y_test

    x_test_proc = np.asarray(x_test_raw, dtype=np.float32)
    x_calib_proc = np.asarray(x_calib_raw, dtype=np.float32)

    if cfg.get("enable_coral", False):
        if source_stats is None:
            raise ValueError("enable_coral=true requires --source-stats-json")
        source_mean, source_cov = source_stats
        target_mean, transform = build_coral_transform(source_mean, source_cov, x_calib_proc)
        x_test_proc = apply_coral(x_test_proc, source_mean, target_mean, transform)
        x_calib_proc = apply_coral(x_calib_proc, source_mean, target_mean, transform)

    robust_center, robust_scale = robust_stats(x_calib_proc)
    if cfg.get("enable_robust_norm", False):
        robust_clip = float(cfg.get("robust_clip", 8.0))
        x_test_proc = robust_normalize(x_test_proc, robust_center, robust_scale, robust_clip)
        x_calib_proc = robust_normalize(x_calib_proc, robust_center, robust_scale, robust_clip)
        robust_center, robust_scale = robust_stats(x_calib_proc)

    checkpoints = load_best_checkpoints(args.results_json, args.models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results: List[Dict[str, Any]] = []

    prior_prob = float(np.mean(y_calib))
    z_threshold = float(cfg.get("quality_z_threshold", 6.0))

    for fold, subj, ckpt in checkpoints:
        model = ClassifierDNN().to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        calib_scores = score_model(model, x_calib_proc, device=device)
        test_scores = score_model(model, x_test_proc, device=device)

        used_temperature = 1.0
        if cfg.get("enable_temp_scaling", False):
            used_temperature = fit_temperature(calib_scores, y_calib)
            calib_scores = apply_temperature(calib_scores, used_temperature)
            test_scores = apply_temperature(test_scores, used_temperature)

        if cfg.get("enable_quality_filter", False):
            z_test = np.abs((x_test_proc.astype(np.float64) - robust_center.reshape(1, -1)) / robust_scale.reshape(1, -1))
            noisy_mask = np.any(z_test > z_threshold, axis=1)
            test_scores = np.where(noisy_mask, prior_prob, test_scores).astype(np.float32)
        else:
            noisy_mask = np.zeros(len(test_scores), dtype=bool)

        used_threshold = float(cfg.get("threshold", args.threshold))
        calib_best = compute_metrics(y_calib, (calib_scores > used_threshold).astype(np.int64))
        if cfg.get("enable_threshold_search", False):
            used_threshold, calib_best = find_best_threshold(calib_scores, y_calib)

        y_pred = (test_scores > used_threshold).astype(np.int64)
        m = compute_metrics(y_test, y_pred)

        fold_results.append(
            {
                "fold": fold,
                "wesad_test_subject": subj,
                "checkpoint": str(ckpt),
                "metrics": m,
                "selected_threshold": float(used_threshold),
                "selected_temperature": float(used_temperature),
                "calib_metrics_at_selected_threshold": calib_best,
                "quality_replaced_samples": int(np.sum(noisy_mask)),
            }
        )
        print(
            f"Fold {fold:02d} ({subj}) | "
            f"Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} "
            f"Rec={m['recall']:.4f} F1={m['f1']:.4f} "
            f"T={used_temperature:.4f} Thr={used_threshold:.4f}"
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

    out = {
        "step_name": cfg.get("step_name", args.step_name),
        "config": cfg,
        "source_stats_json": str(args.source_stats_json) if args.source_stats_json else None,
        "calib_json": str(args.calib_json) if args.calib_json else None,
        "test_json": str(args.test_json),
        "test_samples": int(len(lbl_test)),
        "fold_results": fold_results,
        "average_metrics": avg,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
