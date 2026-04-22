"""
Evaluate SWELL mapped JSON with WESAD DNN checkpoints.

This script does not require pandas. It is intended for remote/container jobs
where mapped 12-dim JSON data has already been prepared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Evaluate mapped JSON on DNN folds")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    with args.test_json.open("r", encoding="utf-8") as f:
        test = json.load(f)
    x = np.array(test["features"], dtype=np.float32)
    lbl = np.array(test["label"], dtype=np.int64)
    y_true = (lbl == 2).astype(np.int64)

    checkpoints = load_best_checkpoints(args.results_json, args.models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_results = []
    for fold, subj, ckpt in checkpoints:
        model = ClassifierDNN().to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            score = model(torch.tensor(x, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
        y_pred = (score > args.threshold).astype(np.int64)
        m = compute_metrics(y_true, y_pred)
        fold_results.append(
            {
                "fold": fold,
                "wesad_test_subject": subj,
                "checkpoint": str(ckpt),
                "metrics": m,
            }
        )
        print(
            f"Fold {fold:02d} ({subj}) | "
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

    out = {
        "threshold": args.threshold,
        "test_samples": int(len(lbl)),
        "fold_results": fold_results,
        "average_metrics": avg,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
