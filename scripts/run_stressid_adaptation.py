from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ClassifierDNN(nn.Module):
    """DNN architecture aligned with existing WESAD checkpoints."""

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
    parser = argparse.ArgumentParser(description="Run StressID lightweight adaptation experiments")
    parser.add_argument("--adapt-json", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--source-stats-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--comparison-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--train-objective",
        type=str,
        choices=("bce", "weighted_bce", "focal", "soft_f1_bce"),
        default="bce",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--soft-f1-lambda", type=float, default=0.5)
    parser.add_argument(
        "--threshold-mode",
        type=str,
        choices=("fixed_0.5", "val_f1"),
        default="fixed_0.5",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument(
        "--finetune-input-space",
        type=str,
        choices=("raw", "coral"),
        default="raw",
    )
    parser.add_argument(
        "--finetune-scope",
        type=str,
        choices=("final_layer", "last_block"),
        default="final_layer",
    )
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = read_json(path)
    x = np.asarray(obj["features"], dtype=np.float32)
    lbl = np.asarray(obj["label"], dtype=np.int64)
    y = (lbl == 2).astype(np.int64)
    return x, y, lbl


def label_dist(lbl: np.ndarray) -> Dict[str, int]:
    return {str(int(v)): int(np.sum(lbl == v)) for v in sorted(np.unique(lbl))}


def stratified_split_indices(y: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    pos = idx[y == 1]
    neg = idx[y == 0]

    def choose(source: np.ndarray) -> np.ndarray:
        if len(source) <= 1:
            return source.copy()
        take = int(round(len(source) * fraction))
        take = max(1, take)
        take = min(len(source) - 1, take)
        return rng.choice(source, size=take, replace=False)

    val_pos = choose(pos)
    val_neg = choose(neg)
    val_idx = np.concatenate([val_pos, val_neg], axis=0)
    val_idx.sort()
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = idx[train_mask]
    return train_idx, val_idx


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
        "positive_rate": float(np.mean(y_pred == 1)) if total else 0.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total_samples": total,
    }


def average_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ("accuracy", "precision", "recall", "f1", "positive_rate")
    return {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys}


def load_best_checkpoints(results_json: Path, models_dir: Path) -> List[Tuple[int, str, Path]]:
    rows = read_json(results_json)
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


def load_source_stats(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = read_json(path)
    return np.asarray(obj["mean"], dtype=np.float64), np.asarray(obj["cov"], dtype=np.float64)


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


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def score_model(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            scores.append(model(xb).view(-1).cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y.astype(np.float32).reshape(-1, 1), dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_finetune_modules(model: ClassifierDNN, scope: str) -> Tuple[nn.Sequential, nn.Sequential]:
    children = list(model.nnECG.children())
    if scope == "final_layer":
        split_idx = 16
    elif scope == "last_block":
        split_idx = 12
    else:
        raise ValueError(f"Unsupported finetune scope: {scope}")
    backbone = nn.Sequential(*children[:split_idx])
    head = nn.Sequential(*children[split_idx:])
    return backbone, head


def weighted_bce_loss(
    out: torch.Tensor,
    y: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
) -> torch.Tensor:
    if abs(pos_weight - 1.0) < 1e-12 and abs(neg_weight - 1.0) < 1e-12:
        return F.binary_cross_entropy(out, y)
    weights = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.full_like(y, neg_weight))
    return F.binary_cross_entropy(out, y, weight=weights)


def focal_loss(
    out: torch.Tensor,
    y: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
    gamma: float,
) -> torch.Tensor:
    probs = out.clamp(min=1e-6, max=1.0 - 1e-6)
    ce = F.binary_cross_entropy(probs, y, reduction="none")
    pt = torch.where(y > 0.5, probs, 1.0 - probs)
    weights = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.full_like(y, neg_weight))
    return (weights * torch.pow(1.0 - pt, gamma) * ce).mean()


def soft_f1_bce_loss(
    out: torch.Tensor,
    y: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
    soft_f1_lambda: float,
) -> torch.Tensor:
    bce = weighted_bce_loss(out, y, pos_weight=pos_weight, neg_weight=neg_weight)
    probs = out.clamp(min=1e-6, max=1.0 - 1e-6)
    soft_tp = torch.sum(probs * y)
    soft_fp = torch.sum(probs * (1.0 - y))
    soft_fn = torch.sum((1.0 - probs) * y)
    soft_f1 = (2.0 * soft_tp) / (2.0 * soft_tp + soft_fp + soft_fn + 1e-6)
    return (1.0 - soft_f1_lambda) * bce + soft_f1_lambda * (1.0 - soft_f1)


def training_loss(
    out: torch.Tensor,
    y: torch.Tensor,
    objective: str,
    pos_weight: float,
    neg_weight: float,
    focal_gamma: float,
    soft_f1_lambda: float,
) -> torch.Tensor:
    if objective == "bce":
        return weighted_bce_loss(out, y, pos_weight=1.0, neg_weight=1.0)
    if objective == "weighted_bce":
        return weighted_bce_loss(out, y, pos_weight=pos_weight, neg_weight=neg_weight)
    if objective == "focal":
        return focal_loss(out, y, pos_weight=pos_weight, neg_weight=neg_weight, gamma=focal_gamma)
    if objective == "soft_f1_bce":
        return soft_f1_bce_loss(
            out,
            y,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            soft_f1_lambda=soft_f1_lambda,
        )
    raise ValueError(f"Unsupported training objective: {objective}")


def select_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    mode: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Tuple[float, Dict[str, float]]:
    if mode == "fixed_0.5":
        pred = (scores > 0.5).astype(np.int64)
        return 0.5, compute_metrics(y_true, pred)

    grid = np.linspace(threshold_min, threshold_max, num=threshold_steps, dtype=np.float64)
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, (scores > 0.5).astype(np.int64))
    for threshold in grid.tolist():
        pred = (scores > threshold).astype(np.int64)
        metrics = compute_metrics(y_true, pred)
        improved = (
            metrics["f1"] > best_metrics["f1"]
            or (
                abs(metrics["f1"] - best_metrics["f1"]) < 1e-12
                and metrics["accuracy"] > best_metrics["accuracy"]
            )
            or (
                abs(metrics["f1"] - best_metrics["f1"]) < 1e-12
                and abs(metrics["accuracy"] - best_metrics["accuracy"]) < 1e-12
                and metrics["precision"] > best_metrics["precision"]
            )
        )
        if improved:
            best_threshold = float(threshold)
            best_metrics = metrics
    return float(best_threshold), best_metrics


def finetune_last_layer(
    model: ClassifierDNN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    pos_weight: float,
    neg_weight: float,
    weight_decay: float,
    scope: str,
    objective: str,
    focal_gamma: float,
    soft_f1_lambda: float,
    threshold_mode: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Dict[str, Any]:
    backbone, head = build_finetune_modules(model, scope=scope)
    backbone.to(device)
    head.to(device)
    backbone.eval()
    head.train()

    for param in backbone.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = build_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = build_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    best_threshold = 0.5
    best_train_loss = None
    best_val_loss = None
    best_val_metrics = compute_metrics(y_val, np.zeros_like(y_val))
    patience_left = patience

    for epoch in range(1, epochs + 1):
        head.train()
        train_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feat = backbone(xb)
            out = head(feat)
            loss = training_loss(
                out,
                yb,
                objective=objective,
                pos_weight=pos_weight,
                neg_weight=neg_weight,
                focal_gamma=focal_gamma,
                soft_f1_lambda=soft_f1_lambda,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        head.eval()
        val_losses: List[float] = []
        val_scores: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                feat = backbone(xb)
                out = head(feat)
                val_losses.append(
                    float(
                        training_loss(
                            out,
                            yb,
                            objective=objective,
                            pos_weight=pos_weight,
                            neg_weight=neg_weight,
                            focal_gamma=focal_gamma,
                            soft_f1_lambda=soft_f1_lambda,
                        ).item()
                    )
                )
                val_scores.append(out.view(-1).cpu().numpy())

        val_score = np.concatenate(val_scores, axis=0).astype(np.float32)
        selected_threshold, val_metrics = select_threshold(
            val_score,
            y_val,
            mode=threshold_mode,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_steps=threshold_steps,
        )
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        improved = (
            val_metrics["f1"] > best_val_metrics["f1"]
            or (
                abs(val_metrics["f1"] - best_val_metrics["f1"]) < 1e-12
                and val_metrics["accuracy"] > best_val_metrics["accuracy"]
            )
        )
        if improved:
            best_state = copy.deepcopy(head.state_dict())
            best_epoch = epoch
            best_threshold = selected_threshold
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    head.load_state_dict(best_state)
    head.eval()

    eval_scores: List[np.ndarray] = []
    eval_loader = build_loader(x_eval, y_eval, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, _ in eval_loader:
            xb = xb.to(device)
            feat = backbone(xb)
            eval_scores.append(head(feat).view(-1).cpu().numpy())
    eval_score = np.concatenate(eval_scores, axis=0).astype(np.float32)
    eval_pred = (eval_score > best_threshold).astype(np.int64)
    eval_metrics = compute_metrics(y_eval, eval_pred)
    return {
        "metrics": eval_metrics,
        "selected_epoch": int(best_epoch),
        "selected_threshold": float(best_threshold),
        "adapt_train_loss": float(best_train_loss if best_train_loss is not None else 0.0),
        "adapt_val_loss": float(best_val_loss if best_val_loss is not None else 0.0),
        "adapt_val_metrics": best_val_metrics,
    }


def summarize_method(name: str, avg: Dict[str, float]) -> str:
    return (
        f"| {name} | {avg['accuracy']:.4f} | {avg['precision']:.4f} | "
        f"{avg['recall']:.4f} | {avg['f1']:.4f} |"
    )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_adapt, y_adapt, lbl_adapt = load_dataset(args.adapt_json)
    x_eval, y_eval, lbl_eval = load_dataset(args.eval_json)
    train_idx, val_idx = stratified_split_indices(y_adapt, fraction=0.2, seed=args.seed)
    x_train = x_adapt[train_idx]
    y_train = y_adapt[train_idx]
    x_val = x_adapt[val_idx]
    y_val = y_adapt[val_idx]

    source_mean, source_cov = load_source_stats(args.source_stats_json)
    target_mean, coral_transform = build_coral_transform(source_mean, source_cov, x_adapt)
    x_train_coral = apply_coral(x_train, source_mean, target_mean, coral_transform)
    x_val_coral = apply_coral(x_val, source_mean, target_mean, coral_transform)
    x_eval_coral = apply_coral(x_eval, source_mean, target_mean, coral_transform)

    if args.finetune_input_space == "coral":
        x_train_ft = x_train_coral
        x_val_ft = x_val_coral
        x_eval_ft = x_eval_coral
    else:
        x_train_ft = x_train
        x_val_ft = x_val
        x_eval_ft = x_eval

    checkpoints = load_best_checkpoints(args.results_json, args.models_dir)
    if args.fold_limit is not None:
        checkpoints = checkpoints[: args.fold_limit]

    baseline_rows: List[Dict[str, Any]] = []
    coral_rows: List[Dict[str, Any]] = []
    finetune_rows: List[Dict[str, Any]] = []

    for fold, subj, ckpt in checkpoints:
        state = torch.load(ckpt, map_location=device)

        baseline_model = ClassifierDNN().to(device)
        baseline_model.load_state_dict(state)
        baseline_scores = score_model(baseline_model, x_eval, device=device, batch_size=args.batch_size)
        baseline_metrics = compute_metrics(y_eval, (baseline_scores > 0.5).astype(np.int64))
        baseline_rows.append(
            {
                "fold": fold,
                "wesad_test_subject": subj,
                "checkpoint": str(ckpt),
                "metrics": baseline_metrics,
            }
        )

        coral_model = ClassifierDNN().to(device)
        coral_model.load_state_dict(state)
        coral_scores = score_model(coral_model, x_eval_coral, device=device, batch_size=args.batch_size)
        coral_metrics = compute_metrics(y_eval, (coral_scores > 0.5).astype(np.int64))
        coral_rows.append(
            {
                "fold": fold,
                "wesad_test_subject": subj,
                "checkpoint": str(ckpt),
                "metrics": coral_metrics,
            }
        )

        finetune_model = ClassifierDNN().to(device)
        finetune_model.load_state_dict(state)
        finetune_out = finetune_last_layer(
            model=finetune_model,
            x_train=x_train_ft,
            y_train=y_train,
            x_val=x_val_ft,
            y_val=y_val,
            x_eval=x_eval_ft,
            y_eval=y_eval,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            pos_weight=args.pos_weight,
            neg_weight=args.neg_weight,
            weight_decay=args.weight_decay,
            scope=args.finetune_scope,
            objective=args.train_objective,
            focal_gamma=args.focal_gamma,
            soft_f1_lambda=args.soft_f1_lambda,
            threshold_mode=args.threshold_mode,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_steps=args.threshold_steps,
        )
        finetune_rows.append(
            {
                "fold": fold,
                "wesad_test_subject": subj,
                "checkpoint": str(ckpt),
                "metrics": finetune_out["metrics"],
                "selected_epoch": finetune_out["selected_epoch"],
                "selected_threshold": finetune_out["selected_threshold"],
                "adapt_train_loss": finetune_out["adapt_train_loss"],
                "adapt_val_loss": finetune_out["adapt_val_loss"],
                "adapt_val_metrics": finetune_out["adapt_val_metrics"],
            }
        )

        print(
            f"Fold {fold:02d} ({subj}) | "
            f"baseline F1={baseline_metrics['f1']:.4f} | "
            f"coral F1={coral_metrics['f1']:.4f} | "
            f"finetune F1={finetune_out['metrics']['f1']:.4f}"
        )

    baseline_avg = average_metrics(baseline_rows)
    coral_avg = average_metrics(coral_rows)
    finetune_avg = average_metrics(finetune_rows)

    report = {
        "protocol": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "patience": int(args.patience),
            "pos_weight": float(args.pos_weight),
            "neg_weight": float(args.neg_weight),
            "weight_decay": float(args.weight_decay),
            "train_objective": args.train_objective,
            "focal_gamma": float(args.focal_gamma),
            "soft_f1_lambda": float(args.soft_f1_lambda),
            "threshold_mode": args.threshold_mode,
            "threshold_min": float(args.threshold_min),
            "threshold_max": float(args.threshold_max),
            "threshold_steps": int(args.threshold_steps),
            "finetune_input_space": args.finetune_input_space,
            "finetune_scope": args.finetune_scope,
            "split": {
                "adapt_samples": int(len(y_adapt)),
                "eval_samples": int(len(y_eval)),
                "adapt_train_samples": int(len(y_train)),
                "adapt_val_samples": int(len(y_val)),
                "adapt_label_dist": label_dist(lbl_adapt),
                "eval_label_dist": label_dist(lbl_eval),
            },
        },
        "methods": {
            "baseline": {
                "fold_results": baseline_rows,
                "average_metrics": baseline_avg,
            },
            "coral": {
                "fold_results": coral_rows,
                "average_metrics": coral_avg,
            },
            "final_layer_finetune": {
                "fold_results": finetune_rows,
                "average_metrics": finetune_avg,
            },
        },
    }
    write_json(args.output_json, report)

    comparison = {
        "protocol": report["protocol"],
            "methods": {
            "baseline": baseline_avg,
            "coral": coral_avg,
            "final_layer_finetune": finetune_avg,
        },
        "delta_vs_baseline": {
            "coral": {key: float(coral_avg[key] - baseline_avg[key]) for key in baseline_avg},
            "final_layer_finetune": {key: float(finetune_avg[key] - baseline_avg[key]) for key in baseline_avg},
        },
    }
    if args.comparison_json is not None:
        write_json(args.comparison_json, comparison)

    if args.summary_md is not None:
        summary_lines = [
            "# StressID External Validation Summary",
            "",
            "## Protocol",
            "",
            "- StressID split: subject-aware adaptation/eval split",
            f"- Adaptation seed: {args.seed}",
            f"- Adaptation train/val split: {len(y_train)} / {len(y_val)} windows",
            f"- Held-out evaluation windows: {len(y_eval)}",
            f"- Fine-tuning: {args.finetune_scope} in {args.finetune_input_space} space, objective={args.train_objective}, threshold={args.threshold_mode}, {args.epochs} max epochs, lr={args.lr}, patience={args.patience}",
            "",
            "## Average Metrics",
            "",
            "| Method | Accuracy | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
            summarize_method("StressID zero-shot", baseline_avg),
            summarize_method("StressID + CORAL", coral_avg),
            summarize_method("StressID + final-layer adaptation", finetune_avg),
            "",
            "## Manuscript Positioning",
            "",
        ]
        if finetune_avg["f1"] > baseline_avg["f1"]:
            summary_lines.append(
                "Minimal target-domain adaptation improved held-out StressID transfer relative to the zero-shot baseline, but the remaining gap still supports the domain-shift narrative."
            )
        else:
            summary_lines.append(
                "Simple final-layer adaptation did not outperform the zero-shot baseline on held-out StressID, which should be framed as evidence that lightweight adaptation alone is insufficient under strong domain shift."
            )
        summary_lines.extend(
            [
                "",
                "The Results section should present StressID as an external validation challenge, and the Discussion should connect the remaining transfer gap to the need for personalization-aware, closed-loop deployment.",
            ]
        )
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
