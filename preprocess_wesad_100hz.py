import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import heartpy
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
import scipy.stats as stats
from dotenv import load_dotenv


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WESAD ECG feature JSON at 100Hz for separate experiments."
    )
    parser.add_argument(
        "--wesad-dir",
        type=Path,
        default=Path(os.getenv("DIR_WESAD", "D:/NUS/BMI5101/WESAD/")),
        help="Directory containing WESAD subject folders (S2...S17).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("D:/NUS/BMI5101/smart-stress-model/Data_Processed_100Hz"),
        help="Output folder for WESADECG_*.json files.",
    )
    parser.add_argument(
        "--source-fs",
        type=int,
        default=700,
        help="Original ECG sampling rate in WESAD.",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=100,
        help="Target ECG sampling rate for this experiment.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=20,
        help="Sliding window length in seconds.",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=1,
        help="Sliding step in seconds.",
    )
    return parser.parse_args()


def tinn(x: np.ndarray):
    kernel = stats.gaussian_kde(x)
    absi = np.linspace(np.min(x), np.max(x), len(x))
    val = kernel.evaluate(absi)
    ecart = absi[1] - absi[0]
    maxind = np.argmax(val)
    max_pos = absi[maxind]
    maxvalue = np.amax(val)
    n_abs = absi[0 : maxind + 1]
    m_abs = absi[maxind:]
    hrv_index = len(x) / maxvalue
    err_n = []
    err_m = []

    for i in range(0, len(n_abs) - 1):
        n = n_abs[i]
        slope = maxvalue / (max_pos - n)
        d = val[0 : maxind + 1]
        q = np.clip(slope * ecart * np.arange(-i, -i + maxind + 1), 0, None)
        diff = d - q
        err = np.multiply(diff, diff)
        err1 = np.delete(err, -1)
        err2 = np.delete(err, 0)
        errint = (err1 + err2) / 2
        errtot = np.linalg.norm(errint)
        err_n.append((errtot, n))

    for i in range(1, len(m_abs)):
        m = m_abs[i]
        slope = maxvalue / (max_pos - m)
        d = val[maxind:]
        q = np.clip(slope * ecart * np.arange(-i, len(d) - i), 0, None)
        diff = d - q
        err = np.multiply(diff, diff)
        err1 = np.delete(err, -1)
        err2 = np.delete(err, 0)
        errint = (err1 + err2) / 2
        errtot = np.linalg.norm(errint)
        err_m.append((errtot, m))

    return err_n, err_m, hrv_index


def best_tinn(x: np.ndarray):
    err_n, err_m, hrv_index = tinn(x)
    n_idx = np.argmin(np.array(err_n, dtype=object)[:, 0])
    m_idx = np.argmin(np.array(err_m, dtype=object)[:, 0])
    abs_n = err_n[n_idx][1]
    abs_m = err_m[m_idx][1]
    return float(abs_n), float(abs_m), float(abs_m - abs_n), hrv_index


def compare_nn50(x: np.ndarray):
    k = 0
    for i in range(0, len(x)):
        ref = x[i]
        diff = np.absolute(x - ref)
        k += np.sum(np.where(diff > 0.05, 1, 0))
    if k == 0:
        k = 1
    return k, (k / (len(x) * len(x)))


def get_freq_features_ecg(x: np.ndarray):
    mean = np.mean(x)
    yf = np.array(scipy.fft.fft(x - mean))
    xf = scipy.fft.fftfreq(len(x), mean)[0 : len(x) // 2]
    psd = (2 / len(yf)) * np.abs(yf)[0 : len(x) // 2]
    fmean = np.mean(xf)
    fstd = np.std(xf)
    sumpsd = np.sum(psd)
    return fmean, fstd, sumpsd


def get_data_ecg(x: np.ndarray, fs: int) -> np.ndarray:
    try:
        working, _ = heartpy.process(x, fs)
    except Exception:
        scaled = heartpy.scale_data(x)
        working, _ = heartpy.process(scaled, fs)

    peak = working["peaklist"]
    if len(peak) < 3:
        raise ValueError("insufficient_peak_count")

    periods = np.array([(peak[i + 1] - peak[i]) / fs for i in range(0, len(peak) - 1)])
    frequency = 1 / periods
    meanfreq = np.mean(frequency)
    stdfreq = np.std(frequency)
    hrv = np.array([(peak[i] - peak[i - 1]) / fs for i in range(1, len(peak))])
    _, _, tinn_value, hrv_index = best_tinn(hrv)
    num50, p50 = compare_nn50(hrv)
    mean_hrv = np.mean(hrv)
    std_hrv = np.std(hrv)
    rms_hrv = np.sqrt(np.mean(hrv**2))
    fmean, fstd, sumpsd = get_freq_features_ecg(hrv)
    return np.array(
        [
            meanfreq,
            stdfreq,
            tinn_value,
            hrv_index,
            num50,
            p50,
            mean_hrv,
            std_hrv,
            rms_hrv,
            fmean,
            fstd,
            sumpsd,
        ],
        dtype=float,
    )


def slice_per_label(labelseq: np.ndarray, fs_label: int, time_window: int, step: int):
    pts_window = time_window * fs_label
    taken_indices = []
    conv = np.array([1 for _ in range(0, pts_window)])
    for i in range(0, len(labelseq) - pts_window, fs_label * step):
        extr = labelseq[i : i + pts_window]
        res = np.sum(np.multiply(extr, conv))
        l = labelseq[i]
        if l in [1, 2, 3, 4]:
            if l * pts_window == res:
                taken_indices.append((i, l))
    return taken_indices


def map_index(idx: int, source_fs: int, target_fs: int) -> int:
    return int(round((idx * target_fs) / source_fs))


def resample_ecg(ecg: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    if source_fs == target_fs:
        return ecg
    return signal.resample_poly(ecg, target_fs, source_fs)


def process_subject(
    pkl_path: Path,
    source_fs: int,
    target_fs: int,
    window_seconds: int,
    step_seconds: int,
) -> Tuple[List[List[float]], List[int], Dict[str, int]]:
    df = pd.read_pickle(str(pkl_path))
    label = np.array(df["label"])
    ecg = np.array(df["signal"]["chest"]["ECG"][:, 0], dtype=float)
    ecg_target = resample_ecg(ecg, source_fs, target_fs)

    taken_indices = slice_per_label(label, source_fs, window_seconds, step_seconds)
    neutral_idx = np.where(label == 1)[0]
    if len(neutral_idx) < 2:
        raise ValueError("neutral_phase_not_found")

    neutral_start = map_index(int(neutral_idx[0]), source_fs, target_fs)
    neutral_end = map_index(int(neutral_idx[-1]), source_fs, target_fs)
    ecg_neutral = ecg_target[neutral_start:neutral_end]
    features_neutral = get_data_ecg(ecg_neutral, target_fs)

    pts_window = window_seconds * target_fs
    features: List[List[float]] = []
    labels: List[int] = []
    discard_inf = 0
    discard_error = 0

    for idx_src, lbl in taken_indices:
        idx_tgt = map_index(idx_src, source_fs, target_fs)
        end_tgt = idx_tgt + pts_window
        if end_tgt > len(ecg_target):
            discard_error += 1
            continue
        window = ecg_target[idx_tgt:end_tgt]
        try:
            result = np.divide(get_data_ecg(window, target_fs), features_neutral)
            if np.isinf(result).any() or np.isnan(result).any():
                discard_inf += 1
            else:
                features.append(result.tolist())
                labels.append(int(lbl))
        except Exception:
            discard_error += 1

    stats_dict = {
        "windows_total": len(taken_indices),
        "windows_ok": len(labels),
        "discard_inf_or_nan": discard_inf,
        "discard_error": discard_error,
    }
    return features, labels, stats_dict


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(
        [p for p in args.wesad_dir.iterdir() if p.is_dir() and p.name.startswith("S")]
    )

    summary = {
        "source_fs": args.source_fs,
        "target_fs": args.target_fs,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "subjects_total": len(subject_dirs),
        "subjects_processed": 0,
        "subjects_failed": 0,
        "total_windows_ok": 0,
        "total_discard_inf_or_nan": 0,
        "total_discard_error": 0,
        "subjects": {},
    }

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        pkl_path = subject_dir / f"{subject}.pkl"
        if not pkl_path.exists():
            summary["subjects_failed"] += 1
            summary["subjects"][subject] = {"error": "pkl_not_found"}
            continue
        try:
            feats, lbls, st = process_subject(
                pkl_path,
                args.source_fs,
                args.target_fs,
                args.window_seconds,
                args.step_seconds,
            )
            data = {"id": [], "label": lbls, "features": feats}
            out_path = output_dir / f"WESADECG_{subject}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f)

            summary["subjects_processed"] += 1
            summary["total_windows_ok"] += st["windows_ok"]
            summary["total_discard_inf_or_nan"] += st["discard_inf_or_nan"]
            summary["total_discard_error"] += st["discard_error"]
            summary["subjects"][subject] = st

            print(
                f"[{subject}] ok={st['windows_ok']} total={st['windows_total']} "
                f"discard_inf={st['discard_inf_or_nan']} discard_err={st['discard_error']}"
            )
        except Exception as ex:
            summary["subjects_failed"] += 1
            summary["subjects"][subject] = {"error": str(ex)}
            print(f"[{subject}] failed: {ex}")

    summary_path = output_dir / "wesad_100hz_preprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Completed WESAD 100Hz preprocessing")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
