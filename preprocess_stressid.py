import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import heartpy
import numpy as np
import scipy
import scipy.stats as stats


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert StressID ECG into WESAD-compatible JSON (features, label)."
    )
    parser.add_argument(
        "--stressid-root",
        type=Path,
        default=Path("D:/NUS/BMI5101/StressID/StressID Dataset"),
        help="StressID dataset root folder.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional labels.csv path. Defaults to <stressid-root>/labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("D:/NUS/BMI5101/smart-stress-model/Data_Processed_StressID"),
        help="Output folder for subject JSONs and merged test JSON.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=100,
        help="StressID ECG sampling rate (Hz).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=20,
        help="Window size (seconds).",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=1,
        help="Sliding step (seconds).",
    )
    parser.add_argument(
        "--merged-filename",
        type=str,
        default="WESADECG_S17.json",
        help="Merged JSON filename for direct use with original Model_testing.py.",
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


def read_ecg_txt(path: Path) -> np.ndarray:
    values: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "ECG" not in reader.fieldnames:
            raise ValueError(f"ECG column missing in {path}")
        for row in reader:
            raw = row.get("ECG", "")
            if raw is None or raw == "":
                continue
            values.append(float(raw))
    if not values:
        raise ValueError(f"no ECG values read from {path}")
    return np.array(values, dtype=float)


def sliding_windows(signal: np.ndarray, fs: int, window_seconds: int, step_seconds: int):
    pts_window = window_seconds * fs
    step = step_seconds * fs
    if len(signal) < pts_window:
        return
    for start in range(0, len(signal) - pts_window + 1, step):
        yield signal[start : start + pts_window]


def robust_baseline_feature(
    baseline_signal: np.ndarray, fs: int, window_seconds: int, step_seconds: int
) -> np.ndarray:
    try:
        return get_data_ecg(baseline_signal, fs)
    except Exception:
        feats = []
        for win in sliding_windows(baseline_signal, fs, window_seconds, step_seconds):
            try:
                f = get_data_ecg(win, fs)
                if np.isfinite(f).all():
                    feats.append(f)
            except Exception:
                continue
        if not feats:
            raise ValueError("baseline_feature_extraction_failed")
        return np.median(np.array(feats, dtype=float), axis=0)


def load_labels(labels_csv: Path) -> List[Dict[str, str]]:
    with labels_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def map_binary_stress_to_label(binary_stress: str) -> int:
    if int(binary_stress) == 1:
        return 2
    return 1


def main() -> None:
    args = parse_args()
    labels_csv = args.labels_csv or (args.stressid_root / "labels.csv")
    phys_root = args.stressid_root / "Physiological"
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_labels(labels_csv)
    subject_rows: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        task_key = row["subject/task"]
        subject = task_key.split("_", 1)[0]
        subject_rows.setdefault(subject, []).append(row)

    merged_features: List[List[float]] = []
    merged_labels: List[int] = []
    summary = {
        "sampling_rate_hz": args.fs,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "subjects_total_in_labels": len(subject_rows),
        "subjects_written": 0,
        "subjects_skipped_no_baseline": 0,
        "task_files_missing": 0,
        "windows_success": 0,
        "windows_discard_nonfinite": 0,
        "windows_discard_error": 0,
    }

    for subject, srows in sorted(subject_rows.items()):
        subj_dir = phys_root / subject
        baseline_file = subj_dir / f"{subject}_Baseline.txt"
        if not baseline_file.exists():
            summary["subjects_skipped_no_baseline"] += 1
            continue

        try:
            baseline_signal = read_ecg_txt(baseline_file)
            baseline_feature = robust_baseline_feature(
                baseline_signal, args.fs, args.window_seconds, args.step_seconds
            )
        except Exception:
            summary["subjects_skipped_no_baseline"] += 1
            continue

        subject_features: List[List[float]] = []
        subject_labels: List[int] = []

        for row in srows:
            task_key = row["subject/task"]
            binary_stress = row["binary-stress"]
            label_mapped = map_binary_stress_to_label(binary_stress)
            signal_file = subj_dir / f"{task_key}.txt"
            if not signal_file.exists():
                summary["task_files_missing"] += 1
                continue

            try:
                signal = read_ecg_txt(signal_file)
            except Exception:
                summary["task_files_missing"] += 1
                continue

            for win in sliding_windows(
                signal, args.fs, args.window_seconds, args.step_seconds
            ):
                try:
                    feat = get_data_ecg(win, args.fs)
                    ratio = np.divide(feat, baseline_feature)
                    if np.isfinite(ratio).all():
                        subject_features.append(ratio.astype(float).tolist())
                        subject_labels.append(label_mapped)
                        summary["windows_success"] += 1
                    else:
                        summary["windows_discard_nonfinite"] += 1
                except Exception:
                    summary["windows_discard_error"] += 1

        if not subject_features:
            continue

        out = {"features": subject_features, "label": subject_labels}
        out_path = output_dir / f"STRESSIDECG_{subject}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f)

        merged_features.extend(subject_features)
        merged_labels.extend(subject_labels)
        summary["subjects_written"] += 1
        print(
            f"[subject={subject}] samples={len(subject_labels)} "
            f"cumulative={len(merged_labels)}"
        )

    merged = {"features": merged_features, "label": merged_labels}
    merged_path = output_dir / args.merged_filename
    with merged_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f)

    with (output_dir / "stressid_preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Completed StressID preprocessing")
    print(f"Feature order: {FEATURE_NAMES}")
    print(f"Merged JSON: {merged_path}")
    print(f"Total samples: {len(merged_labels)}")
    print(f"Label dist (1=no-stress,2=stress): {dict((k, merged_labels.count(k)) for k in sorted(set(merged_labels)))}")
    print(f"Summary JSON: {output_dir / 'stressid_preprocess_summary.json'}")


if __name__ == "__main__":
    main()
