# WORKLOG

## 2026-04-21
- Created WORKLOG.md as required by project instruction.
- Investigated whether StressID can validate the first-agent WESAD DNN model.
- Inspected data schema in `D:\NUS\BMI5101\StressID` and training pipeline in `D:\NUS\BMI5101\smart-stress-model`.
- Key checks done: model input dimensionality, feature extraction assumptions (ECG-based 12 features), label mapping logic, and StressID task/label coverage.
- Added `preprocess_stressid.py` to convert StressID ECG tasks into WESAD-compatible JSON (`features`, `label`) with 12-feature order alignment, baseline normalization, and label mapping `0->1`, `1->2`.
- Installed `heartpy` into conda environment `torch` to satisfy ECG feature extraction dependency for StressID preprocessing and original scripts.
- Updated `preprocess_stressid.py` with a HeartPy fallback path (`scale_data`) when raw StressID ECG windows fail direct peak extraction.
- Added per-subject progress logging in `preprocess_stressid.py` for long-running full-dataset conversion observability.
- Ran full StressID preprocessing in `conda env torch` with `fs=100`, `window=20s`, `step=1s`, output to `Data_Processed_StressID`.
- Generated merged test JSON `Data_Processed_StressID/WESADECG_S17.json` with 306,221 samples, 12-dim features, label map `{1: no-stress, 2: stress}`.
- Executed original `Model_testing.py` on GPU (`CUDA_VISIBLE_DEVICES=0`) for WESAD baseline test (`DIR_DATA_TEST=Data_Processed`), saved logs to `Results_StressID_Compare/wesad_run/run.log`.
- Executed original `Model_testing.py` on GPU (`CUDA_VISIBLE_DEVICES=0`) for StressID external test (`DIR_DATA_TEST=Data_Processed_StressID`), saved logs to `Results_StressID_Compare/stressid_run/run.log`.
- Recorded final metrics for comparison: WESAD test `Acc 0.9645 / Prec 0.8787 / Rec 0.9904 / F1 0.9312`; StressID test `Acc 0.4807 / Prec 0.4221 / Rec 0.6733 / F1 0.5189`.
- Exported structured comparison report to `Results_StressID_Compare/comparison_summary.json` (metrics, deltas, preprocessing summary).
- Added `preprocess_wesad_100hz.py` as a new standalone preprocessing pipeline to downsample WESAD ECG from 700Hz to 100Hz, extract the same 12 features, and output JSON files for a separate 100Hz experiment without modifying original scripts.
- Ran `preprocess_wesad_100hz.py` in `conda env torch` and generated `Data_Processed_100Hz/WESADECG_*.json` for subjects S2-S17.
- Preprocessing summary saved to `Data_Processed_100Hz/wesad_100hz_preprocess_summary.json` with per-subject window statistics and discard counts.
- Executed original `Model_testing.py` with `DIR_DATA` and `DIR_DATA_TEST` both pointing to `Data_Processed_100Hz` to run a new WESAD 100Hz retraining/testing experiment on GPU.
- Saved 100Hz run log to `Results_100Hz_Experiment/wesad_100hz_run/run.log` with final metrics `Acc 0.9406 / Prec 0.8034 / Rec 1.0000 / F1 0.8910`.
- Created comparison report `Results_100Hz_Experiment/comparison_wesad_700_vs_100.json` against the previous 700Hz WESAD run.
- Added `Cross_validation_ablation_100hz.py` as a separate 100Hz LOSO cross-validation pipeline (copied from the original script, with isolated data/model/result directories) to avoid modifying or overwriting the 700Hz experiment.
- Created dedicated 100Hz cross-validation output folders: `Results_CrossVal_100Hz` and `Models_CrossVal_100Hz`.
- Executed `Cross_validation_ablation_100hz.py` (15-fold LOSO, DNN+Attention, 50 epochs) on GPU with `DIR_DATA_100HZ=Data_Processed_100Hz`; generated `Results_CrossVal_100Hz/cross_val_results.json` and `Results_CrossVal_100Hz/ablation_report.md`.
- Evaluated StressID with 100Hz cross-validation DNN checkpoints using `evaluate_mapped_json.py`; saved report to `Results_StressID_Compare/stressid_eval_with_100hz_cv.json`.
- Computed comparison summaries against previous 700Hz cross-validation and StressID external evaluation metrics.
- Saved external-eval delta report `Results_StressID_Compare/comparison_stressid_700_vs_100_cv.json` (100Hz-vs-700Hz checkpoints on StressID).
- Saved WESAD LOSO DNN aggregate comparison report `Results_CrossVal_100Hz/comparison_wesad_cv_700_vs_100.json` (100Hz-vs-700Hz cross-validation means).
