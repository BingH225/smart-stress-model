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

## 2026-04-22
- Reviewed full end-to-end WESAD DNN workflow from `Data_preprocessing.py`, `Model-training.py`, `Model_testing.py`, and cross-validation scripts/results.
- Reviewed full StressID transfer workflow from `preprocess_stressid.py`, `Model_testing.py`, `evaluate_mapped_json.py`, and `Results_StressID_Compare/*` outputs.
- Extracted and compared official run metrics/logs for WESAD in-domain testing and StressID external testing.
- Performed additional quantitative validation in `conda env torch` to verify cross-dataset confusion structure, score distribution, and threshold sensitivity.
- Performed additional dataset-level statistics check (label distribution, Cohen's d, feature distribution shift) to support cause-by-cause reverse reasoning.
- Removed temporary analysis artifacts created during this investigation: `_tmp_*.py` and `Results_StressID_Compare/_tmp_*.json`.
- Updated `.gitignore` to explicitly ignore `.codex/WORKLOG.md` per project instruction.

## 2026-04-23
- Started implementation of the stepwise StressID repair-and-gate workflow requested by the user.
- Created working branch `codex/stressid-stepwise-fix` for gated stepwise fixes.
- Prepared baseline commit scope (`.gitignore` + `.codex/WORKLOG.md`) before adding orchestration and stepwise repair code.
- Added `remote_bundle_gpu_20260410/evaluate_stressid_stepwise.py` to support optional CORAL alignment, robust normalization, threshold/temperature calibration, and quality filtering under a single remote evaluation entrypoint.
- Added `remote_bundle_gpu_20260410/run_gpu_container.template.pbs` as a parameterized PBS template preserving the existing Apptainer execution logic.
- Added `scripts/prepare_stepwise_assets.py` to generate stepwise runtime assets (`STRESSID_TEST`, `STRESSID_CALIB`, and WESAD source-domain stats variants).
- Added `scripts/Invoke-StepwiseStressIDFix.ps1` to orchestrate bundle assembly, SSH upload, qsub execution, polling, result fetch, gating decision, and git branch workflow per step.
- Ran local smoke checks: Python syntax checks, asset preparation run, and local evaluation smoke run with the new stepwise evaluator.
- Updated the orchestration script to persist step decision logs on the main branch for both accepted and rejected steps.
- Fixed null-safe handling for empty git tag/branch query output in `Invoke-StepwiseStressIDFix.ps1`.
- Fixed model directory wildcard copy logic in `Invoke-StepwiseStressIDFix.ps1` (`Copy-Item` path construction).
- Fixed `scp` remote path formatting in `Invoke-StepwiseStressIDFix.ps1` to avoid path canonicalization errors on Aspire.
- Replaced remote bundle upload from `scp -r` to `tar | ssh` streaming in `Invoke-StepwiseStressIDFix.ps1` for Aspire compatibility.
- Reworked remote bundle upload to `tar.gz + scp + remote extract` in `Invoke-StepwiseStressIDFix.ps1` after stream mode failed on Aspire.
- Updated PBS template rendering to ASCII encoding (no BOM) and hardened qsub submission failure handling in `Invoke-StepwiseStressIDFix.ps1`.
- Fixed `Wait-RemoteJob` state polling command in `Invoke-StepwiseStressIDFix.ps1` (switched from fragile awk quoting to sed-based parsing).
- Step 1 (domain_coral) remote run 20260423_144056: F1 0.354945, Acc 0.541772, decision=REJECT, job=13855843.pbs101 (rule: F1 must improve).
- Step 2 (semantic_negative_tighten) remote run 20260423_152611: F1 0.409855, Acc 0.513380, decision=REJECT, job=13856640.pbs101
- Fixed commit message metric formatting in scripts/Invoke-StepwiseStressIDFix.ps1 so step accept/reject commits include numeric F1/Acc.
- Step 3 (robust_baseline_norm) remote run 20260423_153017: F1 0.507153, Acc 0.444729, decision=REJECT, job=13856710.pbs101
- Step 4 (calibration_threshold) remote run 20260423_155947: F1 0.587444, Acc 0.415873, decision=REJECT, job=13857346.pbs101
- Step 5 (sampling_protocol_100hz) remote run 20260423_160305: F1 0.587444, Acc 0.415876, decision=REJECT, job=13857508.pbs101
- Step 6 (feature_quality_filter) remote run 20260423_160735: F1 0.587444, Acc 0.415875, decision=REJECT, job=13858006.pbs101

## 2026-05-27
- Added `docs/stressid_followup_plan.md` to turn the advisor feedback into a concrete external-validation and lightweight-adaptation execution plan.
- Added `docs/stressid_manuscript_outline.md` with manuscript-ready Results/Discussion framing for StressID as external validation.
- Extended `scripts/prepare_stepwise_assets.py` with a `subject` split mode, `--subject-fraction`, and `subject_split_manifest.json` support for subject-aware StressID adaptation/eval asset generation.
- Added `scripts/run_stressid_adaptation.py` to compare held-out StressID zero-shot baseline, CORAL, and final-layer fine-tuning on top of the existing WESAD DNN checkpoints.
- Ran `scripts/prepare_stepwise_assets.py` in subject-aware mode with `--subject-fraction 0.1 --seed 42`, generating `Results_StressID_Compare/subject_adapt_runtime/{asset_summary.json,subject_split_manifest.json,data/STRESSID_ADAPT.json,data/STRESSID_EVAL.json}`.
- Ran a 1-fold smoke validation of `scripts/run_stressid_adaptation.py` in `conda env torch` with `Results_CrossVal/cross_val_results.json`; verified baseline, CORAL, and final-layer fine-tuning outputs plus JSON/Markdown artifact generation.
- Confirmed that `Results_CrossVal/cross_val_results.json` only contains 5 folds and switched the formal adaptation run to `Results_CrossVal_Full/cross_val_results.json` with `Models_CrossVal_Full`.
- Ran the full 15-fold subject-aware adaptation comparison in `conda env torch`; saved formal outputs to `Results_StressID_Compare/stressid_subject_adaptation_{report,comparison,summary}.{json,json,md}`.
- Recorded full 15-fold held-out StressID averages: zero-shot `Acc 0.4903 / Prec 0.4299 / Rec 0.6306 / F1 0.5106`; CORAL `Acc 0.4977 / Prec 0.4180 / Rec 0.4782 / F1 0.4460`; final-layer fine-tune `Acc 0.4904 / Prec 0.4299 / Rec 0.6300 / F1 0.5104`.
- Extended `scripts/run_stressid_adaptation.py` with `--pos-weight`, `--neg-weight`, and `--weight-decay` options so final-layer fine-tuning can be tuned without changing the adaptation protocol.
- Ran a 10-config hyperparameter sweep over final-layer fine-tuning (`lr`, `epochs`, `patience`, `batch-size`, internal seed) using the fixed subject-aware 10% StressID split; no configuration exceeded the zero-shot baseline F1.
- Ran an 8-config weighted-loss sweep over final-layer fine-tuning (`pos_weight`, `neg_weight`, `weight_decay` plus training hyperparameters); best F1 remained slightly below baseline (`delta_f1` about `-3.0e-05`), confirming no meaningful uplift from final-layer-only tuning under the current protocol.
- Extended `scripts/run_stressid_adaptation.py` with `--finetune-input-space {raw,coral}` and `--finetune-scope {final_layer,last_block}` so the requested direction-2 / direction-1 adaptation variants can reuse the same runner and reporting format.
- Ran a 6-config sweep for direction 2 (`CORAL + final-layer fine-tune`) on the fixed 10% subject-aware StressID split; all configurations converged near the CORAL-only behavior and remained well below baseline (`best F1 delta` about `-0.0647`).
- Generated a new 20% subject-aware StressID split with `scripts/prepare_stepwise_assets.py --subject-fraction 0.2`, saved under `Results_StressID_Compare/subject_adapt_runtime_20pct`.
- Ran a 5-config sweep for direction 3 (20% target-domain budget) on the new split; raw final-layer fine-tune nearly matched baseline but still stayed below it (`best F1 delta` about `-8.3e-05`), while CORAL + final-layer remained substantially worse (`F1` about `0.4070` vs baseline `0.5020`).
- Ran a 6-config sweep for direction 1 (`last-block` fine-tune) on the 20% subject-aware split; this consistently underperformed the 20% baseline by about `0.0093` to `0.0123` F1, despite small accuracy increases.
- Wrote aggregate sweep summaries to `Results_StressID_Compare/{direction2_coral_final_layer_10pct_summary.json,direction3_20pct_summary.json,direction1_last_block_20pct_summary.json}` for quick best-config lookup and manuscript-side evidence review.
- Extended `scripts/prepare_stepwise_assets.py` with stricter subject sampling controls: `--subject-selection {random,stratified}` plus `--subject-bins`, and recorded per-subject stress ratios / bin selections in `subject_split_manifest.json`.
- Extended `scripts/run_stressid_adaptation.py` with adaptation-objective controls: `--train-objective {bce,weighted_bce,focal,soft_f1_bce}`, `--threshold-mode {fixed_0.5,val_f1}`, threshold range/grid options, and per-fold selected-threshold reporting.
- Generated a new stratified 20% subject-aware runtime split at `Results_StressID_Compare/subject_adapt_runtime_20pct_stratified_seed42` and verified the new manifest captures balanced low/mid/high stress-ratio subject coverage.
- Ran a 5-fold proxy search on the original random 20% split for step-1 objective redesign (`BCE+val-threshold`, `weighted BCE`, `focal`, `soft-F1+BCE`); all threshold-aware variants improved over the zero-shot baseline, with F1 rising from `0.4984` to about `0.5407-0.5410`.
- Ran full 15-fold confirmation on the original random 20% split for `BCE + validation-threshold` and `focal + validation-threshold`; the best full result was `focal + validation-threshold`, improving held-out StressID F1 from `0.5020` to `0.5355` (`delta_f1 = +0.0335`) while reducing accuracy from `0.4946` to `0.4790`.
- Ran stricter step-2 multi-seed evaluation with stratified 20% subject sampling for seeds `42-46`, each with full 15-fold `focal + validation-threshold` adaptation; all 5 seeds improved over their respective baselines.
- Aggregated step-1/step-2 summaries to `Results_StressID_Compare/{step12_stratified_multiseed_summary.json,step12_stratified_multiseed_summary.md}`; stratified multi-seed averages were baseline `F1 0.5029 ± 0.0105` vs adaptation `F1 0.5325 ± 0.0111`, mean `delta_f1 = +0.0296 ± 0.0025`, with mean selected threshold `0.1296`.

## 2026-07-10 21:12:12 +08:00
- Reviewed the latest advisor feedback and the current `SmartStress` manuscript state to decide the next project priority.
- Confirmed the advisor wants work to shift away from additional StressID adaptation experiments and toward manuscript-level strengthening: novelty statement, system-level differentiation, Discussion framing, figure quality, and final abstract/conclusion alignment.
- Confirmed the manuscript already contains the needed StressID external-validation narrative and multi-split lightweight-adaptation evidence, so the next phase should focus on paper polishing rather than new model-method expansion.

## 2026-07-13 16:11:10 +08:00
- Cross-checked the manuscript's StressID claims against `step12_stratified_multiseed_summary`: baseline F1 `0.5029 +/- 0.0105`, final-layer adaptation F1 `0.5325 +/- 0.0111`, five wins across seeds 42-46, and validation-selected mean threshold `0.1296`; the reported manuscript metrics and trade-off interpretation are consistent with these artifacts.
- Audited SHAP model provenance and confirmed `Model_interpretability.py` currently imports the standard `ClassifierECG` from `Model_testing.py`, while the manuscript attributes the SHAP analysis to the attention-augmented PhysioSense model. Recorded this as a pre-submission evidence-consistency issue. No experiment code or result artifact was modified.

## 2026-07-15 15:03:55 +08:00
- Scoped possible SmartStress capstone extensions against the original course prototype and current journal manuscript. Prioritized an uncertainty-aware, personalization-adaptive closed-loop system combining calibration, drift/OOD detection, label-efficient subject adaptation, and abstention-aware orchestration; also identified RAG safety/provenance, multimodal missing-sensor robustness, edge deployment, and adaptive intervention learning as alternative directions. No source code, experiment artifact, or manuscript content was modified.
