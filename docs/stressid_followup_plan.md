# StressID Follow-up Plan

## Goal
Position StressID as an external validation challenge under domain shift, then support the manuscript with one lightweight and reproducible adaptation experiment instead of switching the paper toward dataset-specific models.

## Fixed Protocol
- StressID role: external validation, not a failed side experiment.
- Adaptation split: `subject-aware`.
- Labeled target budget: `10%` of StressID subjects.
- Seed: `42`.
- Main comparison: `zero-shot baseline`, `CORAL`, `final-layer fine-tune`.
- Main success metric: held-out StressID `F1`, with `accuracy`, `precision`, `recall`, and confusion structure reported alongside it.

## Execution Steps
1. Prepare subject-aware StressID assets from `Data_Processed_StressID/STRESSIDECG_*.json`.
2. Reuse the existing WESAD DNN checkpoints from `Models_CrossVal`.
3. Evaluate the held-out StressID split with:
   - zero-shot baseline
   - CORAL using target statistics estimated from the adaptation subset
   - final-layer fine-tuning on the adaptation subset with early stopping
4. Save per-fold, average, comparison, and manuscript-ready summary artifacts under `Results_StressID_Compare`.

## Evidence Boundaries
- Keep the existing zero-shot and CORAL results as valid evidence.
- Do not use the previous threshold-search result as the main adaptation claim, because the selected threshold collapsed to `0.0`, which degenerated into predicting every sample as stress.
- Do not expand into domain-adversarial training unless the lightweight adaptation result clearly justifies another round.

## Manuscript Tasks
### Results
- Add `External Validation on StressID`.
- Show the in-domain vs out-of-domain gap directly.
- Add one short table covering `WESAD in-domain`, `StressID zero-shot`, `StressID + CORAL`, and `StressID + final-layer adaptation`.

### Discussion
- State that cross-dataset physiological stress transfer remains difficult under domain shift.
- Explain that minimal adaptation may help but does not remove the gap.
- Tie the remaining gap back to SmartStress as a closed-loop, personalization-aware system rather than a pure classifier benchmark.

## Acceptance Criteria
- Subject split manifest exists and shows disjoint adaptation/eval subjects.
- Adaptation/eval JSON files are readable and non-empty.
- The adaptation runner can complete a single-fold smoke run without modifying source checkpoints.
- The final report explicitly answers whether final-layer fine-tuning improves held-out StressID `F1` over the zero-shot baseline.
