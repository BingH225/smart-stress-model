# StressID Manuscript Outline

## Results: External Validation on StressID
StressID should be introduced as a deliberate external validation dataset used to test cross-dataset robustness under domain shift. The section should first report the strong WESAD in-domain result, then show the performance drop on StressID zero-shot transfer, and finally compare the lightweight adaptation baselines.

Recommended table:

| Setting | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| WESAD in-domain | TBD | TBD | TBD | TBD |
| StressID zero-shot | TBD | TBD | TBD | TBD |
| StressID + CORAL | TBD | TBD | TBD | TBD |
| StressID + final-layer adaptation | TBD | TBD | TBD | TBD |

Suggested framing sentence:

> Strong in-domain performance does not translate under domain shift, which motivates adaptation-aware and personalization-aware stress support designs.

## Discussion
The StressID result should not be framed as a failure of SmartStress. Instead, it should be used to argue that cross-dataset physiological stress detection remains difficult even when the base classifier performs strongly in-domain. Any improvement from minimal adaptation supports the idea that small target-domain updates can help, while any remaining gap reinforces the need for closed-loop orchestration, baseline normalization, and personalization in deployment.

If final-layer adaptation improves transfer:

> Lightweight target-domain adaptation improved held-out transfer performance, but a substantial residual gap remained, indicating that domain mismatch is only partially addressable through minimal post hoc tuning.

If final-layer adaptation does not improve transfer:

> Simple adaptation was insufficient to close the gap on held-out StressID, further motivating personalization-aware and closed-loop support mechanisms beyond a standalone stress classifier.

## Notes for the Final Revision
- Keep StressID in Results and Discussion, not only as a limitation footnote.
- Avoid shifting the paper toward dataset-specific benchmarking.
- Use the external validation section to strengthen the system-level contribution of SmartStress.
