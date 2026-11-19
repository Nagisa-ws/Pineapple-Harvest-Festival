# Mission 3 — 4 fruits per episode

## Files
| File | What it is |
|---|---|
| `Mission3_train_dashboard.html` | Interactive training dashboard (open in any browser, works offline) |
| `Mission3_eval_dashboard.html` | Interactive evaluation dashboard |
| `Mission3_train_overview.png` | Static image of the training view (for slides / reports) |
| `Mission3_eval_overview.png` | Static image of the evaluation view |
| `data/train_log.csv`, `data/eval_log.csv` | The original logs used |

## Training log — key numbers
- **Episodes:** 81 over ≈ 197k timesteps, 1 parallel environment.
- **Success rate:** 58.0% (47 episodes).
- **First 40 → last 40 episodes:** 55% → 60% success. The rolling rate (16 episodes) stays between about 38% and 75%, with no clear trend.
- **Median episode return:** +373.8 (first 40) → +421.0 (last 40).
- **Most common failure:** row finished incomplete (33 of 34 failures), followed by timeout (1).
- **Time per harvested fruit:** median 34.7 s in the last 40 episodes (motion + computing).
- **Reward components:** median leaf penalty -47.3 in failed episodes vs -22.3 in successful ones; median progress +165.0 vs +208.5.

## Evaluation log — key numbers
- **Reference controller alone (before training):** 50.0% of the episodes, 87.5% of the fruits, mean return -242.0, 51.1 s per fruit.
- **Checkpoints:** 50, every 4k timesteps, with 2 evaluation episodes each.
- **Mean success rate:** 48.0% (fruit rate 87.0%). From 45% (first 10 checkpoints) to 45% (last 10).
- **Best success rate:** 100.0%, reached at 1 checkpoint.
- **Mean return:** best 339.7 at 48k; -100.5 → -204.2 (first vs last 10 checkpoints).
- **SAC vs reference:** Best SAC checkpoint (48k steps): 100% of the fruits, 100% of the episodes. Reference controller alone: 88% / 50%. SAC is better than the reference.
- **Per evaluation episode:** not available. At 49 of 50 checkpoints the log names fewer failures than the success rate implies (an episode that only skipped a fruit is logged without a failure), so the episode that was not completed cannot be identified.

## Notes / assumptions
- Episode order in the evaluation heat map comes from the position of each entry in the `failures` column (separated by `|`; an empty entry means success, unless the success rate of that row says otherwise). The evaluation episodes are fixed scenes (same seeds at every checkpoint).
- A failure written as `cause@phase` is counted under its cause; the phase is shown in the tooltip of the dashboard.
- The training was started more than once (the step counter jumps back in the log). Only the last run is shown; left out: 2 rows of `train_log.csv` and 2 rows of `eval_log.csv`.
- Only 2 evaluation episodes per checkpoint: one episode changes the success rate by 50 percentage points, so small differences mean little.
