# Mission 1 — single-fruit harvest (1 target per episode)

## Files
| File | What it is |
|---|---|
| `Mission1_train_dashboard.html` | Interactive training dashboard (open in any browser, works offline) |
| `Mission1_eval_dashboard.html` | Interactive evaluation dashboard |
| `Mission1_train_overview.png` | Static image of the training view (for slides / reports) |
| `Mission1_eval_overview.png` | Static image of the evaluation view |
| `data/train_log.csv`, `data/eval_log.csv` | The original logs used |

## Training log — key numbers
- **Episodes:** 769 over ≈ 300k timesteps, 4 parallel environments.
- **Success rate:** 81.3% (625 episodes).
- **First 100 → last 100 episodes:** 84% → 79% success. The rolling rate (100 episodes) stays between about 75% and 88%, with no clear trend.
- **Median episode return:** +141.1 (first 100) → +137.2 (last 100).
- **Most common failure:** leaf uprooted (58 of 144 failures), followed by fruit dropped (21) and grasp miss (14).
- **Time per harvested fruit:** median 26.2 s in the last 100 episodes (motion + computing).
- **Reward components:** median leaf penalty -18.7 in failed episodes vs -0.7 in successful ones; median progress +6.9 vs +43.3.

## Evaluation log — key numbers
- **Reference controller alone (before training):** 90.0% of the episodes, 90.0% of the fruits, mean return 111.8, 29.0 s per fruit.
- **Checkpoints:** 30, every 10k timesteps, with 10 evaluation episodes each.
- **Mean success rate:** 88.0% (fruit rate 88.0%). From 88% (first 10 checkpoints) to 88% (last 10).
- **Best success rate:** 90.0%, reached at 24 checkpoints.
- **Mean return:** best 112.9 at 180k; 101.1 → 100.5 (first vs last 10 checkpoints).
- **SAC vs reference:** Best SAC checkpoint (10k steps): 90% of the fruits, 90% of the episodes. Reference controller alone: 90% / 90%. SAC is the same as the reference.
- **Per evaluation episode:**
  - Episode 1 always succeeds.
  - Episode 2 always succeeds.
  - Episode 3 always succeeds.
  - Episode 4 always succeeds.
  - Episode 5 never succeeds; fails with leaf uprooted (30).
  - Episode 6 succeeds at 25 of 30 checkpoints; fails with collision (2) or leaf uprooted (1).
  - Episode 7 succeeds at 29 of 30 checkpoints; fails with fruit hit (1).
  - Episode 8 always succeeds.
  - Episode 9 always succeeds.
  - Episode 10 always succeeds.

## Notes / assumptions
- Episode order in the evaluation heat map comes from the position of each entry in the `failures` column (separated by `|`; an empty entry means success, unless the success rate of that row says otherwise). The evaluation episodes are fixed scenes (same seeds at every checkpoint).
- A failure written as `cause@phase` is counted under its cause; the phase is shown in the tooltip of the dashboard.
