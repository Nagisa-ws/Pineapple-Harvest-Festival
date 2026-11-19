# Mission 2 — 2 fruits per episode

## Files
| File | What it is |
|---|---|
| `Mission2_train_dashboard.html` | Interactive training dashboard (open in any browser, works offline) |
| `Mission2_eval_dashboard.html` | Interactive evaluation dashboard |
| `Mission2_train_overview.png` | Static image of the training view (for slides / reports) |
| `Mission2_eval_overview.png` | Static image of the evaluation view |
| `data/train_log.csv`, `data/eval_log.csv` | The original logs used |

## Training log — key numbers
- **Episodes:** 366 over ≈ 299k timesteps, 4 parallel environments.
- **Success rate:** 55.2% (202 episodes).
- **First 100 → last 100 episodes:** 55% → 56% success. The rolling rate (73 episodes) stays between about 45% and 62%, with no clear trend.
- **Median episode return:** +149.4 (first 100) → +172.2 (last 100).
- **Most common failure:** leaf uprooted (61 of 164 failures), followed by grasp miss (23) and fruit dropped (16).
- **Time per harvested fruit:** median 27.1 s in the last 100 episodes (motion + computing).
- **Reward components:** median leaf penalty -29.2 in failed episodes vs -9.5 in successful ones; median progress +12.4 vs +85.3.

## Evaluation log — key numbers
- **Reference controller alone (before training):** 66.7% of the episodes, 75.0% of the fruits, mean return 125.9, 32.0 s per fruit.
- **Checkpoints:** 30, every 10k timesteps, with 6 evaluation episodes each.
- **Mean success rate:** 70.6% (fruit rate 76.9%). From 75% (first 10 checkpoints) to 70% (last 10).
- **Best success rate:** 83.3%, reached at 7 checkpoints.
- **Mean return:** best 198.9 at 240k; 162.1 → 121.8 (first vs last 10 checkpoints).
- **SAC vs reference:** Best SAC checkpoint (30k steps): 83% of the fruits, 83% of the episodes. Reference controller alone: 75% / 67%. SAC is better than the reference.
- **Per evaluation episode:**
  - Episode 1 succeeds at 7 of 30 checkpoints; fails with leaf uprooted (18) or grasp miss (3).
  - Episode 2 always succeeds.
  - Episode 3 always succeeds.
  - Episode 4 never succeeds; fails with fruit dropped (30).
  - Episode 5 always succeeds.
  - Episode 6 always succeeds.

## Notes / assumptions
- Episode order in the evaluation heat map comes from the position of each entry in the `failures` column (separated by `|`; an empty entry means success, unless the success rate of that row says otherwise). The evaluation episodes are fixed scenes (same seeds at every checkpoint).
- A failure written as `cause@phase` is counted under its cause; the phase is shown in the tooltip of the dashboard.
