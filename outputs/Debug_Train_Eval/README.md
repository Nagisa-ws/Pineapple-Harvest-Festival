# Pineapple harvesting robot — RL training and evaluation visualizations

Made by `python scripts/plot_training.py`. Each mission has its own folder. Every folder contains:
- two interactive HTML dashboards (training and evaluation), which open in any browser and work offline;
- two static PNG images for slides and reports;
- a `SUMMARY.md` with the key numbers;
- the original logs in `data/`.

| Folder | Task | Training | Evaluation |
|---|---|---|---|
| `Mission1/` | single-fruit harvest (1 target per episode) | 769 episodes · ≈ 300k steps · 81.3% success | 30 checkpoints × 10 episodes · 88.0% mean success (best 90.0%) · reference controller 90.0% |
| `Mission2/` | 2 fruits per episode | 366 episodes · ≈ 299k steps · 55.2% success | 30 checkpoints × 6 episodes · 70.6% mean success (best 83.3%) · reference controller 66.7% |
| `Mission3/` | 4 fruits per episode | 81 episodes · ≈ 197k steps · 58.0% success | 50 checkpoints × 2 episodes · 48.0% mean success (best 100.0%) · reference controller 50.0% |

The missions use different tasks, numbers of evaluation episodes and training lengths, so their numbers are not directly comparable.
