# Tabel hasil terukur (dibuat otomatis oleh scripts/make_results.py)

## Training mission1

init: `reference_mission3`

```
timesteps,success_rate,fruit_rate,mean_return,failures
10000,0.333,0.500,-292.31,|timeout|fruit_dropped|plan_no_feasible_route|timeout|
20000,0.667,0.667,-156.41,||fruit_dropped|plan_no_feasible_route||
30000,0.167,0.167,-281.22,collision|timeout|fruit_dropped|plan_no_feasible_route|timeout|
```

## Training mission2

init: `project_checkpoint:mission1`

```
timesteps,success_rate,fruit_rate,mean_return,failures
5000,0.250,0.375,-3.74,leaf_uprooted|fruit_hit||fruit_dropped
10000,0.250,0.375,-8.29,leaf_uprooted|fruit_hit||fruit_dropped
15000,0.250,0.375,-6.73,leaf_uprooted|fruit_hit||fruit_dropped
20000,0.250,0.375,-4.98,leaf_uprooted|fruit_hit||fruit_dropped
```

## Training mission3

init: `project_checkpoint:mission2`

```
timesteps,success_rate,fruit_rate,mean_return,failures
1500,0.000,0.500,24.94,collision
3000,0.000,0.500,25.25,collision
```
