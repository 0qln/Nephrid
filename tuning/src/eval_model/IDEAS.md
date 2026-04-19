# Ideas

## Training

### Mcts

- [ ] selection: try to use a selector that weighs the actualy game results higher than the value estimations

### Phases

- larger batchsizes in earlier phases to encourage discovery of more basic rules

#### Phase 0 (MateIn1s)

overfitting the value output to a win seems to resolve itself in later phases, because atleast we can easily farm a good policy.
(but also we could try setting the value loss to 0 so we don't train toward that target)

#### Phase 1

the speedup gained by taking early cuts out if the selfplays are not good seems to really pay off:
```
01:03:16 INFO train  - Resuming training from tuning/out/eval_model/main/phase0/artifacts/model_e-0010_i-00351
01:10:15 INFO train  - [Train - Epoch 1 - Iteration 0.0] Loss 5.59750 (Value: 0.97289, Policy: 4.62461, Solved: 90.23%)
01:12:18 INFO train  - [Train - Epoch 1 - Iteration 1.0] Loss 5.10891 (Value: 1.01191, Policy: 4.09701, Solved: 96.09%)
01:13:56 INFO train  - [Train - Epoch 1 - Iteration 2.0] Loss 4.50106 (Value: 0.91411, Policy: 3.58694, Solved: 96.48%)
01:16:32 INFO train  - [Train - Epoch 1 - Iteration 3.0] Loss 4.42153 (Value: 0.83477, Policy: 3.58677, Solved: 94.14%)
01:18:17 INFO train  - [Train - Epoch 1 - Iteration 4.0] Loss 4.42350 (Value: 0.97697, Policy: 3.44653, Solved: 96.48%)
01:19:35 INFO train  - [Train - Epoch 1 - Iteration 5.0] Loss 4.27622 (Value: 0.94685, Policy: 3.32937, Solved: 97.27%)
01:21:44 INFO train  - [Train - Epoch 1 - Iteration 6.0] Loss 4.35825 (Value: 0.95243, Policy: 3.40582, Solved: 95.31%)
```

### Selfplay

- even in early phases, where we learn mateInX, we allow for deeper than X plies in the selfplay to allow to punish bad moves

- [ ] UnfinishedGameHandling::Stochastic, where we count the ((1\*wins + -1\*losses) / count(wins/losses/draws)) in the subtree

## Model

### Board inputs

- [ ] attack plane
- [ ] checker plane
- [ ] pinner plane
- [ ] pinned plane
- ...etc...
