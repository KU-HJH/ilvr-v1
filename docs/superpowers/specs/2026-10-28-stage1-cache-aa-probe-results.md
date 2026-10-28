# Stage 1 cache premise probe — observed result

## Scope and workflow

The user's “ㅇㅋ. 계속” authorized the previously proposed throwaway A/A
probe, up to four optimizer updates across fixed V5 and native standard EMA.
It did not approve the cache design or production implementation. Brainstorming
remains active; no implementation plan or cache feature has been started.

Native observation hooks were implemented at
`50e0eeb6122b80729149579595be9c27c6b303db` and independently reviewed by
`/root/cache_aa_probe_review`. The reviewer found the existing matched-standard
stop parser gate blocked the probe. That failure was reproduced, narrowly fixed
for explicit fresh probe mode, and checked against both allowed profiles and
ordinary/invalid stop cases. CPU checks also covered signed-zero-sensitive
hashes, unchanged CPU RNG and actual AdamW per-group hooks. Python syntax,
launcher syntax and diff checks passed. This was bounded throwaway probe work,
not a completed production implementation lifecycle.

The reviewer separately approved a proposed narrow native-EMA DS0.18.0
preflight exception. That patch was never applied. The native EMA pair was not
executed after V5 falsified the whole-step bitwise-repeatability premise.
Two of the four authorized updates were executed. No environment was installed
or modified; repository venv symlinks were not changed.

## Direct runtime evidence

Runs `stage1-cache-aa-v5-a-20261028` and
`stage1-cache-aa-v5-b-20261028` each completed exactly one optimizer update,
four accumulated microbatches per rank, through the existing coordinator.
Both nodes and all four global ranks produced completed witnesses. Prepared
DataLoaders used four workers per rank. Both coordinators completed checkpoint
collection and emitted `initial-train-witness.json`.

Both arms retained max_steps2160, epochs10, global batch16, LR5e-5, warmup10,
seed42, BF16, DoRA, ZeRO1, DeepSpeed0.18.7, mixed SDPA/FA2 and the existing
V5 CE0.1/alignment0.1/DIL1/entropy0.1/mass0.05/TextKL0.01 profile. Selection
was clean-entropy top20% with rejected vocabulary top50 and attention-mask V-off.
The first warmup update had LR0. Digest observations add CPU transfers and
synchronization; their runtime is not a speed measurement.

Rank-aligned comparisons of all eight JSON records found:

- Initial trainable student/teacher state, input tensors and recorded RNG match.
- All 16 sample losses, selected targets/indices, observed teacher visual outputs,
  reference head inputs/logits and before/after-microbatch RNG match bitwise.
- The actual optimizer-input gradient digest differs on all four ranks.
- Adam first and second moments differ on all four ranks.
- Student/teacher parameters and FP32 master parameters match after the update,
  but LR0 means this does not establish nonzero-update equivalence.

CPU comparison of saved optimizer shards confirms real non-padding differences:

| Rank | First-moment max absolute difference | First-moment relative L2 difference |
|---|---:|---:|
| 0 | 3.0721258e-4 | 1.2774% |
| 1 | 2.4102279e-5 | 0.8510% |
| 2 | 1.3708952e-5 | 0.6227% |
| 3 | 1.4347839e-5 | 0.3772% |

Each shard has 15,227,904 elements and zero padding. About 8.52 million elements
per shard differ in each moment. These are moment statistics, not a direct
measurement of original gradient magnitudes.

The first observed disagreement is between equal forward losses and the
optimizer's input gradients. Local backward, distributed reduction and clipping
were not separately instrumented. No individual kernel or communication
operation has been established as the cause.

## Consequence and remaining decision

The frozen proposal's strict whole-step bitwise requirement has not cleared its
first-nail check: the observed uncached V5 path itself is not bitwise repeatable.
This does not show that caching changes the forward payload or final accuracy.
The sampled forward payloads agree, but cross-rank replay, later EMA versions,
cache/live A/B, nonzero-LR updates and all-dataset equivalence remain untested.

Do not silently loosen acceptance or change attention/determinism settings.
A clarification is pending between (a) exact cache payloads with unchanged
native computation and RNG, acknowledging existing backward variability, and
(b) investigating native backward repeatability before committing to caching.
Neither choice is recorded as approved yet. The frozen proposal remains intact.

## Evidence and cleanup

- Authorization/review/results: `logs/stage1-cache-design-20261028/`.
- Rank-aligned comparison: `results/stage1-cache-aa-comparison-20261028/v5.json`.
- Moment magnitudes: `results/stage1-cache-aa-comparison-20261028/v5-moment-differences.json`.
- Rank observations: `results/stage1-cache-aa-v5-{a,b}-20261028/aa-rank-{0,1,2,3}.json`.
- Both-node logs: existing run logs plus `logs/stage1-cache-design-20261028/remote-v5-logs/`.
- Model-bearing probe artifacts remain under the existing `checkpoints/` tree.

Temporary probe source was reverted at `6924d75`; comparison against pre-probe
`4e43de7` shows no retained changes under `src/` or the Stage 1 launcher.
No cache data was created. This is a premise-probe readout, not branch completion
or merge handoff; the development-branch finishing gate does not apply.
