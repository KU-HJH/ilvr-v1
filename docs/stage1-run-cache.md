# Stage 1 experiment cache

The optional `--stage1-cache run` path populates records during the original
first epoch and replays them after the coordinator completes both node-local
copies. It is off by default. GPU acceptance is recorded separately; CPU tests
alone do not establish numerical equivalence or production speedup.

Use the existing `run_multinode_stage1_training.sh` coordinator with two nodes,
four ranks, microbatch 1 and `--dataloader-num-workers 4`. Keep the experiment's
approved losses, schedule, sampler and attention settings. Cache initialization
rejects unsupported geometry, a mutable visual encoder and dynamic RoPE.

| Teacher | Stored data | Computed during replay |
|---|---|---|
| Fixed | Static sample preprocessing and final chosen/enabled reverse targets | Student and losses |
| Updating native EMA | Static sample preprocessing and frozen helper vision output | Current teacher language/query/selection, student and losses |
| Frozen V5 reference | All queried final hidden rows | Original reference head with unchanged chunks and autocast |

Helper pixel arrays are transient. No full vocabulary logits or dynamic
entropy/token selection are stored. Records retain their native tensor bits and
dtype. Private mmap and the OS page cache share backing pages without a whole
cache dictionary per rank or guaranteed RAM residency.

The logical `data/cache/<run-id>` link points to the owned
`/root/workspace/ILVR-stage1-cache/<run-id>` directory on the verified root SSD
filesystem on each node. Existing dataset links and unrelated `data/cache`
contents are preserved. Both nodes need space for a complete copy: roughly
55 GiB for the current full fixed V5 dataset and 103–153 GiB for EMA, depending
on the actual vision dtype. No dtype conversion is used to meet the estimate.

At the first epoch boundary, both node leaders publish integrity manifests.
The existing coordinator streams missing records through SSH/container transport
in bounded chunks, checks complete local coverage and releases the next iterator.
There is no tensor payload collective. Missing coverage, conflicting duplicate
payloads or missing replay records fail the run; they never trigger a training
fallback. Dynamic occurrence and batch ownership are rebuilt from the sampler.

Resume retains the saved cache/check policy and objective. Checkpoints contain
the additive `stage1_run_cache.json`; fixed native teachers also retain their
initial state in `stage1_cache_initial_teacher.pt`. V5 reuses its existing
`stage1_v5_reference.pt`. Model-bearing states stay under `checkpoints` and are
excluded from prepared inference artifacts. A later-epoch cache loss is rebuilt
in a separate four-rank preparation phase from the original fixed teacher or
unchanged EMA vision, with training RNG restored. Old checkpoints without this
identity cannot silently enable caching.

`--stage1-cache-completion train` removes only the owned cache after validated
normal training completion. With `train-eval`, retain it until successful
evaluation through `python -m src.multinode evaluate` with the explicit
`--stage1-cache-training-run-id <training-run-id>`. Failed/interrupted runs and
planned checkpoint stops retain caches. Workers/readers close before cleanup;
results, logs, checkpoints and other runs are preserved.

Fresh production uses only essential checks, also when caching is off. The
isolated verifier owns expensive equality/hashing/statistic/gradient audits and
deterministic backward. Historical resumes retain their saved check policy.

The explicit `--stage1-runtime-profile native-ema-0180` uses the preserved
`stage1-visual-history-only-20261019/venv/ilvr-venv` on both nodes for native
standard EMA validation. It is not a V5 or Stage 2 profile. Both node preflight
and worker environments select that same runtime; the default still requires
DeepSpeed 0.18.7. No package reinstall or attention substitution is performed.
