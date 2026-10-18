# Stage1-v5 GPU Utilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (selected by the user) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize four existing Stage1-v5 operator paths while preserving discrete interventions and the approved numerical contract, and measure the actual local effects.

**Architecture:** Preserve the existing clean/reference/packed/backward flow. Freeze independent old sources, record local costs, vectorize masks/dropout, retain selector boundary arithmetic while batching attention, then bind execution identity and verify integration. Do not launch training or change sample/rank allocation.

**Tech Stack:** Repository Python venv, PyTorch, vendored Qwen2.5-VL/Transformers, pytest, existing direct operator benchmark.

---

## Authority, constraints and commands

The user replied `계속` to the exact final commitment approval question,
including the disclosed untested attribution/boundary/memory uncertainty.
Approved artifact: `docs/superpowers/specs/2026-10-18-stage1-v5-gpu-utilization-commitment-v1.md`,
SHA256 `843832ae9b09c10f28dea3dea60ceaa8d845f77f103f98259d4f26304cbe659a`.
No further design approval is needed within this scope. Use the retained SDD
method continuously; root writes control records only. Agents do not invoke
advisory hooks or spawn agents. Each task must pass independent spec review,
then quality review before the next implementer. Fixes use implementer agents.

Workspace `/root/.codex/worktrees/d3c8/ILVR` is already isolated and host-owned.
Source baseline `65643f08cb6e61fbbb200de825d57086f5c7a127`; all product sources
still match it at approval. Preserve existing links, artifacts and other trees.
Use repository `./venv/ilvr-venv/bin/python`; no installs or system Python.
CPU integration tests are allowed. CUDA work is operator-only, without model
training/optimizer steps, after a current idle-device check. Training smoke
would require the two-node coordinator and is not part of this approval.

For each focused pytest command below prepend:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -m pytest -q
```

Use a new `--basetemp=checkpoints/stage1-v5-gpu-utilization-20261018-tests/<task>-<phase>`.
Redirect complete output to `logs/stage1-v5-gpu-utilization-20261018/<task>-<phase>.log`.
Read complete output; record causal RED and GREEN commands, exits and results.
Results/measurements go under `results/stage1-v5-gpu-utilization-20261018/`.
Do not rerun the already-green broad baseline. No post-failure tolerance changes.

## File responsibilities

- `tests/stage1/v5/frozen_gpu_utilization/`: immutable four source files from the baseline revision, plus literal hash expectations in the test helper.
- `tests/stage1/v5/gpu_utilization_fixtures.py`: independent loading, bounded fake operator inputs/counters; no full model loading or expected values derived from candidate logic.
- `tests/stage1/v5/test_gpu_utilization.py`: causal performance witnesses, exact old/new masks/dropout/selection and tolerant attention comparison, boundary and benchmark contract checks.
- Existing `scripts/benchmark_stage1_v5_attention_selection.py`: extend the direct script to all four operators, truthful equality/error/timing/memory metadata. No new framework.
- `transformers/src/transformers/models/qwen2_5_vl/stage1_v5_replay.py`: replay bias vectorization, keeping attached prefixes and branch coordinates.
- `src/stage1/v5_execution.py`: gather mask rows without token Python assembly; keep RNG/lifecycle semantics.
- `src/stage1/v5_selection.py`: stable tensor ranking, exact integer coordinates, original Python cumulative decisions.
- `transformers/src/transformers/models/qwen2_5_vl/stage1_v5_trace.py`: bounded query chunks and exported internal policy constants.
- `src/stage1/v5_trainer.py`, `scripts/verify_stage1_v5.py`, existing resume/protocol tests: actual source/profile identity, admission and early incompatible-resume rejection.
- Existing throughput/replay/selection/execution/trainer tests: retain or narrowly update assertions explicitly affected by the approved numerical/transfer contract; do not weaken independent oracles.

## Task 1: Freeze baseline and extend the existing operator measurement

**Files:** Create frozen sources/helper/test file above; modify the existing benchmark and its old benchmark assertions only as needed.

- [ ] **1. Write causal benchmark contract and frozen-hash tests first.** The new measurement must return before/after data for replay mask and dropout as well as existing attention/selection, include raw paired times, geometry, scope, source identity and actual memory. A minimal RED assertion is:

```python
def test_benchmark_contains_all_four_independent_operators(tmp_path, monkeypatch):
    import json
    from scripts.benchmark_stage1_v5_attention_selection import main
    output = tmp_path / 'operators.json'
    monkeypatch.setattr('sys.argv', ['benchmark', '--device', 'cpu', '--warmup', '0',
                                  '--repetitions', '1', '--output', str(output)])
    main()
    report = json.loads(output.read_text())
    for name in ('attention', 'selection', 'replay_mask', 'dropout'):
        assert report[name + '_before']['milliseconds']
        assert report[name + '_after']['milliseconds']
    assert report['mask_exact_equal']
    assert report['dropout_exact_equal']
```

- [ ] **2. Run just the new failing benchmark test**, using the standard CPU environment and unique task1-red basetemp/log. Expect a missing replay/dropout record, not an import or fixture error.
- [ ] **3. Freeze and implement the minimal measurement extension.** Copy bytes with `git show 65643f08cb6e61fbbb200de825d57086f5c7a127:<path>` for each of the four baseline modules, checking the commitment's literal hashes. Load them under their existing packages with unique frozen names. Keep the historical frozen_throughput files unchanged. Build deterministic bounded fixtures and a no-model replay decoder that captures the actual bias; label fixed stub overhead in measurement. Use real private-generator dropout execution contexts. Do not rewrite production operator algorithms in this task.

The paired timing loop must use the same prebuilt inputs and alternate order:

```python
before_times, after_times = [], []
for repetition in range(repetitions):
    calls = ((before, before_times), (after, after_times))
    if repetition % 2:
        calls = tuple(reversed(calls))
    for function, samples in calls:
        synchronize()
        start = time.perf_counter_ns()
        function()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
```

Retain independent per-implementation peak-memory checks, synchronize CUDA,
and label CPU tracemalloc as Python-only. Keep exact equality booleans truthful.
Add max attention errors/within-tolerance fields without falsely reporting exact
equality. Reuse existing CLI with bounded geometry options only if necessary;
do not load production weights or save model-bearing artifacts.

- [ ] **4. Run task1 GREEN and existing benchmark contract tests.** Run `tests/stage1/v5/test_gpu_utilization.py tests/stage1/v5/test_throughput_attention.py`. Expected green; old tests can gain additional report fields but must still compare identical candidate inputs.
- [ ] **5. Record source65643f0 costs before any product rewrite.** Run the existing script with warmup3/repetitions20 on CPU and idle CUDA device0; save distinct `baseline-cpu.json` and `baseline-cuda.json` plus full logs. Check `nvidia-smi` first. Report current GPU name/version (the fresh observation shows H100 NVL, not a presumed B200). This is operator measurement, not training.
- [ ] **6. Self-review and commit.** Record byte hashes and actual command results. Do not change root workflow records. Return DONE/CONCERNS with commit and log paths; root obtains sequential spec then quality review.

## Task 2: Exact replay-mask and dropout tensor assembly

**Files:** Modify replay/execution; add test_gpu_utilization exact and dispatched-operation witnesses; retain existing execution/replay tests.

- [ ] **1. Add tests that compare actual bias and raw masks to the pinned modules**, with padding, V-only/Text-only/both, inactive blocks, repeated/permuted multi-source coordinates, variable lengths, empty input and unchanged global CPU/CUDA RNG. Add an actual dispatched-operation count witness that fails the old per-query bias assignments and token-length-dependent select/stack assembly. Do not time pytest to decide correctness. For example, evaluate raw masks on both implementations in their own F/checkpoint/backward scopes, then:

```python
assert torch.equal(actual_mask, frozen_mask)
assert torch.equal(actual_bias, frozen_bias)
assert torch.equal(torch.get_rng_state(), rng_state_before)
```

Reuse the existing real native replay oracle for gradients; do not derive the
expected mask from a newly shared helper. Establish any new fixture before
the causal operation-count assertion so RED reflects the missing optimization.
- [ ] **2. Run the causal new tests on the old code and read their RED output.** Expect excess per-token/select or per-query assignment operations; exact baseline comparisons should pass.
- [ ] **3. Replace replay query loops with broadcast fill and indexed exclusions.** The core replacement uses one floating output bias, row/prefix and suffix visibility, and no second dense floating workspace:

```python
row_ends = torch.tensor([b.end for _, b in rows], device=x.device)
prefix_keys = torch.arange(prefix_length, device=x.device)
prefix_valid = (prefix_keys[None] < row_ends[:, None])
prefix_valid &= trace.padding[samples[:, 0], :prefix_length].bool()
suffix_keys = torch.arange(length, device=x.device)
causal = suffix_keys[None] <= suffix_keys[:, None]
bias = x.new_full((len(rows), 1, length, prefix_length + length), float('-inf'))
bias[..., :prefix_length].masked_fill_(prefix_valid[:, None, None], 0.)
bias[..., prefix_length:].masked_fill_(causal[None, None], 0.)
bias.masked_fill_(~valid[:, None, :, None], float('-inf'))
bias[:, 0, :, 0].masked_fill_(~valid, 0.)
```

Apply V body exclusion with row/prefix-sized boolean broadcasts (body cannot
include key0). Flatten Text exclusion triples from existing host block tuples
and assign -infinity once; preserve `target-1-end`, empty exclusions and branch
order. Vectorize row-valid lengths rather than launching work per query.
Keep prefix concatenation/gradients and dropout callback ownership unchanged.

Replace dropout's per-token assembly with per-source tensor indexing. Retain
validation, source seed and whole original random shape. A compact unique row
read is allowed; no global generator or persistent all-site cache. The core is:

```python
source_indices = sample_indices.to(device=device)
source_positions = token_positions.to(device=device)
result = torch.empty((source_indices.numel(), hidden_size), device=device,
                     dtype=torch.float32)
for row in source_indices.unique().tolist():
    slot, source_length = self.spec.rows[row]
    generator = torch.Generator(device=device)
    generator.manual_seed(stage1_sample_seed(self.spec.base_seed,
        self.spec.optimizer_global_step, slot, site))
    mask = torch.rand((source_length, hidden_size), device=device,
                      dtype=torch.float32, generator=generator).ge(p).float()
    indices = (source_indices == row).nonzero(as_tuple=True)[0]
    result.index_copy_(0, indices, mask.index_select(0, source_positions[indices]))
return result
```

Keep debug bounds validation and native division/packed multiplication exactly.
Do not assert that remaining unique/nonzero calls eliminate every CUDA sync.
- [ ] **4. Run focused GREEN:** `tests/stage1/v5/test_gpu_utilization.py tests/stage1/v5/test_execution.py tests/stage1/v5/test_replay.py`. Include full/early-stop checkpoint tests and the existing FP32 loss/gradient tolerance `1e-5/1e-4`. Use one idle CUDA operator-only exact-mask comparison after CPU checks; no model training on CUDA.
- [ ] **5. Self-review and commit.** Return source diff, causal logs and exact-mask/RNG evidence. Root obtains spec then quality review; resolve findings with this implementer before task3.

## Task 3: Stable selector ranking and bounded attention observation

**Files:** Modify selection/trace; extend independent joint fixtures/test_gpu_utilization; narrowly update approved throughput floating/transfer assertions and benchmark acceptance.

- [ ] **1. Write selector tests first.** Assert exact ordered original keys against the frozen selector for ties, cap0, empty candidates, low-mass and `nextafter` cumulative thresholds, independent CPU/CUDA attention rows and interleaved samples. Add a causal transferred-element/Python-sort witness: fewer transferred candidate weights than the original full candidate rows when cap<1, while integer identities remain exact and Python accumulation is unchanged.
- [ ] **2. Run selector RED then implement stable prefix ranking.** Compute original-device row mass before any sorting/copy. For each aligned weights vector, retain original FP32 values and produce:

```python
order = torch.argsort(weights, descending=True, stable=True)[:cap]
prefix_weights = weights[order]
```

Bulk-transfer order as int64 separately from FP32 prefix weights/masses/rhos.
Reconstruct original integer positions from each block's sorted unique tuple.
Keep the existing Python sequential decision verbatim over the transferred
ranked prefix, including low-mass fraction/cap, `total += weight` and
`total >= host_rhos[key] * mass`. Do not use a GPU prefix sum or FP32 index
packing. Preserve validation and mixed attention/logit device callers.
Run `test_selection.py`, new selector tests and existing selector throughput
tests GREEN before proceeding to attention.

- [ ] **3. Write attention operation-count and joint comparison tests first.** Keep the frozen source; set ATOL1e-7/RTOL1e-5 before running candidate results. Compare each candidate row and exact selected tuple after both old/new traces using identical projections/logits; selector-only equality does not suffice. Include >32 queries, short final chunks, padding, GQA, nontrivial multimodal RoPE, ties and threshold neighbors. Keep old exact equality fields truthful. RED must show per-query matmul count rather than import errors.
- [ ] **4. Implement minimal bounded query computation and run the joint probe before broad integration.** Define internal constants `V5_ATTENTION_QUERY_CHUNK_MAX=32` and `V5_ATTENTION_WORKSPACE_BYTES=33554432`; effective chunk is at least1 and at most32 with target `2*heads*chunk*keys*4` bytes. Within each sample/layer:

```python
positions = torch.tensor([key[1] for key, _ in queries_chunk],
                         device=q.device, dtype=torch.long)
scores = q[row].index_select(1, positions).float() @ sample_k
visible = (coordinates[None] <= positions[:, None]) & sample_padding[row][None]
weights = (scores * attn.scaling).masked_fill(~visible[None],
             float('-inf')).softmax(-1).mean(0)
for index, (key, candidates) in enumerate(queries_chunk):
    values = weights[index, candidate_indices[candidates]] / len(decoder.layers)
    trace.attention_rows[key] = trace.attention_rows.get(key, 0) + values
```

Retain detach/autocast-off/RoPE/GQA setup, per-sample FP32 K reuse and sequential
layer accumulation. This bounds transient score/probability tensors, not total
VRAM. Do not truncate causal denominator. Any exact selected-key mismatch
must be reported immediately with the input and error; stop this activation
and contact root. Never loosen tolerance or silently adopt a partial scope.

- [ ] **5. Run CPU GREEN and one idle CUDA joint operator comparison**, including boundary cases, before accepting attention. Run `test_gpu_utilization.py test_throughput_attention.py test_selection.py test_execution.py test_replay.py test_trainer.py`. CPU-only model integration uses current tiny fixtures and existing FP32 gradient tolerances. Benchmark report must include both attention closeness and joint exact selected keys; no production-BF16 equivalence claim.
- [ ] **6. Self-review and commit.** Report exact discrete results, maximum floating error, policy bounds, relevant operator counts and logs. Root obtains sequential spec and quality review.

## Task 4: Actual execution identity, integration and final measurements

**Files:** Modify v5_trainer, existing verifier, literal resume/protocol tests, benchmark integration as required; no coefficient/formula/schema changes.

- [ ] **1. Add causal runtime tests before code.** Actual `_build_v5_runtime_contract` must bind four added modules and an operator profile. Extend the existing literal runtime-source assertions (do not import the production mapping as the expected value). Test missing/changed added bindings and profile, different per-rank contracts, old historical load-only contracts and the actual `train` dispatch rejecting incompatible resume before parent training/optimizer work. RED must demonstrate the currently missing bindings/profile.
- [ ] **2. Bind actual runtime module bytes**, not decorated-wrapper source files. Import the four modules and hash `Path(module.__file__).read_bytes()`; `inspect.getsourcefile` on contextmanager/no_grad wrappers can accidentally hash contextlib. Add explicit source keys `v5_execution`, `v5_selection`, `v5_trace`, `v5_replay`. Save:

```python
execution['operator_profile'] = {
    'id': 'stage1_v5_vectorized_ops_v1',
    'attention_query_chunk_max': V5_ATTENTION_QUERY_CHUNK_MAX,
    'attention_score_probability_target_bytes': V5_ATTENTION_WORKSPACE_BYTES,
}
```

Extend verifier `RUNTIME_SOURCE_PATHS` to the actual four files and check the
profile in admission. Preserve all eight older bindings. Keep historical
parsing separate from strict current runtime admission. Existing effective
contract equality must reject changed source/profile before parent `train`;
give guidance to use the original reviewed code. No hash normalization or
checkpoint migration; schema2 and objective remain unchanged.
- [ ] **3. Run focused GREEN:** `tests/stage1/v5/test_resume.py tests/stage1/v5/test_protocol.py tests/stage1/v5/test_trainer.py tests/stage1/v5/test_history_integration.py tests/stage1/v5/test_gpu_utilization.py`. Read complete logs and verify actual dispatch witnesses.
- [ ] **4. Run final operator measurements once after focused correctness.** Check idle GPUs; run paired CPU and CUDA0 benchmark with warmup3/repetitions20 at default and bounded long/multi-block shapes. Record raw medians/peak memory/transfers/equality/error and current source/profile/GPU/package identity. Use distinct `candidate-{cpu,cuda}-{default,long}.json`. Label fixture stub overhead and scope; report any operator regression, no end-to-end training claim.
- [ ] **5. Self-review and commit.** Return measured comparisons, reviewed source identity and limitations. Root obtains task spec then quality review, and finally a separate whole-change reviewer. Only after findings close, root invokes verification-before-completion and finishing, runs the project suite once and additional distinct claim checks.

## Controller completion and plan self-review

- [ ] Record each task base/head, implementer/reviewer IDs, causal commands/results and review dispositions in the existing workflow ledger; preserve unresolved findings.
- [ ] Dispatch an independent final reviewer over the entire baseline-to-final diff after all task reviews. Fix requested changes through agents and re-review affected claims.
- [ ] Read finishing and verification skills, announce finishing, run one shared project full suite on the final reviewed revision using a fresh checkpoint basetemp; read complete output. Do not repeat the same broad suite back-to-back. Additional performance/artifact/identity claims need their distinct existing witnesses.
- [ ] Report actual operator results and limitations. Keep source/checkpoints and host-managed worktree; human owns final merge/PR/keep decision. Write ntfy summary before the final response.

Inline plan self-review: all four approved paths, source/resume consumers,
independent exact/numerical comparisons, staged joint boundary gate, measured
wall time/memory, strict no-training scope and selected review/finish method
map to tasks above. Constants and source keys match across tasks3/4. Frozen
historical sources stay untouched. No placeholder code or generic new recovery,
cache or launch framework is proposed. Existing method selection is reused;
do not ask the execution-choice question again.
