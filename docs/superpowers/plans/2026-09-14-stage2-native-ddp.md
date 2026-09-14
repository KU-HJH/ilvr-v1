# Stage 2 Native DDP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native DDP execution path to the existing Stage2 coordinator, with faithful checkpoint/resume and observed execution evidence.

**Architecture:** Keep the current coordinator, worker, ownership and transport. Add an explicit backend branch and strict DDP-only V2 artifact/evidence contracts while old ZeRO V1 remains unchanged. Rejoin the approved objective's comparison-example delivery after integration.

**Tech Stack:** Repo Python3.11 venv, installed PyTorch2.7.1, Accelerate1.10.1, patched Transformers Trainer, existing SSH/tmux/rsync coordinator.

**Spec:** `docs/superpowers/specs/2026-09-14-stage2-native-ddp-design.md`, commitment v1/section12 approved 2026-09-14. The unchanged objective is specified by `docs/superpowers/specs/2026-09-14-stage2-entropy-vision-design.md` and `documents/ILVR_Codex_Superpowers_Implementation_Prompt.md`.

## Global Constraints

- Add Stage2-only native DDP to the existing fixed two-node/four-rank coordinator to enable the approved three comparison examples.
- Keep Stage1, legacy ZeRO, objective math/F/M/reference and training baseline unchanged. No new coordinator, generic backend framework, historical matrix expansion, evaluation migration or deployment automation.
- Stage2 `train-stage2` and standalone preflight accept `--backend {deepspeed,ddp}`, default `deepspeed`.
- Native launch retains 2 machines, 4 processes, remote machine rank0/local rank1, master `10.1.102.41`, interface `bond1.102`, same explicit port and repository venv.
- DDP rejects an explicit `--zero-stage`, a nonempty `DEEPSPEED_CONFIG`, and conflicting enabled Accelerate DeepSpeed/FSDP/plugin environment settings.
- Keep legacy artifact manifest/identity V1 emitters and accepted field sets unchanged for Stage1/Stage2 ZeRO.
- V2 requires `execution_backend="ddp"`, `objective_enabled: bool`, `stage="stage2"`, `zero_stage=null`, `deepspeed_config_digest=null`, `model_state_layout="trainer"` plus the existing identity fields. Never encode DDP as ZeRO0.
- `objective_enabled` is exactly the resolved `enable_stage2_objective` boolean, not `config.active` and not inferred from reference presence.
- Native rank0 owns common model/optimizer/scheduler/Trainer/sidecar state; all ranks own their RNG files. Original local manifests may be RNG2/3-only, canonical manifests may not.
- Use existing no-overwrite collection/redistribution and exact-owned process control. Cross-backend resume is unsupported and rejected before launch.
- Run CPU tests locally; do not launch GPU jobs, merge, push, or overwrite/synchronize canonical user changes without the separate required user decision.
- Work only in `/root/workspace/ILVR/.worktrees/stage2-entropy-vision`; reuse its linked venv/data/checkpoints. Never edit canonical user's overlapping changes.
- Python/tests use `CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python` or pytest. Do not install/download dependencies. Test model fixtures belong under checkpoints; logs under logs/stage2-ddp-dev; results under results.
- Every production behavior has an observed causal RED first; a missing module/signature can establish API RED, but does not cover later normalization, ownership or distributed behavior. Add the corresponding behavioral RED before those implementations too.
- Subagents do not spawn subagents or invoke paperthin. One implementer at a time, task-scoped spec and quality review before the next task. Preserve exact reports and unrun limitations.

## Files and task boundaries

`src/multinode.py` owns CLI/LaunchPlan/node-info/training dispatch and expected identity;
`run_stage2_training.sh` owns actual Accelerate argv and local environment guards;
`src/multinode_runtime.py` owns existing command builders, transport and final witness;
`src/multinode_artifacts.py` owns observed checkpoint contents and manifest validation;
`src/stage2/evidence.py` owns observed rank records and collective publication;
`src/main.py` connects the actual Stage2 runtime to that publisher.

Do not split these existing large modules into a new framework. New test-only
helpers may live in `tests/multinode/native_ddp_helpers.py`; never place test
runner/cleanup APIs in production classes. Task4 includes the original objective
plan's pending Task5 delivery (one wrapper, one document, one final branch review).

### Task 1: Explicit backend, real worker argv, and preflight inputs

**Status:** Complete; implementation4c02422, no-write regression testsd2e26cb;
task review and scoped fix review approved. Covering suite213passed/1known
baseline platform failure; amended dry-run tests2passed after causal temporary
write mutation2failed. No remote/GPU/training verification was run.

**Files:** Modify `src/multinode.py`, `src/multinode_runtime.py`, `run_stage2_training.sh`; create `tests/multinode/test_native_ddp_launch.py`; extend `tests/stage2/test_launcher.py` only for executable worker behavior.

**Interfaces:**
- Extend `LaunchPlan` with defaulted `backend: Literal['deepspeed','ddp']='deepspeed'` and `stage2_objective_config: Stage2ObjectiveConfig | None=None`; the config is the existing frozen class, not a mutable dict. Widen existing zero/config/digest fields to optional only under the DDP invariant.
- Extend `build_launch_plan(*, stage, tuning_method, zero_stage=None, run_id, main_process_port, attempt_id='initial', backend='deepspeed', stage2_objective_config=None)->LaunchPlan`. Old callers with explicit ZeRO behave identically; default DeepSpeed still rejects omitted ZeRO.
- Add `resolve_stage2_objective_config(arguments: Sequence[str], *, supervise_latent_boundaries: bool)->Stage2ObjectiveConfig` in `src/multinode_runtime.py`. Reuse `utils_deepseed.parse_args(['--stage','stage2',*arguments])` and existing `Stage2ObjectiveConfig.from_args/validate`; do not duplicate nine CLI defaults or load a model. Lazy imports keep Stage1/preflight cold paths unchanged.
- Extend `validated_stage2_environment(..., deepspeed_config: str|None, backend='deepspeed', ...)`; DDP returns `ILVR_STAGE2_BACKEND='ddp'` and no nonempty DS config. Keep old environment/argv unchanged when omitted/default DeepSpeed.
- Forward existing objective options through the allowlist: enable/no-enable boolean and lambda_dyn/lambda_vis/reference_model_path/gamma_H/m_V/entropy_eps/pre_window/post_window values. Freeze resolved config into the Stage2 plan so evidence checking does not reread mutable environment later.
- Environment guard scope is `ACCELERATE_USE_DEEPSPEED`, `ACCELERATE_USE_FSDP`, `ACCELERATE_USE_MEGATRON_LM`, and `ACCELERATE_USE_CPU`: reject truthy values using installed Accelerate str_to_bool spellings (case-insensitive y/yes/t/true/on/1; false n/no/f/false/off/0; unset=false; malformed including explicit empty raises). Reject `ACCELERATE_DYNAMO_BACKEND` other than unset or case-insensitive `no`; do not reject harmless subordinate DS option names when the plugin is disabled. DDP coordinator also rejects forwarded `--use_cpu`, preserving mandated GPU topology; separate CPU subprocess unit tests do not use that coordinator path.

- [x] Write real coordinator dry-run and worker-shell RED tests. Name failure mutations: wrong launcher backend, missing topology, inherited DS mode, dropped objective args, invalid active aux/ZeRO launch. Use controlled shell inputs and assert parsed argv/exit/no files, not source text.

```python
plan = build_launch_plan(stage='stage2', tuning_method='dora', backend='ddp',
                        run_id='ddp-contract-test', main_process_port=29871)
assert plan.zero_stage is None and plan.deepspeed_config is None
assert plan.world_size == 4
with pytest.raises(ValueError):
    build_launch_plan(stage='stage2', tuning_method='dora', backend='ddp',
                     zero_stage=1, run_id='ddp-contract-test', main_process_port=29871)
# Run the real worker --dry-run with complete ILVR topology and fixed fake input
# paths through the existing launcher test fixture; inspect its JSON argv.
assert '--multi_gpu' in argv
assert '--use_deepspeed' not in argv
assert '--deepspeed_config_file' not in argv
assert argv[argv.index('--num_processes') + 1] == '4'
```

- [x] Run RED `pytest -q tests/multinode/test_native_ddp_launch.py` with the repo environment above; capture actual missing-feature failures. Add separate causal RED before each environment/preflight guard beyond the initial API test.
- [x] Implement the exact LaunchPlan two-way validation and conditional node-info argv/config inspection. Reject DDP for Stage1 and unsupported historical matrix/capacity/inference commands; do not make shared parser additions silently enable them.

```python
if backend == 'ddp':
    if stage != 'stage2' or zero_stage is not None:
        raise ValueError('native DDP requires Stage2 without a ZeRO stage')
    config_path = config_digest = None
else:
    # Keep existing selected_deepspeed_config/load_deepspeed_runtime_profile branch.
    if zero_stage not in (1, 2, 3):
        raise ValueError('DeepSpeed requires an explicit ZeRO stage')
```

- [x] Implement worker backend selection only after strict fixed-topology and conflicting-environment validation. On DDP use `ACCELERATE_BACKEND_ARGS=(--multi_gpu)` and no training `--deepspeed_config`; retain old DS arguments verbatim. Validate on coordinator and worker (remote inheritance matters). Active auxiliaries fail early on DeepSpeed; zero-weight/off retain existing backend rules.
- [x] Add input-preflight RED tests for entropy reference and original DoRA base missing/nonlocal on either node. In live preflight, use existing Stage1 PEFT metadata snapshot decoder to discover local base before any model/config download boundary; append reference/base to existing required paths. `lambda_dyn=0` skips reference work. Readiness, ownership, storage checks and existing installed-version checks remain intact. Dry-run is not live remote input validation.
- [x] Run covering GREEN `pytest -q tests/multinode/test_native_ddp_launch.py tests/multinode/test_topology.py tests/multinode/test_preflight.py tests/multinode/test_coordinator.py tests/stage2/test_launcher.py`; preserve the known baseline platform test failure if reached, do not alter it. Exact known node is `tests/stage2/test_launcher.py::test_production_entrypoint_imports_with_build_marker_then_restores_platform`, which asserts original platform.python_version ends in `+`, but host is `3.11.15`. Run `bash -n run_stage2_training.sh`, diff check, self-review, report exact red/green and commit only Task1 files.

### Task 2: Native manifest ownership, collection, and actual save/resume

**Status:** Complete; implementation615a626, task review approved. Covering
artifact suite190passed111.98s. Real four-rank CPU Full/DoRA resume/control
matches model/optimizer/RNG exactly; actual LoRA loader probe passes. Worker
warnings retained; GPU/NCCL/real two-host transfer remain unrun.

**Files:** Modify `src/multinode_artifacts.py`, `src/multinode.py:_checkpoint_artifact_identity` and narrowly necessary `src/multinode_runtime.py` artifact runner call sites; create `tests/multinode/test_native_ddp_artifacts.py`, `tests/multinode/test_native_ddp_checkpoint.py`, optional `tests/multinode/native_ddp_helpers.py`.

**Interfaces:**
- Keep public `build_node_manifest`, `observe_checkpoint_identity`, `validate_runtime_identity`, `validate_collection`, `validate_resume_identity`, collection and redistribution signatures. Select strict V1 versus V2 from runtime identity and matching manifest schema version; do not accept V2 fields in V1.
- `_checkpoint_artifact_identity(plan, *, commit, global_step)->dict` emits old identity unchanged for DeepSpeed, V2 fields from the DDP plan/config for native. `stage2_objective_config=None` means feature off; enabled weights-zero has objective_enabled true.
- `build_node_manifest(...canonical_owner=False)` validates original node ownership; `canonical_owner=True` requires actual full identity/common state plus ranks0..3. Original machine1 only allows top-level RNG2/3, with exact trusted identity equality before transfer.
- The native node1 resume read may pass optional `resume_destination=True` through the existing runner and bounded internal artifact-manifest CLI flag. Classify pinned actual contents as exact original RNG2/3 or fully validated canonical common/all-four RNG; reject partial/mixed states without retrying weaker validation. This preserves already-complete destination handling in existing redistribution; public artifact/collection/redistribution signatures and legacy calls remain unchanged. Obtain causal complete-destination and partial-state tests before implementing this seam.
- Required `deepspeed-runtime.json` is NOT a forbidden DeepSpeed state marker: native requires its existing schema with `enabled=false`, `zero_stage=null`, `config_path=null`, `config_digest=null`, `world_size=4`. Forbidden state means `latest`, `global_stepN` and DeepSpeed model/optimizer shard files.
- Test helper, if needed: `run_native_checkpoint_probe(method: str, *, output_root: Path, log_path: Path)->dict` launches four CPU Gloo processes, two logical node roots, real tiny Stage2Trainer save/resume, and returns observed JSON summary. The helper and concrete filesystem-mapped transport live only in tests. Output roots are distinct directories below repo checkpoints.

- [x] Write artifact contract RED tests with literal V2 identity fields, legacy V1 controls, malformed mixed identities, missing optimizer/RNG/sidecar, conflicting stage/backend/step, illegal duplicates/paths and DS markers. Test actual files/readers, not same-builder expectations.

```python
ddp_identity = dict(stage='stage2', tuning_method='full', zero_stage=None,
    deepspeed_config_digest=None, world_size=4, rank_mapping={'0':[0,1],'1':[2,3]},
    global_step=2, model_state_layout='trainer', optimizer=True, scheduler=True,
    rng_ranks=[0,1,2,3], stage_sidecar='stage2-sampler.json', commit='b'*40,
    run_id='ddp-checkpoint-contract', execution_backend='ddp', objective_enabled=True)
assert validate_runtime_identity(ddp_identity) == ddp_identity
with pytest.raises(ValueError):
    validate_runtime_identity(dict(ddp_identity, zero_stage=0))
```

- [x] BEFORE collection implementation, add and run actual native-save behavioral RED: a tiny full model uses four CPU Trainer ranks with root selected by `global_rank//2`; observe real rank0 common files and local RNG2/3. Feed those actual directories into production manifest/collection functions; today's DS assumptions must fail causally, not due missing fixtures. The subprocess explicitly sets `CUDA_VISIBLE_DEVICES=''`, `ACCELERATE_MIXED_PRECISION=no`, OMP/MKL threads1, bounded timeout and retained failure log.
- [x] Implement strict DDP V2 schema and native observed identity. Verify disabled/world4 runtime sidecar, checkpoint-N/trainer_state step, method-specific full/PEFT loader requirements, sampler and conditional objective sidecars. Required native common files include optimizer.pt, scheduler.pt, trainer_state.json, training_args.bin. Reject latest/global_step DS state. Do not deserialize optimizer weights merely to classify layout.

```python
if identity.get('execution_backend') == 'ddp':
    assert identity['model_state_layout'] == 'trainer'
    # Native structural validation uses actual filenames plus bounded sidecars.
    required = {Path('optimizer.pt'), Path('scheduler.pt'), Path('trainer_state.json'),
                Path('training_args.bin'), Path('deepspeed-runtime.json'),
                Path('stage2-sampler.json')}
    if identity['objective_enabled']:
        required.add(Path('stage2_objective.json'))
    missing = required - observed_paths
    if missing:
        raise ValueError(f'missing native checkpoint file: {sorted(map(str, missing))[0]}')
```

- [x] Implement original RNG-only node rule and canonical re-observation through existing pinned-directory/readers. Keep no-overwrite collection/redistribution transport; test only RNG2/3 is pushed, then complete canonical state is copied back for native resume. Require exact all-four RNG and correct owner/common-file rules.
- [x] BEFORE resume changes, run behavioral RED through production redistribution with the actual saved state; no preconstructed successful manifests. Resume all four real Trainer ranks and compare optimizer/scheduler/state/RNG/reference invariance against the saved checkpoint and uninterrupted tiny-model control. Keep total max_steps3/schedule unchanged: stop the initial test at step2 with callback, resume to3 rather than changing the schedule. Include full and actual PEFT/DoRA fixtures; LoRA loader-contract cases too. Preserve fixed reference identity; no GPU or SSH launch.

```python
observed = run_native_checkpoint_probe('dora', output_root=checkpoint_fixture_root,
                                       log_path=probe_log)
assert observed['original_local_files'] == ['rng_state_2.pth','rng_state_3.pth']
assert observed['resumed_steps'] == [3,3,3,3]
assert observed['reference_changed'] == [False,False,False,False]
assert observed['matches_uninterrupted'] is True
```

- [x] Run GREEN `pytest -q tests/multinode/test_native_ddp_artifacts.py tests/multinode/test_native_ddp_checkpoint.py tests/multinode/test_artifacts.py`; report original-save/collection/resume red chronology, exact observed file sets and numerical comparisons, unrun GPU limitations; self-review and commit Task2.

### Task 3: Actual DDP producer and final completion witness

**Status:** Complete; implementationc07a026, task review approved. Frozen-tree
covering182passed4skipped161.16s; actual four-rank CPU success/failure records,
CE-only canonical digest and completion verified. Four existing CUDA/opt-in
skips and worker warnings are disclosed; GPU/real-host execution unrun.

**Files:** Modify `src/stage2/evidence.py`, `src/main.py:publish_stage2_training_evidence`, `src/multinode_runtime.py:_finite_loss_reference/write_training_witness`, narrowly necessary `src/multinode.py` training call sites; create `tests/multinode/test_native_ddp_evidence.py`; extend native checkpoint probe to exercise publication.

**Interfaces:**
- Extend existing build/decode/publish Stage2 producer APIs with defaulted `execution_backend='deepspeed'`, `objective_identity_sha256: str|None=None`; zero_stage optional only when backend ddp. Decoder expected backend defaults deepspeed. Old calls emit exact V1; DDP emits schema_version2 plus execution_backend and objective_identity_sha256; other existing evidence/rank fields unchanged.
- Actual main publisher derives backend from prepared Trainer/Accelerator runtime and bound profile/world, not CLI/environment labels alone. DDP must have native distributed world4 and disabled DS. Rank0 alone observes/hashes common checkpoint state; all four ranks contribute actual optimizer/global step/loss/gradient records.
- Final witness uses schema_version2 and execution_backend ddp for native, with existing run/attempt/ranks/path/finite_loss_evidence fields. Add optional `objective_identity_bytes: bytes|None=None` to `write_training_witness` and its finite-loss helper; legacy callers remain unchanged. Expected objective config comes from Task1 frozen plan.
- Existing remote readers handle lifecycle/Stage1 evidence, not the new objective sidecar. Add narrowly scoped `SubprocessCoordinatorRunner.read_stage2_objective_identity(checkpoint_step: int)->bytes` and internal CLI `stage2-objective-identity-show --run-id RUN --checkpoint-step N`. It reads only the pinned canonical checkpoints/RUN/checkpoint-N/stage2_objective.json using the existing bounded no-follow file primitives (maximum4MiB), after plan/run/step validation. No arbitrary path argument, generic metadata service or additional persistent copy. DDP coordinator calls it only when objective enabled, after canonical collection validation, and passes bytes directly to the witness verifier.

- [x] Write RED on real builders/readers and actual main publisher boundary: strict V2 types/nullable hash, V1 roundtrip unchanged, missing ranks/nonfinite/wrong backend/world mismatch fail; enabled CE-only requires digest and disabled config rejects one. Use literal expected JSON fields, not builder-vs-itself comparisons.
- [x] Run RED `pytest -q tests/multinode/test_native_ddp_evidence.py`. Add separate distributed behavioral RED before changing collective observation/publication: original machine1 has no model/common sidecar, rank0 observation fails, every process reports failure within a bounded subprocess without hanging.
- [x] Implement only V2 branch and actual prepared-runtime checks; rank0 observe/hash and collective error propagation BEFORE success broadcast, using existing failure-reduction pattern on the actual backend-compatible device (CPU for Gloo tests, local CUDA for NCCL). Do not alter V1 semantics or add generic recovery.

```python
# All ranks execute the same reduction before any success broadcast.
local_error = None
if dist.get_rank() == 0:
    try:
        observed = observe_rank0_checkpoint()
    except Exception as error:
        local_error = error
if collective_failure_count(int(local_error is not None)):
    raise RuntimeError('native checkpoint observation failed') from local_error
# Broadcast only after successful collective status; never read common files on node1.
```

Here `observe_rank0_checkpoint` and `collective_failure_count` are the local
observation/reduction blocks in the existing publisher, not required new public
functions. Keep them small and use existing helpers where their device/error
contracts match; no test-only production extension.

- [x] Wire coordinator producer config/digest check against collected canonical sidecar and frozen requested config; missing/changed sidecar or wrong backend cannot produce success. Ensure a local RNG-only checkpoint path is not used for common-file hashing before redistribution: use the fixed-path remote objective reader described above and existing bounded read primitives. Test its path/no-follow/size failures before implementation. Reuse full collected identity rather than fabricating claims.
- [x] Extend the four-process CPU native checkpoint probe to exercise actual publisher and coordinator witness consumer; verify four real rank records and digest equality, enabled-zero-weight behavior, propagated failure, and no success record on incomplete/cross-backend state.
- [x] Run GREEN `pytest -q tests/multinode/test_native_ddp_evidence.py tests/multinode/test_native_ddp_checkpoint.py tests/multinode/test_coordinator.py tests/stage2/integration/test_distributed_contract.py -m 'not gpu'`; record warnings/failures at exact scope, self-review and commit Task3.

### Task 4: Comparison examples, operator docs and combined handoff

**Status:** Delivery implemented67a7676, task review and docs-only remainder
clarification41a6285 approved. Executable shell26passed; actual coordinator
dry-run verified without launching. Combined whole-branch review and final
project CPU verification remain pending below.

**Files:** Create `run_stage2_objective_examples.sh`, `docs/stage2-objective.md`, `tests/stage2/test_objective_examples.py`; modify only relevant `docs/multinode-training.md` native-DDP instructions. This is also original objective plan Task5, not a second wrapper/review cycle.

**Interfaces:** Wrapper mode `ce|entropy|combined`; required --stage1, --data, --image-root, --run-id, --max-steps, --main-process-port; entropy/combined require positive --lambda-dyn, combined positive --lambda-vis. --dry-run prints actual coordinated command without launching ranks; require explicit --launch for execution, reject both together. Same inputs/seed/baseline across modes; select explicit --backend ddp and enable_stage2_objective for ALL three, including CE.

- [ ] Write executable-shell RED tests using controlled environment and coordinator boundary: compare actual argv/env for three modes, same source/data/seed/LR/batch/update budget, different weights/skipped branches; errors for missing paths/invalid mode/weights/port/output collisions. Dry-run starts no ranks/writes no checkpoints. Do not grep shell source or assert a mock's existence.

```python
assert ce_args[ce_args.index('--lambda_dyn') + 1] == '0'
assert ce_args[ce_args.index('--lambda_vis') + 1] == '0'
for argv in (ce_args, entropy_args, combined_args):
    assert argv[argv.index('--backend') + 1] == 'ddp'
    assert '--enable_stage2_objective' in argv
    assert argv[argv.index('--seed') + 1] == '42'
assert observed_environment['GRADIENT_ACCUMULATION_STEPS'] == '2'
assert observed_environment['LEARNING_RATE'] == '5e-5'
```

- [ ] Run RED `pytest -q tests/stage2/test_objective_examples.py`; implement direct readable wrapper. Baseline micro1/global8/LR5e-5/seed42/warmup10/BF16/DoRA r32 alpha64 dropout0.05/latent8, supervised markers, supported SDPA and native non-reentrant GC. Use supplied max-steps (derived by operator from actual retained data/remainder), save_steps same terminal budget; never hardcode Stage2=431. No GPU launch in tests.
- [ ] Write operator/objective docs with exact formulas/window/vocab/graph/reference/M semantics from approved original design, backend and checkpoint support, actual CPU evidence, known platform failures and unrun GPU/ZeRO/training/capacity. Show three same-source/data/seed examples with distinct run IDs, required free ports and explicit step budget; default Stage1 is reproduced checkpoint-431, not old comt checkpoint. D/G do not establish task performance.
- [ ] Run GREEN `pytest -q tests/stage2/test_objective_examples.py`; `bash -n run_stage2_objective_examples.sh run_stage2_training.sh`; use actual coordinator dry-run to confirm final forwarded args, no training. Consolidate exact objective/DDP evidence into durable docs before any scratch cleanup.
- [ ] Self-review and commit delivery. Root conducts ONE combined whole-branch requesting-code-review including both approved plans and deferred objective minors, then one consolidated fix wave/scoped rereview if needed. Reuse this gate for original Task5; no duplicate final review.
- [ ] Root selects final verification/finishing applicability first. Project CPU suite is `pytest -q tests -m 'not gpu'` with CUDA hidden and repo PYTHONPATH; use it once for the final suite claim, plus distinct shell/dry-run/requirements checks once. Known baseline failures must remain visible; do not declare a full pass or advance finishing options if required evidence fails. No GPU training, merge, push or canonical deployment without separate authority.

## Plan self-review coverage

Spec sections1–4 map to Task1; sections5–6 to Task2; section7 to Task3;
sections8–10 to Tasks1–4 and exact-scope final verification; section12 enabled
CE-only/trusted identity clarifications to Tasks1–3. GPU first nail is explicitly
unrun and not converted into a CPU pass. All nine original objective options
are forwarded by Task1, no second parser/default source is introduced.

Existing public artifact APIs stay unchanged; newly optional producer args have
legacy defaults. Task1 frozen config feeds Task2 identity and Task3 evidence;
Task4 consumes Task1 CLI/forwarding after Tasks2–3 complete. Same-file work is
sequential. Full/PEFT runtime ownership tests precede native artifact behavior,
and distributed failure tests precede changed collectives. No task authorizes
new objective/backward code or a whole ZeRO/lifecycle framework.
