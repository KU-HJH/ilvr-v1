# Stage 1 Visual History Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development, the user's retained implementation choice. Steps use checkbox syntax. No switch to root implementation.

**Goal:** Train the approved historical-baseline-controlled experiment changing only the visual-history summary.

**Architecture:** Development starts at pinned baseline c21b4e5c311204df4281317219b3f4a6b95e4d3b in native isolated checkout /root/.codex/worktrees/0121/ILVR, branch codex/stage1-visual-history-only-20261019. Two sequential reviewed tasks modify the target builder and bind the existing coordinator to an identical fixed execution checkout on both nodes: /root/workspace/ILVR/.worktrees/stage1-visual-history-only-20261019. Runtime copies use existing canonical data/checkpoint links. Historical DeepSpeed0.18.0 environment is reused locally and copied into the owned remote venv without installation; canonical shared environments are preserved. The archived Trainer/model/collator/loss/initializer stay byte-identical.

**Tech Stack:** Repo Python3.11 venv, existing PyTorch/PEFT/Transformers, pytest, unchanged two-node coordinator, FA2, BF16/ZeRO1. No install/download or new orchestration.

**Approval:** User `진행해` approved exact commitment v2 SHA256486770c75a155ba2187488db221cab77c580fe1121e54df106b753cf9a4ec70b after the final short design. Full commitment in adjacent specs; frozen bytes preserved. Training is authorized after mandatory development/verification gates. No evaluation requested.

## Setup and evidence

- [x] Canonical AGENTS/training-defaults/operator guide read; baseline log and saved training_args examined.
- [x] Native worktree created from exact baseline via approved new-checkout scope, data/checkpoint/venv links prepared; no dependency install. Historical target baseline command passed26 tests in2.91s, complete log logs/stage1-visual-history-only-20261019/baseline-targets.log.
- [ ] Keep logs under logs/stage1-visual-history-only-20261019, generated observations under results/stage1-visual-history-only-20261019. Pytest model fixtures use a fresh --basetemp below checkpoints/stage1-visual-history-only-dev-20261019 (parent exists), never /tmp.
- [ ] Every command uses PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src and ./venv/ilvr-venv/bin/python. CPU tests do not start training.

## Task 1: Replace only the historical visual-history term

**Files:** Modify src/stage1/targets.py; create src/stage1/history_metric.py by exact reuse of reviewed current metric; create tests/stage1/test_visual_history_only.py; update only causally changed expectations in tests/stage1/test_targets.py if necessary. No Trainer, EMA, composition, parser, tokenizer, model, collator or loss edits.

- [ ] Write actual comparative failing tests before production edits. Load inspected historical target builder from git:c21b4e5c311204df4281317219b3f4a6b95e4d3b:src/stage1/targets.py in an isolated test namespace; existing LiteralSample/make_case fixtures in tests/stage1/test_targets.py can supply frozen teacher observations. Record _select inputs to compare global intent/text positions, first-block native-dtype query/candidates/order and downstream visual-only differences. A three-block nonuniform, unequal-size visual history fixture must distinguish all-history content weights from baseline last-nonempty mean. Use literal independent FP32 cosine/softmax arithmetic as the expected history; do not compute expected outputs by calling the product helper.
- [ ] Cover first-block/header/native BF16 parity; later content-dependent selection; no text/history; empty middle selection; per-sample reset; future/current helper exclusion; raw target packing, stable ties; zero/near-zero norm errors. Do not assume synthetic tests establish production accuracy.
- [ ] Run the new file with the environment above and --basetemp=checkpoints/stage1-visual-history-only-dev-20261019/task1-red; capture complete RED output. Missing-feature assertion failure is required, not an import/fixture error.
- [ ] Copy the self-contained file /root/.codex/worktrees/6b9d/ILVR/src/stage1/history_metric.py exactly, verifying SHA25613264bb8cd0a2bdbd2ed43f7c0b1f2c0d3a046d5ddef34157bab4b97cd245f9e. Import only content_history. Do not call compose_content_query: historical query composition remains intact.
- [ ] In the historical loop, initialize current_text = row_hidden_last[:0] on every block; assign the existing selected text rows and preserve their existing mean. Replace previous_selected_mean only with:

```python
previous = torch.cat(packed, dim=0) if packed else row_hidden_last[:0]
history = content_history(current_text, previous)
if history.present:
    query_parts.append(history.summary.to(dtype=row_hidden_last.dtype))
```

Keep the existing query stack/mean, _select call, packing and returned data. Remove obsolete last-block state only. All history is this builder's own earlier raw selections; no student latents, accumulated text, future values or injected teacher prefix.
- [ ] Run GREEN focused command for new file plus tests/stage1/test_targets.py, with fresh task1-green basetemp. Preserve RED/GREEN logs, self-review actual diff, commit only Task1 files and report exact commit/status/concerns.
- [ ] Independent spec review against commitment/task and actual c21 diff. Fix only through original implementer and re-review.
- [ ] After SPEC PASS, independent quality review with actual code/evidence. Fix and re-review. Root reads complete logs/diff; no agent assertion substitutes for evidence.

## Task 2: Fixed isolated repository bindings for the existing coordinator

**Files:** Minimal necessary bindings in src/artifact_paths.py, src/multinode.py, src/multinode_artifacts.py, src/multinode_capacity.py, src/multinode_runtime.py, and the necessary fixed checkpoint/result constants in src/multinode_evidence.py and src/multinode_results.py; path-only literals in run_stage1_batch_probe.sh, run_stage1_training.sh, run_stage2_training.sh, run_stage1_zero_matrix.sh and the two lifecycle harness _phase_paths functions identified by the full-suite failure; create tests/multinode/test_visual_history_checkout.py; only tests with causally stale literal-root expectations may be adjusted. Keep all ownership, pidfd, path traversal, same-commit, environment, rendezvous, node and artifact validation behavior.

- [ ] Inspect the actual bound paths and source-supported worktree conventions. The experiment execution root is fixed and identical inside both containers:

```python
RUNTIME_ROOT = Path('/root/workspace/ILVR/.worktrees/stage1-visual-history-only-20261019')
REMOTE_HOST_ROOT = Path('/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-visual-history-only-20261019')
```

Use these exact per-experiment fixed bindings in the historical constants, matching its pre-existing fixed-root architecture. No generic new CLI/configuration, lifecycle, transport, retry/recovery system or weakened check. Existing canonical base-model/data input paths remain unchanged. The root controller provisions committed runtime copies at these paths; do not launch training from development checkout.
- [ ] Write a failing test that exercises actual build_launch_plan/ClusterSpec and both constructed node commands/dry-run paths, requiring four fixed ranks, master10.1.102.41, bond1.102, isolated coordinator/script/config/Python/log/result paths and preserved canonical checkpoint physical storage. Reject foreign/traversal/unsafe ownership paths through existing guards. Do not merely assert a constant exists.
- [ ] Run RED and capture complete output before binding edits.
- [ ] Change only necessary repository/host-root constants and related minimal PATH binding, for example historical _REPOSITORY_ROOT, _LOCAL_REPOSITORY, _REMOTE_HOST_REPOSITORY and CANONICAL_REPOSITORY_ROOT. Preserve historical input resolution and all coordinator logic. If an unexpected required change exceeds path binding, report NEEDS_CONTEXT before coding it.
- [ ] Run GREEN for the new file, relevant existing artifact/topology/coordinator/preflight tests and shell dry-run integration. Update only stale path literals established by those failures, retaining assertions about protection and call/order behavior. Commit Task2 with RED/GREEN evidence and actual changed-file inventory.
- [ ] Independent SPEC PASS, then separate QUALITY PASS, using actual diff and command outputs. Fixes go through implementer with renewed applicable review.

## Final review, verification and requested training

Live-preflight recovery after fullsuite r2 PASS: historical validator correctly rejected current shared DeepSpeed0.18.7. Reuse existing `/raid/workspace/ILVR/venv/ilvr-venv`0.18.0, with task-owned links/local and copied remote environment. Its stale absolute Accelerate shebang requires one additional necessary interpreter-path binding: run_stage1_training.sh prepends repository_python to existing repository_accelerate. Original Task2 implementer owns this change and causal real-console --help test, commit55ae6923277ad00a7eb2fbd8331f1deaadeac885. No package installs, version-guard weakening, other dispatcher or training flags. Renewed SPEC then QUALITY and final whole review precede fresh fullsuite r3 in the actual0.18.0 environment. r2 remains recorded successful2384passed6skipped in0.18.7, not a runtime launch pass. New source/environment justify r3, with unused final-pytest-r3/final-full-suite-r3.log instead of overwriting r2.

- [ ] Independent final whole-change review from baseline c21 through both tasks, including loss/profile/source-preservation inventory and probe evidence. Root resolves all required findings before proceeding.
- [ ] Invoke verification-before-completion and finishing-a-development-branch together. The applicable project full-suite command below is the justified r3 after the recorded live-preflight environment failure and narrowly reviewed interpreter recovery; run once after renewed final review; read complete output, not just the tail. Precreate only its dev basetemp parent and require unused basetemp/log identity. No extra identical broad suite unless an actual change/failure requires it.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest tests -q --basetemp=checkpoints/stage1-visual-history-only-dev-20261019/final-pytest-r3 > logs/stage1-visual-history-only-20261019/final-full-suite-r3.log 2>&1
```

- [ ] Verify copied metric hash, git diff --check and exact zero diff against c21 for Trainer/composition/EMA/model/collator/PEFT/loss/initializer/Transformers. Record reviewed product revision and any subsequent control-only commits.
- [ ] Preserve branch/worktrees. Human owns merge/push/discard; do not block already-approved training on a branch disposition decision or perform unrequested merge/cleanup.
- [ ] Provision identical committed execution copies via existing snapshot/bundle/checkout procedure; use checkpoint/data/venv links and separate results/logs; verify matching source and effective environment on both nodes. Existing runtime directories must not be overwritten. No manual rank launch.
- [ ] Verify original base/model/processor/TRAIN and retained order against archived baseline. Inspect archived saved training_args and actual sampler/remainder to retain1 epoch/431 cap, micro1/GAS2/global8. Record declared8192 as unused; actual saved max_seq_length32768 with skipped dataset preparation and collator truncationFalse remains unchanged. Historical framework compatibility evidence must be stated at its supported scope.
- [ ] Source no stale run environment. Explicit applicable baseline overrides: original base, CE1/sim1, no preference/entropy/KL, DoRA r32/alpha64/dropout0.05/seven projections, latent8/EMA1, BF16/FA2/nonreentrant checkpointing, workers0, LR5e-5/warmup10/seed42, AdamW fused/linear/decay0.01/clipping1, GAS2/micro1, epochs1/max431/save431/final save. Use existing script's actual resolved arguments and constants; verify effective Python Namespace/SFTConfig on both nodes rather than invent unsupported shell flags.
- [ ] Existing coordinator dry-run and live preflight must pass for both nodes/four ranks/compatible source and environments/input paths/free resources. Then launch fresh detached run ID stage1-visual-history-only-egb8-s431-<UTC timestamp> through run_multinode_stage1_training.sh with explicit free port and original base/TRAIN/images. Save command/launch identity, runtime actual ranks0..3, and process ownership.
- [ ] Continue monitoring detached training without signaling foreign jobs. On runtime failure preserve logs, identify the cause and route any actual code fix through implementer/TDD/review, then fresh run ID. Do not invent recovery infrastructure or run an extra control/evaluation.
- [ ] Validate terminal checkpoint431/finite producer evidence and all owned worker/coordinator exits once; report requested training result with precise scope. Before final user response, run canonical write_ntfy_summary.sh with actual outcome.
