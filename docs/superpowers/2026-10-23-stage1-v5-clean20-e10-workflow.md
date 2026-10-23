# Stage1-v5 clean20, entropy/mass and ten epochs

## Intake and actual phase

User requests: `Clean entropy 기준 dil-bce-orig, entropy와 mass KL을 합친
setting으로 10 epoch 학습진행해보고자 한다.` User explicitly answered `20%`
to the question asking the clean-entropy selection percentage.

Understood as a fresh Stage1-v5 run using clean-entropy top20% targets for
DIL-BCE-ORIG and the existing V-off entropy/mass objective, with the other
current Stage1-v5 terms retained. The requested ten epochs override only the
standing exposure duration. Training is already authorized; development needs
the applicable final design approval because current training rejects clean
selection. No new product implementation or training launch has occurred.

Active lifecycle: bounded brainstorming design-commitment gate. Root applied
readchk using the conversation and code, read canonical AGENTS.md,
documents/training-defaults.md and the complete multinode operator guide.
The user-selected Subagent-driven method persists. Prior task's full-suite
exception is not a standing exception for this new change.

## Frozen proposal and first-nail critique

Proposal v1: `specs/2026-10-23-stage1-v5-clean20-e10-design.md`, SHA256
`2d07abdb82fcd698477e4998227a03b0315f48ae00bd8126b6ff4f87bb16be77`.
Both advisory passes use this same version; it was not rewritten between them.

Root applied hate and captured exactly:

```json
{"root":"Merely enabling the clean selector can select20% of the numerator while the prefetched DIL denominator still counts100%, changing loss strength and wasting the intended ten-epoch comparison.","first_nail":"Before any GPU launch, compare the actual existing clean-selector mask cardinality with the proposed metadata count on ragged per-sample token lengths, stable ties, empty blocks/ranks and a partial accumulation window, then witness accumulated gradients against an independently formed global selected-token mean. This is a proposed post-approval causal verification, not evidence that the support already works."}
```

Static premise evidence: v5_selection.py:125-134 ranks each sample and uses
select_entropy_masks; entropy_selection.py uses exact Fraction(str(percent))
and ceil. v5_trainer.py:76-82 currently reduces all metadata query counts.
The selected cardinality is therefore metadata-computable without a new model
pass. Runtime count/gradient agreement is still pending implementation/TDD and
is a strict launch gate; no costly premise experiment was performed.

## Macrothink: independent spread

This gate applies because running the wrong ten-epoch objective materially
costs more than two bounded fresh read-only sessions. Stripped facts input:
`specs/2026-10-23-stage1-v5-clean20-e10-fresh-read.md`, SHA256
`ea2e9a392836ff22aa1f57d5c6f9c42089ddd6c602f3cf4a8db157f8802eef7a`.

Readers `/root/clean20_fresh_a` and `/root/clean20_fresh_b` inherited the root
model with fresh/no-history contexts. Neither edited files, ran training or
spawned agents. Root collected their reports without steering them.

Incompatible divergence relative to frozen proposal v1: none. Compatible
divergence: none that requires a different implementation direction. Each read
is classified **convergent**: it independently identified selected-vs-eligible
denominator semantics, same-mask scope for DIL/entropy/mass, unchanged native
and Text-KL terms, original-block auxiliary normalization, stable per-sample
ceiling, explicit saved identity, actual sampler tails and real worker profile.
ReaderA additionally preferred the training-only coordinator over the wrapper
that hardcodes one epoch and adds evaluation; proposal v1 already specifies
that coordinator and no evaluation campaign. Both warned against treating a
3450-line count or matching package versions as runtime proof. Convergence is
reassurance only and does not verify the implementation or authorize training.

## Read-only preparation evidence

Observed two idle H100 NVL GPUs on each node. Local bond0.11 has172.16.11.40;
remote has172.16.11.41. Remote llm_kd is running; preserve all containers.
Observed matching principal package versions on both nodes: Python3.11.15+,
torch2.7.1+cu128, transformers4.50.0.dev0, deepspeed0.18.7,
accelerate1.10.1, peft0.17.1, flash-attn2.8.3. This is preliminary observation,
not final compatible-code/resource preflight or runtime participation evidence.
Original base and input locations exist on both nodes; checkpoint storage has
ample free space and its symlink is preserved. No new checkpoint was loaded.

TRAIN.jsonl SHA256 matches both nodes:
`5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8`.
Fresh prepare_stage1_comt_examples plus unchanged audit_prepared_loader report
3450 retained samples, four actual CPU-sharded loader lengths863,216 optimizer
updates for one exposure, final partial window3 and configured native
denominator16. Ten epochs derive2160 updates; final actual ten-epoch
CLI/Trainer resolution is still required before launch.
Artifacts: `results/stage1-v5-clean20-e10-preflight-20261023/loader-budget.json`
and `logs/stage1-v5-clean20-e10-preflight-20261023/loader-budget.log`.

Source worktree HEAD68b46d906243b5338350990f42d429d8063a7deb has unrelated
pre-existing network modifications. A clean reviewed execution copy is needed;
do not stage or overwrite those changes silently. Minimal current-network
constants required by the existing coordinator must be independently reviewed
as part of execution readiness. Reviewed Stage2 runtime a39fe867 already uses
the current address/interface but is not a substitute for Stage1-v5 algorithms.

## Design commitment v1, final approval pending

Agreed problem/scope and final design are the frozen proposal v1. No material
algorithm or scope change follows the critiques. Updated factual evidence is
the fresh3450/863/216 audit and derived2160 total, matching the proposal's
earlier tentative budget. There are no unresolved disagreements.

The exact short commitment to present is:

- Clean entropy top20% per sample; use one mask for DIL, V-off entropy and mass.
  DIL averages globally over selected tokens; entropy/mass keep their existing
  original-block average, including selection-empty blocks.
- CE0.1, alignment0.1, DIL1, entropy0.1, mass0.05, Text-KL0.01;
  independent k50, beta1/margin0, no GT union. Keep original-base/frozen-teacher
  DoRA/legacy-query/ZeRO1 profile, global16 and LR5e-5.
- Ten epochs, derived2160 updates with the actual schedule confirmed before
  launch. Use existing four-rank coordinator and safe fresh run paths; retain
  epoch checkpoints and final artifact. No evaluation campaign is added.
- Extend the currently blocked clean selection path, give selected-count
  normalization an explicit saved identity, review/test the minimum needed
  support and current-network execution profile, then perform both-node final
  preflight and launch without another training-authorization question.
- First-nail runtime mask/count and gradient agreement remains unrun and must
  pass before the expensive launch. All other applicable development gates
  remain binding. No material empirical premise is waived by design approval.

This record is not approval. Final human approval of this exact commitment v1
is pending; no implementation agent has been dispatched for the support change.

## Final approval and bounded execution brief

The user replied `진행해.` to the exact short design commitment v1.
Approval binds this record at SHA256
`2a2fc1454a3028eafe1034500837af4899f055a5b1450a003a418db8b0a83da6`
and frozen proposal SHA256
`2d07abdb82fcd698477e4998227a03b0315f48ae00bd8126b6ff4f87bb16be77`.
Brainstorming's bounded path is complete; use one concrete task brief through
the retained Subagent-driven method, then independent task/final review and
fresh verification before the already-authorized training launch. No new full
implementation-plan artifact is needed for this bounded extension.

### Global constraints

- Keep per-sample detached clean full-vocabulary entropy selection with exact
  rational ceiling and stable predictor-position ties. One selected mask is
  shared by DIL, entropy and mass; CE/alignment/Text-KL scopes remain unchanged.
- DIL uses the actual-window all-rank selected-token denominator. Entropy/mass
  retain original-block normalization including empty blocks. No extra model
  forward, graph retention across the window, or new backward/optimizer owner.
- Preserve all-token training, historical checkpoint meaning and strict source
  identity. New selected-count contracts must be unambiguous and roundtrip;
  unsupported selectors remain rejected. Do not rewrite existing artifacts.
- Use only repository venv Python. No model-bearing files outside checkpoints;
  logs under logs and outputs under results. No training, remote mutation or
  extra agents from implementers/reviewers. Root owns operational launch.
- Preserve unrelated dirty changes. The exact pre-existing network-only code
  patch cohort below is approved for adoption/review into the execution revision;
  preserve every other pre-existing hunk, especially documentation network edits.

### Task 1: Enable clean selection with correct selected-token normalization

Implement approved design v1, BASE68b46d906243b5338350990f42d429d8063a7deb,
in /root/.codex/worktrees/1061/ILVR on existing branch stage1_v5.

Product scope: src/stage1/v5_config.py, v5_metadata.py, v5_trainer.py,
v5_resume.py, and only necessary closely related v5 config/selection/reduction
helpers; focused tests under tests/stage1/v5; docs/stage1-v5.md. Reuse existing
selection, reductions and contract patterns. No unrelated refactoring.

1. TDD first: add behavioral tests and observe missing-support RED, then enable
   clean selection. Only all and clean training are supported. Existing
   percentile selection and stable ties must remain exact.
2. Compute selected count per sample with Fraction(str(percent))/100 and ceil,
   matching the actual selector, including nontrivial fractions, ragged blocks,
   ties and empty samples. For clean20 use ceil(N/5), not ceil of the sum across
   samples. Prefetch counts across actual microbatches/ranks through the existing
   collective; preserve all's count. Witness actual mask/count equality and
   accumulated gradients against an independent selected-token global mean,
   including unequal/empty ranks, zero windows and partial tails. Keep auxiliary
   and native denominators unchanged. Preserve one ordinary backward.
3. Give selected-token reduction an explicit saved identity and matching
   schedule denominator. Preserve strict historical reads/rejections, all-token
   contracts, immutable reference and execution/source agreement. Clean resume
   must roundtrip and reject changed selection, percent or represented semantics.
   A narrow new clean schema/identity is acceptable; no general migration layer.
4. Retain numeric defaults already approved, including entropy0.1 -> mass0.05
   in the requested run; no unrelated coefficient or schedule changes. Do not
   make the existing one-epoch/all-token train/eval recipe silently launch this
   experiment. Root will use the existing training-only coordinator explicitly.
5. Adopt/review the exact pre-existing current-network substitutions in these
   files into the execution commit: run_stage1_training.sh,
   run_stage2_training.sh, src/multinode.py, src/multinode_artifacts.py,
   src/multinode_runtime.py; tests/multinode/test_artifacts.py, test_capacity.py,
   test_coordinator.py, test_dil_full_model_probe.py,
   test_evaluation_coordinator.py, test_lifecycle_matrix.py, test_preflight.py,
   test_topology.py; tests/stage1/test_launcher.py;
   tests/stage2/integration/test_distributed_contract.py;
   tests/stage2/test_launcher.py. These substitutions are only
   10.1.102.41 -> 172.16.11.41 and bond1.102 -> bond0.11. Verify the exact cohort
   against pre-existing-working-tree.diff, and do not create a new transport
   system. Existing patches/tests predate this task: do not invent TDD or review
   history for their original authorship; independently inspect and exercise
   their actual coordinator/worker behavior now.
6. Update current v5 usage to describe clean20 support, selected-token DIL mean,
   shared mask and saved identity. Preserve the documentation's separate dirty
   network hunk when staging only your task changes.
7. Use focused RED/GREEN, existing v5 tests and the smallest coordinator/worker
   current-network integration checks. The root owns one final full project
   suite after reviews; do not duplicate it. Logs go to
   logs/stage1-v5-clean20-e10-development-20261023. Use
   PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   PYTHONPATH=.:src:transformers/src with ./venv/ilvr-venv/bin/python.
8. Self-review, commit only approved task/network hunks, preserve all other
   existing modifications and root control records. Write a detailed report
   with RED/GREEN evidence, commands/exits/logs, files/commits, self-review and
   limitations; return the short status contract. Do not invoke Paperthin,
   spawn agents, launch training, or provision remote repositories.

### Preflight task/interface scan

| Task / interface | Producer and consumer inspected | Finding |
| --- | --- | --- |
| Task1 internal | metadata cardinality -> actual entropy mask -> accumulation count | deterministic per-sample ceiling permits prefetch; summing before per-sample ceiling is wrong |
| Task1 internal | clean config -> saved contract -> runtime schedule -> strict resume | new selected meaning must be explicit while all/historical meaning stays bound |
| Task1 internal | objective selected token sums -> parent GAS/rank scaling | token denominator changes only for clean; auxiliary/native reductions stay intact |
| Task1 internal | shared network constants -> coordinator/workers/snapshot -> matching tests | adopt exact existing mechanical patch cohort, independently review/test; preserve unrelated drafts |
| Task1 internal | code commits -> clean execution copy -> two-node launch | root provisions after review; original dirty worktree is not launched |

No contradictory tasks exist. Final full-suite verification is centralized
after independent reviews and will be reused for overlapping completion claims.

### Implementation clarification: checkpoint retention
Operational inspection found the existing argument builder always sets
save_total_limit=1. The approved outcome retains ten epoch checkpoints and a
final artifact. Root therefore clarified Task1 to add only explicit
--save-total-limit -> --save_total_limit plumbing through the existing path,
default1 unchanged, this run10. This realizes the already-approved retention
requirement; no new retention system or objective change is authorized.

## Reviewed implementation and fresh operational evidence

Task1 implemented by /root/clean20_impl (gpt-6.1-sol/high), commit
1a2356a4dacf63bd21a25ad8bda40c78262792f6, base68b46d9. Expected clean
RED15fail/130pass and retention RED2fail/10pass preceded their product changes.
Focused GREEN145, affected fixtures104, real tiny integration56, retention147;
current-network299pass/3opt-in Stage2 skips. The intermediate v5-wide run's
11 fixture failures were fixed and affected tests rerun; it is not a full pass.
Root read complete reports and successful logs.

Independent task reviewer /root/clean20_task_review (gpt-6.1-sol/high) approved
specification and quality with no findings. Independent final whole-change
reviewer /root/clean20_final_review (gpt-6-astra/high) approved68b46d9..1a2356a,
no findings/fix wave. Reports, briefs, diff packages and ledger were copied
with SHA256 inventory to logs/stage1-v5-clean20-e10-development-20261023/
workflow-evidence before any scratch cleanup. Original source worktree's11
unrelated dirty diff bodies match the pre-task snapshot; approved16-file
network cohort was adopted. See results/stage1-v5-clean20-e10-preflight-20261023/
dirty-preservation.json.

Native clean execution copy: /root/.codex/worktrees/f187/ILVR, reviewed1a2356a.
Remote clean copy: /root/workspace/ILVR/.worktrees/stage1-v5-clean20-e10-20261023,
inside original llm_kd, same commit. Existing canonical data/checkpoints/venv
symlinks were linked without changing their targets. Remote bundle verified
against an existing prerequisite; no old checkout/container was overwritten.

Fresh actual coordinator/worker dry-run argv was parsed and built independently
on both nodes. Source hashes, packages, dataset SHA and base metadata SHA match;
both tracked trees are clean, imports resolve current vendored Transformers.
BF16, AdamW fused/linear, LR5e-5, decay0.01/clip1, seed42/warmup10,
micro1/GAS4, query legacy, EMA1/DoRA r32/a64/dropout.05, decoderSDPA/visionFA2,
non-reentrant checkpointing, maxseq32768/workers0 and all approved losses match.
10epochs/max2160/save216/limit10/final are explicit. No resume requested.
Profiles and exact launch command are under execution-copy logs/
stage1-v5-clean20-entropy-mass-e10-20261023-075931/.

Actual worker CLI + freshly retained3450 examples + inherited Trainer loader
and actual Accelerate four-way CPU sharder + v5 parent schedule resolve all four
modeled ranks to863 microbatches/epoch,216 updates/epoch,10epochs/2160total,
final window3 and configured native denominator16. This is CPU metadata evidence,
not live distributed participation. ten-epoch-schedule.json records the witness.
A single fresh full-project pytest -q is running in the clean reviewed execution
copy. No full-suite or live training success is asserted yet. Branch-finishing
menu remains pending its green gate; no push/merge/cleanup was performed.

### Fresh full-suite and detached coordinator launch

On exact clean execution revision1a2356a, root ran once:
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q.
Exit0,4933passed/11skipped/1warning,1234.63s. Root read the complete output.
The warning is the existing Stage2 CPUAdam extension test reporting unset
TORCH_CUDA_ARCH_LIST; no failed test or training correctness error occurred.
Log/exit under execution-copy logs/stage1-v5-clean20-e10-verification-20261023/
final-full-suite.{log,exit}. This single run serves both completion verification
and finishing Step1; no duplicate suite was run. Fresh git diff --check range
68b46d9..1a2356a passed, and execution tracked state is clean.

Finishing detected an externally managed detached execution copy, Git dir
/root/workspace/ILVR/.git/worktrees/ILVR7 and common/root/workspace/ILVR/.git.
Root presented its required two-option menu through request_user_input_async;
integration choice is pending and no push/merge or inferred keep decision
occurred. Training is separately already authorized and continues without
requiring that optional integration decision.

Both nodes' original/helper image contents were hashed once:13231 unique files,
10072073582bytes, aggregate SHA256
0bf94ceb11e0623ac0979f5ef1ca2a63723615eedebc5a424b138cb3b8521d75, equal.
The coordinator was detached in owned tmux session
coord-clean20-e10-20261023-075931 using the recorded launch.sh and existing
run_multinode_stage1_training.sh. Its own final resource/input/ownership preflight
and runtime participation are still underway. Starting a tmux session alone
is not evidence of four-rank training or completed epochs.

### Preflight-only failure and minimal operational correction

The075931 launch exited2 before worker start: local fresh run paths required.
Root initially suspected profile parsing/logging but direct inspection disproved
that: no actual stage log/evidence existed. The run's checkpoint directory was
empty, created at the CPU schedule audit time. Traced to actual inherited
Trainer.__init__, transformers/trainer.py:693, os.makedirs(output_dir).
No GPU training or checkpoint weights had been created by that attempt.
Failure logs and the empty directory were preserved, with no foreign signaling.

Root applied systematic-debugging to this operational input failure. One causal
regression witness used a distinct audit checkpoint path: TrainingArguments
alone left it absent; constructing the actual CPU Trainer created it empty.
Evidence logs/stage1-v5-clean20-e10-verification-20261023/
fresh-path-causal-regression.json. This confirms the side effect rather than
changing a correct preflight guard. No product code, reviewed revision, objective,
schedule or environment changed; no new implementation/review/suite was needed.

Retry ID stage1-v5-clean20-entropy-mass-e10-20261023-082213 changes only the
run-ID argument; all other CLI tokens are identical to the independently parsed
both-node profile. Coordinator dry-run left its actual checkpoint path absent,
asserted before launch. Root did not instantiate an audit Trainer against this
new output. Detached coordinator coord-clean20-e10-20261023-082213 launched;
its own preflight/runtime progress remains to be witnessed. The first failure
is not marked as a successful training start.

### Initial live training witness and handoff

2026-10-23T08:35:36Z: detached coordinator remains alive, no exit receipt.
Root freshly observed remote global0/1 (pids1179203/1179204) and local global2/3
(pids987952/987953), all actual worker env WORLD_SIZE4/LOCAL_WORLD_SIZE2,
MASTER172.16.11.41:29843, both socket variables bond0.11. Saved schema3 was
published after the runtime cross-rank effective-contract agreement. Actual
parent optimizer progress reached22/2160. First ordinary step20 log: loss1.652,
grad_norm0.805355966091156, learning_rate4.976744186046512e-05, epoch0.09.
All three numeric metrics are finite; no traceback in the captured startup log.
This is logged aggregate/current norm evidence, not proof of every raw loss.

Artifacts in /root/.codex/worktrees/f187/ILVR:
- results/stage1-v5-clean20-entropy-mass-e10-20261023-082213/
  initial-runtime-observation.json and initial-runtime-contract.json
- logs/stage1-v5-clean20-entropy-mass-e10-20261023-082213/
  rank0-initial-stdout-rank0.log, rank0-initial-stderr-rank0.log,
  runtime-machine{0,1}-processes-final.json, exact launch command/session/config
- original rank0 live logs on remote llm_kd under the matching worktree's
  logs/multinode/<run-id>/initial/{stdout,stderr}-rank0.log.

Requested training is now running detached with clean20/entropy.1/mass.05,
10epochs2160updates/global16. Save cadence216 and retention10 explicitly realize
the approved epoch-preservation requirement; actual epoch outputs/final artifact
are not yet observed. The existing coordinator will collect the terminal
checkpoint after zero exits; earlier numbered distributed epoch shards must be
validated/collected with the existing route when complete. No ten-epoch finish,
checkpoint-completeness, accuracy, IB/GDR or convergence claim is made here.

Human integration menu remains unanswered; no push/merge/keep decision is
inferred. Workspaces are needed by the live experiment and remain in place.
The launch and initial-runtime validation are complete; eventual training results
and human branch handoff remain open. No extra training authorization was asked.

Controller ruling: to realize already-approved epoch preservation, expose the
existing Trainer save_total_limit with default1 preserved and run10. Existing
limit1 would rotate previous epochs; review/tests verify explicit10 reaches
Trainer, while actual ten-checkpoint retention remains unobserved. This is the
only recorded Ruling event; no findings were parked or silently waived.
