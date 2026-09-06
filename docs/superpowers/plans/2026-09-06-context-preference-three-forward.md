# Stage 2B Three-Forward Context Preference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 ours1 Stage 2B의 기본 학습 경로를 negative/reference 없이 clean target의 local context preference와 entropy를 계산하는 최대 3-scoring-forward 경로로 교정한다.

**Architecture:** 기존 differentiable shared-prefix KV와 chosen V/T suffix replay를 재사용한다. Loss 함수와 unchunked replay를 먼저 독립 검증한 뒤 설정·preflight·collator·trainer·launcher를 한 vertical slice로 연결한다. 마지막에 독립 dense oracle, 실제 호출 계측, GPU/PEFT smoke와 baseline 회귀를 검증한다.

**Tech Stack:** Python 3.11, repository venv, PyTorch, repository-local Transformers/Qwen2.5-VL, PEFT, DeepSpeed, pytest; dependency 변경 없음.

**Spec:** [2026-09-06-context-preference-three-forward-design.md](../specs/2026-09-06-context-preference-three-forward-design.md). 작업자는 이 plan과 spec을 모두 읽는다. 사용자의 writing-plans 명시 호출로 서면 검토 단계에서 계획 작성 단계로 진행했다.

## Global Constraints

다음 문장은 spec의 제약을 그대로 인용한다. 각 task에 모두 적용한다.

- 기존 ILVR Stage 2의 clean latent 생성은 유지하고 3-forward 제한에서 제외한다.
- preference margin `gamma_P=0`.
- entropy는 각 step 내부에서 토큰 평균을 구한 뒤 V/T 모든 step을 동일 가중치로 평균한다.
- K=1/4/8/16 모두 scoring decoder 호출은 최대 3회다.
- 현재 collator와 span mapper의 local microbatch size=1 제약을 유지한다.
- 기존 gradient accumulation과 distributed training의 batch 처리 의미는 보존한다.
- 공유 KV와 clean latent의 autograd 연결을 보존하며, 편의를 위한 detach나 inference cache 전환을 금지한다.
- 초기 기본값: beta=0.1, gamma_P=0, lambda_P=1, lambda_H=0.1, gamma_H=0.1.
- Clean CE 계수는 1이다. 기존 clean CE의 supervised label construction을 보존한다.
- 새 objective를 DPO loss라는 이름으로 보고하지 않는다.
- Python과 테스트는 worktree의 `./venv/ilvr-venv`만 사용한다.
- 결과는 worktree `results/`, 로그는 `logs/`, model-bearing artifact는 기존 `checkpoints/` symlink 아래에만 둔다.
- 본 학습 실행은 이번 구현 검증에 포함하지 않는다.

추가 운영 계약: 모든 실행은 /root/workspace/ILVR/.worktrees/ours1에서 한다. 결과/로그는 서로 같은 명시적 run ID를 사용하고 기존 실패 로그를 덮어쓰지 않는다. 실제 기존 clean latent decoder 호출과 backward recomputation만 제한에서 제외한다. Step sampling, sidecar chunking, reference forward, negative generation/preference rollout, 새로운 prediction branch는 금지한다. 기존 helper 파일을 삭제하거나 Stage 1/ordinary Stage 2를 재구현하지 않는다.

---

## 현재 상태와 실행 전 준비

- 코드 조사 기준 a69b9c0; 서면 spec commit 9c1ab94.
- 이미 추가한 tests/step_preference/test_context_preference.py는 untracked이며 보존한다.
- 기존 baseline 36 passed: logs/contextpref-implementation-20260906-01/baseline.log.
- 새 테스트 6 failed: logs/contextpref-implementation-20260906-01/red-core.log. 원인은 context_preference_loss 부재와 negative 필수 설정이다.
- 이 RED 파일의 preflight/collator 검사와 trainer 검사는 현재 분리되어 있다. Task 3에서 한 경로로 연결해야 first_nail을 충족한다.
- 위 결과는 기존 코드의 증거이며 이 plan의 완료 증거가 아니다. 아직 production 코드를 수정하지 않았다.

실행 세션에서 한 번 설정한다. 기존 디렉터리가 있으면 다른 run ID를 사용한다.

```bash
cd /root/workspace/ILVR/.worktrees/ours1
export PYTHONPATH="$PWD/transformers/src:$PWD/src"
export ILVR_CONTEXT_RUN_ID=contextpref-20260906-02
mkdir "logs/$ILVR_CONTEXT_RUN_ID" "results/$ILVR_CONTEXT_RUN_ID"
```

테스트는 OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""로 CPU 검증을 명시한다. GPU 검증은 Task 5에서만 선택한 유휴 GPU를 노출한다. Model weights를 저장하는 테스트가 필요하면 임시 디렉터리도 checkpoints 아래에 만든다. 가짜 파일 fixture를 실제 pretrained artifact로 표현하지 않는다.

## File Structure와 경계

| 파일 | 책임 |
| --- | --- |
| src/step_preference/dpo_loss.py | 기존 DPO 함수 유지, context_preference_loss 추가 |
| src/step_preference/sidecar.py | context-only plan flag와 기존 chosen branch를 이용한 planner |
| transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py | context plan에서 각 modality를 분할 없이 한 번 실행 |
| src/step_preference/config.py, src/utils_deepseed.py | 기본 objective 및 명시적 legacy 선택, preference 계수 API |
| src/main.py, src/stage2/data.py, run_stage2_training.sh | negative 없는 기본 입력 경로 |
| src/stage2/trainer.py | 새 objective route, scalar loss/metrics; 기존 legacy reducer 보존 |
| src/step_preference/equivalence.py | 기존 DPO 전용 harness의 objective를 legacy로 명시 |
| tests/step_preference/test_context_preference.py | 보존한 RED 테스트 확장, 새 기본 경로의 통합 검증 |
| tests/step_preference/test_context_sidecars.py | planner/replay/gradient oracle 검증 |
| tests/stage2/test_launcher.py | 실제 shell dry-run argv 검증 |
| tests/stage2/integration/shared_prefix_smoke.py | 기존 작은 DoRA/ZeRO2 smoke를 새 기본 objective로 연결 |
| tests/stage2/integration/test_shared_prefix_distributed.py | smoke 결과의 호출 수/legacy 비호출 검증 |
| README.md | 새 Stage 2B 명령과 호출 범위, legacy 선택 설명 |

새 production subsystem이나 범용 orchestration 파일은 만들지 않는다. 기존 legacy 관련 테스트는 기대값을 새 objective에 억지로 맞추지 않고 legacy 선택을 명시한다.

## Task 1: Context preference scalar loss

**Files:**
- Modify: src/step_preference/dpo_loss.py — dpo_pair_losses 옆에 함수 추가.
- Test: tests/step_preference/test_context_preference.py — 기존 첫 테스트와 경계값 검사.
- Regression: tests/step_preference/test_dpo_loss.py.

**Interfaces:**
- Consumes: 기존 _validate_score_tensors(*scores), _require_positive_finite(name, value).
- Produces: context_preference_loss(clean_logps: torch.Tensor, ablated_logps: torch.Tensor, beta: float, margin: float) -> torch.Tensor. 동일한 비어 있지 않은 shape, floating finite score; 양쪽 gradient를 유지하는 scalar 평균.

- [ ] **Step 1: 실패 테스트를 보존하고 margin/invalid-input 사례를 추가한다.**

```python
@pytest.mark.parametrize("margin", [0.0, 0.4])
def test_context_loss_uses_margin_before_sigmoid(margin):
    clean = torch.tensor([-1., -3.], requires_grad=True)
    off = torch.tensor([-3., -1.], requires_grad=True)
    fn = getattr(dpo_loss, "context_preference_loss", None)
    assert callable(fn), "context_preference_loss is missing"
    got = fn(clean, off, beta=1., margin=margin)
    expected = (math.log1p(math.exp(margin-2.)) +
                math.log1p(math.exp(margin+2.))) / 2
    assert float(got.detach()) == pytest.approx(expected)
    got.backward()
    assert torch.all(clean.grad < 0) and torch.all(off.grad > 0)

@pytest.mark.parametrize("clean,off", [
    (torch.empty(0), torch.empty(0)),
    (torch.zeros(2), torch.zeros(1)),
    (torch.tensor([float("nan")]), torch.zeros(1)),
])
def test_context_loss_rejects_invalid_scores(clean, off):
    fn = getattr(dpo_loss, "context_preference_loss", None)
    assert callable(fn), "context_preference_loss is missing"
    with pytest.raises(ValueError):
        fn(clean, off, beta=0.1, margin=0.0)
```

- [ ] **Step 2: RED를 확인한다.**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_context_preference.py -k 'context_loss or averages_individual' > "logs/$ILVR_CONTEXT_RUN_ID/task1-red.log" 2>&1
```

예상: 함수가 없어 assertion failure. Import typo나 fixture 오류라면 먼저 테스트를 교정한다.

- [ ] **Step 3: 기존 파일에 최소 함수를 추가한다.**

```python
def context_preference_loss(clean_logps, ablated_logps, beta, margin):
    beta = _require_positive_finite("beta", beta)
    _validate_score_tensors(clean_logps, ablated_logps)
    if clean_logps.numel() == 0:
        raise ValueError("context preference requires at least one step")
    if isinstance(margin, bool) or not isinstance(margin, Real) or not math.isfinite(margin):
        raise ValueError("margin must be finite")
    delta = clean_logps - ablated_logps
    return -F.logsigmoid(beta * (delta - float(margin))).mean()
```

빈 step의 zero loss는 trainer가 처리한다. 함수 내부에서 detach나 reference 항을 넣지 않는다.

- [ ] **Step 4: GREEN과 기존 DPO 회귀를 확인한다.**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_context_preference.py tests/step_preference/test_dpo_loss.py -k 'context_loss or averages_individual or dpo' > "logs/$ILVR_CONTEXT_RUN_ID/task1-green.log" 2>&1
```

- [ ] **Step 5: 이 함수와 테스트만 커밋한다.**

```bash
git add src/step_preference/dpo_loss.py tests/step_preference/test_context_preference.py
git commit -m "feat: add reference-free context preference loss"
```

기존 파일에 남은 Stage 2B RED 테스트가 아직 실패한다는 상태를 task handoff에 명시한다. 통합 완료로 보고하지 않는다.

## Task 2: 모든 step의 context-only sidecar를 한 V/T 호출로 실행

**Files:**
- Modify: src/step_preference/sidecar.py — SidecarPlan, planner 추가.
- Modify: transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py — validate_shared_inputs/score_sidecars.
- Create: tests/step_preference/test_context_sidecars.py.
- Regression: tests/step_preference/test_sidecar.py, test_shared_prefix_model.py.

**Interfaces:**
- Consumes: 기존 _validate_clean_inputs, _validate_spans, _chosen_branch, SidecarBranch, _branch_inputs, _score_chunk, mask builders.
- Produces: SidecarPlan.context_preference: bool = False.
- Produces: build_context_sidecar_plan(*, clean_input_ids: torch.Tensor, attention_mask: torch.Tensor, spans: Sequence[ReasoningStepSpan]) -> SidecarPlan.
- 기존 build_sidecar_plan과 legacy plan의 기본값은 그대로 유지한다. 새 planner만 context_preference=True를 만든다.
- Output의 stage2_sidecar_results는 기존 dict 항목 name/kind/step_id/mean_logp/entropies를 유지하며 visual_off/textual_off만 포함한다.

- [ ] **Step 1: planner 부재와 분할 호출을 잡는 테스트를 작성한다.**

```python
import pytest
import torch
import step_preference.sidecar as sidecar
from step_preference.shared_prefix_measurement import RuntimeCounters, tiny_measurement_case

@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_context_group_ignores_legacy_chunk_budget(k):
    model, batch = tiny_measurement_case(steps=k, total_target_tokens=2*k, device="cpu")
    builder = getattr(sidecar, "build_context_sidecar_plan", None)
    assert callable(builder), "context sidecar planner missing"
    plan = builder(clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"], spans=batch["reasoning_step_spans"])
    assert [len(g) for g in plan.groups] == [k, k]
    assert all(b.rejected_model_inputs is None for g in plan.groups for b in g)
    with RuntimeCounters(model) as counters:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            use_cache=False, stage2_shared_prefix_inputs=dict(
                plan=plan, backend="dense", query_budget=1, checkpoint=True))
    calls = counters.snapshot()
    assert calls["decoder_calls_forward"] - calls["latent_decoder_calls_forward"] == 3
    assert calls["sidecar_chunks_forward"] == 2
    assert len(out.stage2_sidecar_results) == 2*k
    for span, branch in zip(batch["reasoning_step_spans"], plan.groups[0]):
        assert branch.logical_positions == tuple(range(span.target_text_start-1, span.target_text_end-1))
        assert branch.target_token_ids == tuple(batch["input_ids"][0, span.target_text_start:span.target_text_end].tolist())
```

- [ ] **Step 2: RED를 실행하고 planner missing을 확인한다.**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_context_sidecars.py > "logs/$ILVR_CONTEXT_RUN_ID/task2-red.log" 2>&1
```

- [ ] **Step 3: 기존 chosen planner를 재사용한다.**

```python
# Add the last field to SidecarPlan:
context_preference: bool = False

def build_context_sidecar_plan(*, clean_input_ids, attention_mask, spans):
    clean_ids, root_length = _validate_clean_inputs(clean_input_ids, attention_mask)
    validated = _validate_spans(spans, root_length) if spans else ()
    positions = tuple(p for s in validated
                      for p in range(s.target_text_start-1, s.target_text_end-1))
    groups = tuple(tuple(_chosen_branch(clean_ids=clean_ids, span=s, modality=m)
                         for s in validated) for m in ("visual", "textual"))
    return SidecarPlan(root_length, positions, groups, context_preference=True)
```

- [ ] **Step 4: context plan에서만 chunk iterator를 우회한다.**

```python
# validate_shared_inputs: after validating the plan type
plan = options["plan"]
if plan.context_preference:
    expected = ({"visual_off"}, {"textual_off"})
    for group, kinds in zip(plan.groups, expected):
        if any(b.kind not in kinds or b.rejected_model_inputs is not None for b in group):
            raise ValueError("context preference permits only local chosen branches")

# score_sidecars: replace only the nested-loop selection
for group in plan.groups:
    batches = ((group,) if group else ()) if plan.context_preference else chunk_sidecar_branches(group, budget)
    for branches in batches:
        # Existing branch embedding, mask, checkpoint and scoring body stays here.
        parts = [_branch_inputs(model, branch, input_ids, clean_embeds, clean_positions)
                 for branch in branches]
```

마지막 code block은 기존 loop body의 삽입 위치를 보여준다. parts 이후 기존 body를 삭제하지 않는다. Query budget 숫자를 크게 바꿔 우연히 3회를 맞추는 구현은 금지한다. 기존 zero-copy prefix slice, logical positions, non-reentrant checkpoint closure를 유지한다.

- [ ] **Step 5: GREEN 및 기존 legacy replay 회귀를 확인하고 커밋한다.**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_context_sidecars.py tests/step_preference/test_sidecar.py tests/step_preference/test_shared_prefix_model.py > "logs/$ILVR_CONTEXT_RUN_ID/task2-green.log" 2>&1
git add src/step_preference/sidecar.py transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py tests/step_preference/test_context_sidecars.py
git commit -m "feat: replay all context sidecars without forward chunking"
```

## Task 3: Negative 없는 기본 Stage 2B vertical slice

**Files:**
- Modify: src/step_preference/config.py — StepPreferenceConfig validation/from_args.
- Modify: src/utils_deepseed.py — parse_args.
- Modify: src/main.py — Stage2BPreflight, preflight_stage2b_artifacts.
- Modify: src/stage2/data.py — Stage2DataCollator.__call__.
- Modify: src/stage2/trainer.py — compute_loss와 새 context reducer/route.
- Modify: run_stage2_training.sh — preference env/argv validation.
- Modify: src/step_preference/equivalence.py — _config와 load_production_context의 legacy 명시.
- Test: tests/step_preference/test_context_preference.py, tests/stage2/test_launcher.py.
- Update legacy fixtures only: tests/step_preference/test_config.py, test_stage2b_trainer.py, test_stage2b_artifacts.py, test_shared_prefix_trainer.py, test_dora_ablation_reporting.py, test_packed_admission_protocol.py, test_packed_capacity_evidence.py, test_packed_capacity_probe.py, test_phase_a_verifier.py; tests/stage2/test_routing.py.

**Interfaces:**
- Consumes: Task 1 context_preference_loss, Task 2 build_context_sidecar_plan.
- Produces: StepPreferenceConfig.objective: Literal["context_preference", "legacy_dpo"] = "context_preference"; lambda_preference: float = 1.0; preference_margin: float = 0.0; is_context_preference property = is_stage2b and objective == "context_preference".
- Produces CLI: --step_preference_objective, --step_preference_beta (alias of existing --step_dpo_beta using dest=step_dpo_beta), --step_preference_weight, --step_preference_margin.
- Produces env: STEP_PREFERENCE_OBJECTIVE, STEP_PREFERENCE_BETA, STEP_PREFERENCE_WEIGHT, STEP_PREFERENCE_MARGIN.
- Produces: Stage2BPreflight.negative_store: Mapping[tuple[str,int], object] | None. None means no join; do not return an empty mapping, which would enter the legacy join branch.
- Produces: Stage2Trainer._compute_context_preference_loss(model, inputs, *, collect_diagnostics: bool, return_outputs: bool), returning scalar loss or (loss, outputs) with existing return_outputs behavior.
- Produces: Stage2Trainer._reduce_context_objective(clean_ce, clean_scores, clean_entropies, results) -> tuple[torch.Tensor, dict]. clean_scores/clean_entropies are lists in step order; results maps (step_id, "visual"|"textual") to existing sidecar result dict.
- The new default does not depend on --step_shared_prefix_enabled. Explicit legacy_dpo alone retains old routing.

이 task의 deliverable은 실행 가능한 기본 경로 전체다. 설정만 바꾸고 trainer가 legacy에 남은 상태를 완료로 커밋하지 않는다.

- [ ] **Step 1: 현재 통합 RED를 재현하고 launcher/no-legacy route 검사를 추가한다.**

기존 test_default_actual_scoring_calls_are_three_without_legacy에서 추가로 기존 _compute_stage2b_loss 자체를 forbidden으로 patch한다. 새 route가 legacy metadata validation 전에 분기하는지 잡는다.

```python
# Add to tests/stage2/test_launcher.py
def test_context_default_launch_needs_no_negatives():
    result = run_launcher(env_overrides={
        "STEP_PREFERENCE_ENABLED": "true",
        "STEP_PREFERENCE_MODE": "stage2b",
        "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
        "STEP_PREFERENCE_MARGIN": "0",
        "STEP_PREFERENCE_WEIGHT": "1",
        "STEP_SIDECAR_ATTENTION_BACKEND": "dense",
    })
    tokens = dry_run_tokens(result)
    assert option_value(tokens, "--step_preference_objective") == "context_preference"
    assert option_value(tokens, "--step_preference_weight") == "1"
    assert option_value(tokens, "--step_sidecar_attention_backend") == "dense"
    assert "--step_negative_path" not in tokens
    assert "--step_negative_manifest" not in tokens
```

기존 preflight/collator 테스트의 batch 검증 뒤 다음을 추가해 first_nail을 한 실행 경로로 연결한다. 이 controlled processor는 테스트 tokenizer를 사용한다. 실제 pretrained processor 정확성을 주장하지 않는다.

```python
# Inside test_default_preflight_and_collator_need_no_negative_artifacts:
from step_preference.equivalence import _tiny_model
config = copy.deepcopy(_tiny_model().config)
config.vocab_size = 512
config.latent_size = 8
config.image_token_id = 70
config.num_hidden_layers = 2
config.vision_config.window_size = 8
config.vision_config.spatial_merge_size = 2
model = type(_tiny_model())(config).float().train()
model.visual.requires_grad_(False)
# The existing test processor emits an image placeholder grid, not Qwen patches.
# Supply matching tiny vision patches; keep the real tiny Qwen vision module.
patch_width = (config.vision_config.in_channels *
               config.vision_config.temporal_patch_size *
               config.vision_config.patch_size ** 2)
batch["pixel_values"] = torch.zeros(16, patch_width)
batch["position_ids"] = torch.arange(batch["input_ids"].shape[1]).view(1,1,-1).expand(3,1,-1)
trainer = _trainer(replace(args.step_preference_config, sidecar_attention_backend="dense"))
monkeypatch.setattr(model, "generate", forbidden)
monkeypatch.setattr(trainer, "_compute_stage2b_loss", forbidden)
monkeypatch.setattr("stage2.trainer.aggregate_step_dpo", forbidden)
monkeypatch.setattr("step_preference.sidecar._rejected_branch", forbidden)
with RuntimeCounters(model) as counters:
    loss = trainer.compute_loss(model, batch)
    before_backward = counters.snapshot()
    counters.phase = "backward"
    loss.backward()
assert torch.isfinite(loss)
assert before_backward["decoder_calls_forward"] - before_backward["latent_decoder_calls_forward"] == 3
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
```

- [ ] **Step 2: RED 로그를 확인한다.**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_context_preference.py tests/stage2/test_launcher.py -k 'default or context' > "logs/$ILVR_CONTEXT_RUN_ID/task3-red.log" 2>&1
```

예상: negative 필수 설정/launcher error. Tiny vision fixture가 shape 오류를 내면 실제 config의 patch dimensions로 fixture만 교정하며 production validation을 약화하지 않는다.

- [ ] **Step 3: 명시적인 objective와 새 scalar 설정을 추가한다.**

```python
# Fields on StepPreferenceConfig:
objective: Literal["context_preference", "legacy_dpo"] = "context_preference"
lambda_preference: float = 1.0
preference_margin: float = 0.0

@property
def is_context_preference(self):
    return self.is_stage2b and self.objective == "context_preference"
```

기존 __post_init__에 다음 검증을 추가하고 negative 필수 요구는 legacy에만 남긴다.

```python
if self.objective not in ("context_preference", "legacy_dpo"):
    raise ValueError("unknown step preference objective")
if _finite("step_preference_weight", self.lambda_preference) < 0:
    raise ValueError("step_preference_weight must be nonnegative")
_finite("step_preference_margin", self.preference_margin)
if self.is_context_preference:
    if self.lambda_ce != 1.0:
        raise ValueError("context preference requires ce_weight=1")
    if self.negative_path is not None or self.negative_manifest is not None:
        raise ValueError("context preference does not consume negative artifacts; select legacy_dpo explicitly")
    if self.packing_enabled or any((self.packed_admission_manifest,
            self.packed_admission_manifest_sha256, self.packed_admission_verification,
            self.packed_admission_verification_sha256)):
        raise ValueError("context preference does not consume legacy packed admission")

# Replace the Stage 2B required-field tuple, preserving existing attention,
# checkpoint initialization, budget split and resume validation below it.
required = [("stage2a_checkpoint_path", "--stage2a_checkpoint_path")]
if not self.is_context_preference:
    required += [("negative_path", "--step_negative_path"),
                 ("negative_manifest", "--step_negative_manifest")]
for field_name, option in required:
    if getattr(self, field_name) is None:
        raise ValueError(f"Stage 2B requires {option}")
```

Parser에 다음을 추가하고 from_args에 대응 필드를 전달한다:

```python
parser.add_argument("--step_preference_objective", choices=("context_preference", "legacy_dpo"),
                    default="context_preference")
parser.add_argument("--step_preference_weight", type=float, default=1.0)
parser.add_argument("--step_preference_margin", type=float, default=0.0)
# Replace the existing --step_dpo_beta declaration:
parser.add_argument("--step_preference_beta", "--step_dpo_beta",
                    dest="step_dpo_beta", type=float, default=0.1)

# Additional keywords passed by StepPreferenceConfig.from_args:
objective=getattr(args, "step_preference_objective", "context_preference"),
lambda_preference=getattr(args, "step_preference_weight", 1.0),
preference_margin=getattr(args, "step_preference_margin", 0.0),
```

- [ ] **Step 4: 기존 preflight와 collator를 negative 없는 경로로 연결한다.**

Stage2BPreflight의 negative_store 타입에 None을 허용한다. preflight의 기존 manifest 읽기/비교 블록은 legacy 분기로 옮긴다. Source 로드, prepare_stage2_example, duplicate ID, validate_stage2b_peft_base_identity를 유지한다. 해당 base identity 검증 뒤에 다음 분기를 놓는다. inspect_stage2a_checkpoint_identity는 기존 checkpoint inventory 검사이며 reference 점수 계산이 아니다.

```python
checkpoint_files, tokenizer_identity = inspect_stage2a_checkpoint_identity(
    config.stage2a_checkpoint_path)
if config.is_context_preference:
    return Stage2BPreflight(prepared_examples=prepared, negative_store=None)
manifest = read_negative_manifest(config.negative_manifest)
if manifest.source_sha256 != source_sha256:
    raise ValueError("source JSONL SHA-256 does not match negative manifest")
# Existing generation-settings identity, load_negative_store and admission path follow.
```

main의 build_stage2_training_components는 기존처럼 preflight.negative_store를 전달한다. None을 {}로 변환하지 않는다. Collator의 step_preference metadata 생성 시 steps를 추가하고, 기존 negative_store is not None 분기는 legacy에서만 원래대로 덮어쓰게 한다.

```python
batch["step_preference"] = {
    "expanded_rendered_text": expanded_text,
    "offset_mapping": offset_mapping,
    "steps": tuple({"span": span} for span in reasoning_step_spans),
}
```

- [ ] **Step 5: 독립 context reducer를 추가하고 기존 metric accumulator를 재사용한다.**

```python
def _reduce_context_objective(self, clean_ce, clean_scores, clean_entropies, results):
    config = self.step_preference_config
    zero = clean_ce * 0.0
    pref, ent, margins, off_logps, off_entropies = {}, {}, {}, {}, {}
    for modality in ("visual", "textual"):
        if clean_scores:
            clean = torch.stack(clean_scores)
            off = torch.stack([results[(k, modality)]["mean_logp"]
                               for k in range(len(clean_scores))])
            pref[modality] = context_preference_loss(
                clean, off, config.beta, config.preference_margin)
            hs = [results[(k, modality)]["entropies"] for k in range(len(clean_scores))]
            if any(a.shape != b.shape for a, b in zip(hs, clean_entropies)):
                raise ValueError("sidecar entropy rows must match clean target rows")
            ent[modality] = torch.stack([
                F.relu(config.entropy_margin - (off_h - clean_h)).mean()
                for clean_h, off_h in zip(clean_entropies, hs)]).mean()
            margins[modality] = (clean-off).mean()
            off_logps[modality] = off.mean()
            off_entropies[modality] = torch.stack([h.mean() for h in hs]).mean()
        else:
            pref[modality] = ent[modality] = margins[modality] = zero
            off_logps[modality] = off_entropies[modality] = zero
    preference = (pref["visual"] + pref["textual"]) / 2
    entropy = (ent["visual"] + ent["textual"]) / 2
    total = clean_ce + config.lambda_preference * preference + config.lambda_entropy * entropy
    metrics = {
        "loss/total": total, "loss/ce": clean_ce,
        "loss/context_preference": preference, "loss/entropy": entropy,
        "stats/logp_clean": torch.stack(clean_scores).mean() if clean_scores else zero,
        "stats/entropy_clean": torch.stack([h.mean() for h in clean_entropies]).mean() if clean_entropies else zero,
    }
    for modality, label in (("visual", "visual"), ("textual", "text")):
        metrics[f"loss/context_preference_{label}"] = pref[modality]
        metrics[f"loss/entropy_{label}"] = ent[modality]
        metrics[f"stats/context_margin_{label}"] = margins[modality]
        metrics[f"stats/logp_{label}_off"] = off_logps[modality]
        metrics[f"stats/entropy_{label}_off"] = off_entropies[modality]
    if not bool(torch.isfinite(total)):
        raise ValueError("context preference total loss must be finite")
    return total, metrics
```

Import context_preference_loss from step_preference.dpo_loss. _accumulate_step_preference_metrics의 required metric 집합을 objective별로 선택한다. Context 필수 집합은 위 metrics 키들과 아래 route가 추가하는 stats/scored_tokens, stats/logical_scoring_groups다. Scalar/finite 검사, 누적 및 Trainer.log 동작은 그대로 유지한다. Logical group 값을 실제 hook 계측값이라고 표기하지 않는다.

- [ ] **Step 6: 모델을 한 outer call로 호출하는 새 trainer route를 추가한다.**

```python
def _compute_context_preference_loss(self, model, inputs, *, collect_diagnostics, return_outputs):
    config = self.step_preference_config
    clean_inputs = dict(inputs)
    sample_ids = clean_inputs.pop("sample_ids", None)
    spans = clean_inputs.pop("reasoning_step_spans", None)
    metadata = clean_inputs.pop("step_preference", None)
    if not isinstance(sample_ids, (tuple, list)) or len(sample_ids) != 1 or not isinstance(sample_ids[0], str) or not sample_ids[0]:
        raise ValueError("context preference requires one sample_id")
    if not isinstance(spans, (tuple, list)) or not isinstance(metadata, dict):
        raise ValueError("context preference requires reasoning step metadata")
    steps = metadata.get("steps")
    if not isinstance(steps, (tuple, list)) or len(steps) != len(spans):
        raise ValueError("context preference step cardinality mismatch")
    if any(not isinstance(item, dict) or set(item) != {"span"} or item["span"] != span
           for item, span in zip(steps, spans)):
        raise ValueError("context preference requires clean span-only metadata")
    plan = build_context_sidecar_plan(clean_input_ids=clean_inputs["input_ids"],
        attention_mask=clean_inputs["attention_mask"], spans=spans)
    call = dict(clean_inputs, use_cache=False, stage2_loss_only=not return_outputs,
        collect_stage2_diagnostics=collect_diagnostics,
        stage2_shared_prefix_inputs=dict(plan=plan, backend=config.sidecar_attention_backend,
                                        query_budget=config.sidecar_query_budget))
    if plan.root_selected_positions:
        call["stage2_logit_positions"] = torch.tensor(plan.root_selected_positions,
            device=clean_inputs["input_ids"].device, dtype=torch.long)
    outputs = model(**call)
    clean_ce = outputs.loss
    if not isinstance(clean_ce, torch.Tensor) or clean_ce.numel() != 1 or not bool(torch.isfinite(clean_ce)):
        raise ValueError("clean CE must be a finite scalar")
    diagnostics = getattr(outputs, "stage2_diagnostics", None)
    self._latest_stage2_diagnostics = None if diagnostics is None else {
        key: value.detach() for key, value in diagnostics.items()}
    rows = _selected_logits(outputs, len(plan.root_selected_positions)) if spans else None
    results = {}
    for result in outputs.stage2_sidecar_results:
        modality = {"visual_off": "visual", "textual_off": "textual"}.get(result["kind"])
        key = (result["step_id"], modality)
        if modality is None or key in results:
            raise ValueError("unexpected or duplicate context sidecar")
        results[key] = result
    if set(results) != {(s.step_id, m) for s in spans for m in ("visual", "textual")}:
        raise ValueError("context sidecar layout mismatch")
    clean_scores, clean_entropies, offset = [], [], 0
    for span in spans:
        count = span.target_text_end - span.target_text_start
        logits = rows[:, offset:offset+count]
        targets = clean_inputs["input_ids"][:, span.target_text_start:span.target_text_end]
        clean_scores.append(_mean_selected_logp(logits, targets))
        clean_entropies.append(_selected_entropies(logits))
        offset += count
    total, metrics = self._reduce_context_objective(clean_ce, clean_scores, clean_entropies, results)
    metrics.update({"stats/scored_tokens": 3*len(plan.root_selected_positions),
                    "stats/logical_scoring_groups": 1 + sum(bool(g) for g in plan.groups)})
    self._accumulate_step_preference_metrics(metrics)
    return (total, outputs) if return_outputs else total
```

build_context_sidecar_plan를 import한다. 기존 compute_loss의 helper-image tensor guard와 microbatch index 증가 뒤, 기존 is_stage2b 분기 앞에 다음을 삽입한다. Legacy _compute_stage2b_loss와 기존 DPO reducer는 삭제하지 않는다.

```python
if isinstance(config, StepPreferenceConfig) and config.is_context_preference:
    return self._compute_context_preference_loss(model, inputs,
        collect_diagnostics=collect_diagnostics, return_outputs=return_outputs)
```

- [ ] **Step 7: launcher의 새 env/argv를 연결하고 legacy 자동 진입을 막는다.**

기존 preference parsing 블록에서 objective를 아래처럼 검증한다. STEP_PREFERENCE_ENABLED=false일 때 기존 Stage 2 argv를 바꾸지 않는다.

```bash
STEP_PREFERENCE_OBJECTIVE=${STEP_PREFERENCE_OBJECTIVE:-context_preference}
STEP_PREFERENCE_BETA=${STEP_PREFERENCE_BETA:-${STEP_DPO_BETA:-0.1}}
STEP_PREFERENCE_WEIGHT=${STEP_PREFERENCE_WEIGHT:-1.0}
STEP_PREFERENCE_MARGIN=${STEP_PREFERENCE_MARGIN:-0.0}
case "$STEP_PREFERENCE_OBJECTIVE" in
  context_preference|legacy_dpo) ;;
  *) printf 'Unknown STEP_PREFERENCE_OBJECTIVE\n' >&2; exit 2 ;;
esac
STEP_PREFERENCE_ARGS+=(--step_preference_objective "$STEP_PREFERENCE_OBJECTIVE")
if [[ "$STEP_PREFERENCE_OBJECTIVE" == context_preference ]]; then
  STEP_PREFERENCE_ARGS+=(--step_preference_beta "$STEP_PREFERENCE_BETA"
    --step_preference_weight "$STEP_PREFERENCE_WEIGHT"
    --step_preference_margin "$STEP_PREFERENCE_MARGIN"
    --step_sidecar_attention_backend "$STEP_SIDECAR_ATTENTION_BACKEND"
    --step_sidecar_query_budget "$STEP_SIDECAR_QUERY_BUDGET")
else
  STEP_PREFERENCE_ARGS+=(--step_dpo_beta "$STEP_DPO_BETA" --step_dpo_weight "$STEP_DPO_WEIGHT")
fi
```

기존 unconditional --step_dpo_beta/weight append를 제거해 중복 argv를 만들지 않는다. Context에서 negative env가 존재하거나 legacy packing/admission이 켜지면 Python config와 같은 이유로 즉시 오류를 낸다. Stage 2B checkpoint 필수 검사와 attention/vision backend 검사는 그대로 유지한다. Negative path 필수 검사와 argv append는 objective=legacy_dpo 안으로 옮긴다. 새 플래그를 launcher의 managed-option 목록에 추가하고, 숫자 유효성 검사는 기존 venv 기반 validator를 재사용한다. Margin은 finite real, beta는 positive finite, weight는 nonnegative finite, context CE는 정확히 1이다.

- [ ] **Step 8: legacy fixture/harness의 의도를 명시한다.**

equivalence._config의 함수는 기존 인자에 objective만 추가한다:

```python
def _config(*, packing: bool) -> StepPreferenceConfig:
    return StepPreferenceConfig(
        enabled=True, mode="stage2b", objective="legacy_dpo",
        negative_path="tiny-negatives.jsonl",
        negative_manifest="tiny-manifest.json",
        stage2a_checkpoint_path="tiny-stage2a",
        attention_implementation="eager", packing_enabled=packing,
        max_packed_sequence_length=4608 if packing else None,
    )
```

기존 legacy CLI 배열은 parse_args에 전달하기 전에 다음과 같이 명시한다. 직접 list literal을 parse_args에 넘기는 곳은 아래 두 token을 그 literal 앞에 삽입한다.

```python
argv = ["--step_preference_objective", "legacy_dpo", *argv]
env_overrides["STEP_PREFERENCE_OBJECTIVE"] = "legacy_dpo"
```

정확한 추가 점검 대상: tests/stage2/test_routing.py,
tests/step_preference/test_dora_ablation_reporting.py,
test_packed_admission_protocol.py, test_packed_capacity_evidence.py,
test_packed_capacity_probe.py, test_phase_a_verifier.py.
이 파일들의 **legacy Stage 2B** constructor/CLI/env에만 같은 objective 선택을 추가한다.
Schema serialization을 검사하는 DPO reporting fixture도 legacy 의미를 유지한다.
대상은 위 파일과 Task 3 Files에 명시한 fixture, equivalence.load_production_context argv다. 기본/disabled 경로 테스트에는 legacy를 강제로 주입하지 않는다. 새 기본 테스트를 legacy로 바꿔 GREEN을 만드는 행위는 금지한다.

- [ ] **Step 9: first_nail과 focused regression을 실행하고 커밋한다.**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q -s tests/step_preference/test_context_preference.py tests/step_preference/test_config.py tests/step_preference/test_stage2b_artifacts.py tests/step_preference/test_stage2b_trainer.py tests/step_preference/test_shared_prefix_trainer.py tests/stage2/test_launcher.py > "logs/$ILVR_CONTEXT_RUN_ID/task3-green.log" 2>&1
git add src/step_preference/config.py src/utils_deepseed.py src/main.py src/stage2/data.py src/stage2/trainer.py run_stage2_training.sh src/step_preference/equivalence.py tests/step_preference/test_context_preference.py tests/step_preference/test_config.py tests/step_preference/test_stage2b_artifacts.py tests/step_preference/test_stage2b_trainer.py tests/step_preference/test_shared_prefix_trainer.py tests/stage2/test_launcher.py
git add tests/stage2/test_routing.py tests/step_preference/test_dora_ablation_reporting.py tests/step_preference/test_packed_admission_protocol.py tests/step_preference/test_packed_capacity_evidence.py tests/step_preference/test_packed_capacity_probe.py tests/step_preference/test_phase_a_verifier.py
git commit -m "fix: make Stage 2B default to three-forward context preference"
```

First_nail 통과를 즉시 보고한다. 이 결과만으로 수치/gradient 동등성까지 주장하지 않는다.

## Task 4: 독립 oracle로 loss·gradient·첫 token·locality 검증

**Files:**
- Modify: tests/step_preference/test_context_sidecars.py — test-only unequal-length fixture/oracle.
- Modify: tests/step_preference/test_context_preference.py — reduction, empty steps, disabled baseline.
- Production fix only if a new causal regression fails: Task 1–3의 담당 파일. 실패 없이 refactor하지 않는다.

**Interfaces:**
- Consumes: Task 2 build_context_sidecar_plan, Task 3 context_config/_trainer and new trainer route.
- Produces test-only two_step_case() -> (model, batch) and literal_context_oracle(model, batch) -> (loss, clean_outputs, sidecar_scores).
- Oracle는 production planner/mask/reducer를 정답 생성에 사용하지 않는다. 검증용 독립 full-prefix forwards는 training microbatch 상한 측정에 합산하지 않는다.
- Numerical scope: two-layer FP32, dropout=0, atol=1e-5/rtol=1e-4, checkpointing off/on. 실제 trained/pretrained 품질 주장은 하지 않는다.

- [ ] **Step 1: 서로 다른 target 길이 fixture와 수동 oracle를 작성한다.**

```python
import copy
from dataclasses import replace
import torch.nn.functional as F

def two_step_case():
    model, batch = tiny_measurement_case(steps=2, total_target_tokens=4, device="cpu")
    ids = torch.cat((batch["input_ids"], torch.tensor([[17,18,19]])), dim=1)
    spans = batch["reasoning_step_spans"]
    spans = (spans[0], replace(spans[1], target_text_end=ids.shape[1]))
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
        sample_ids=("context-two-step",), reasoning_step_spans=spans,
        step_preference={"steps": tuple({"span": s} for s in spans)})
    # Distinct temporal/height/width coordinates catch accidental 1D position reuse.
    positions = torch.arange(ids.shape[1])[None,None,:]
    batch["position_ids"] = torch.cat((positions, 2*positions, 3*positions), dim=0)
    return model, batch

def literal_context_oracle(model, batch):
    ids = batch["input_ids"]
    kwargs = {key: batch[key] for key in ("input_ids", "attention_mask", "labels", "position_ids")}
    clean = model(**kwargs, use_cache=False)
    losses, entropy_losses, scores = [], [], {}
    def scored(logits, targets):
        lp = logits.float().log_softmax(-1)
        return lp.gather(-1, targets[:,None]).mean(), -(lp.exp()*lp).sum(-1)
    for span in batch["reasoning_step_spans"]:
        a, b = span.target_text_start, span.target_text_end
        clean_lp, clean_h = scored(clean.logits[0,a-1:b-1], ids[0,a:b])
        for modality, block in (("visual",(span.latent_start,span.latent_end)),
                                ("textual",(span.prev_text_start,span.prev_text_end))):
            n = ids.shape[1]
            visible = torch.ones(n,n,dtype=torch.bool,device=ids.device).tril()
            visible[a-1:b-1,block[0]:block[1]] = False
            mask = torch.zeros(n,n,device=ids.device).masked_fill(
                ~visible, torch.finfo(torch.float32).min)[None,None]
            out = model(**kwargs, stage2_final_attention_mask=mask, use_cache=False)
            off_lp, off_h = scored(out.logits[0,a-1:b-1], ids[0,a:b])
            losses.append(-F.logsigmoid(0.1*(clean_lp-off_lp)))
            entropy_losses.append(F.relu(0.1-(off_h-clean_h)).mean())
            scores[(span.step_id,modality)] = (off_lp,off_h,out.logits[0,a-1:b-1])
    return clean.loss + torch.stack(losses).mean() + 0.1*torch.stack(entropy_losses).mean(), clean, scores
```

- [ ] **Step 2: checkpoint와 gradient 비교 테스트를 작성하고 실행한다.**

```python
from tests.step_preference.test_context_preference import context_config
from step_preference.equivalence import _trainer

@pytest.mark.parametrize("checkpoint", [False, True])
def test_context_matches_literal_loss_and_all_trainable_gradients(checkpoint):
    model, batch = two_step_case()
    reference = copy.deepcopy(model)
    if checkpoint:
        for current in (model, reference):
            current.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = _trainer(context_config())
    actual, out = trainer.compute_loss(model, batch, return_outputs=True)
    expected, clean, scores = literal_context_oracle(reference, batch)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out.logits, clean.logits, atol=1e-5, rtol=1e-4)
    for result in out.stage2_sidecar_results:
        mode = {"visual_off":"visual","textual_off":"textual"}[result["kind"]]
        lp, entropy, _ = scores[(result["step_id"],mode)]
        torch.testing.assert_close(result["mean_logp"], lp, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(result["entropies"], entropy, atol=1e-5, rtol=1e-4)
    actual.backward()
    expected.backward()
    for (name,p), (other_name,q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4, msg=name)
```

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_context_sidecars.py -k literal > "logs/$ILVR_CONTEXT_RUN_ID/task4-oracle.log" 2>&1
```

이 시점에 통과하면 독립 검증 증거로 보존하고 불필요한 production 변경은 하지 않는다. 실패하면 test failure를 해당 구현 결함의 RED로 사용하고 root cause에 맞는 최소 수정만 한다.

- [ ] **Step 3: 토큰 가중 평균과 빈 step 동작을 구별한다.**

```python
def test_entropy_reduces_steps_equally_even_with_different_lengths():
    trainer = _trainer(context_config())
    ce = torch.tensor(2., requires_grad=True)
    clean_scores = [torch.tensor(-2.), torch.tensor(-2.)]
    clean_h = [torch.zeros(1), torch.zeros(3)]
    results = {(k,m): {"mean_logp": torch.tensor(-2.),
        "entropies": torch.zeros(1) if k == 0 else torch.full((3,),2.)}
        for k in (0,1) for m in ("visual","textual")}
    total, metrics = trainer._reduce_context_objective(ce, clean_scores, clean_h, results)
    assert float(metrics["loss/entropy"]) == pytest.approx(0.05)
    assert float(total.detach()) == pytest.approx(2 + math.log(2) + 0.1*0.05)

def test_zero_steps_execute_only_clean_and_report_zero_step_metrics():
    model, batch = tiny_measurement_case(steps=1, total_target_tokens=2, device="cpu")
    ids = torch.tensor([[11,12,13]])
    batch.update(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                 reasoning_step_spans=(), step_preference={"steps": ()})
    trainer = _trainer(context_config())
    with RuntimeCounters(model) as counters:
        loss = trainer.compute_loss(model,batch)
    counts = counters.snapshot()
    assert counts["decoder_calls_forward"] == 1
    assert counts.get("latent_decoder_calls_forward",0) == 0
    assert counts.get("sidecar_chunks_forward",0) == 0
    metrics = trainer._step_preference_metric_totals
    assert metrics["loss/context_preference"] == metrics["loss/entropy"] == 0
    assert metrics["stats/scored_tokens"] == 0
    assert torch.isfinite(loss)

def test_disabled_preference_matches_direct_stage2_loss_and_gradients():
    model,batch = tiny_measurement_case(steps=1,total_target_tokens=2,device="cpu")
    reference = copy.deepcopy(model)
    inputs = {k:batch[k] for k in ("input_ids","attention_mask","labels")}
    loss = _trainer(StepPreferenceConfig()).compute_loss(model,inputs)
    baseline = reference(**inputs,stage2_loss_only=True,collect_stage2_diagnostics=False).loss
    torch.testing.assert_close(loss,baseline,atol=1e-5,rtol=1e-4)
    loss.backward()
    baseline.backward()
    for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4,msg=name)
```

- [ ] **Step 4: 독립 visibility, KV identity, first-token/no-propagation witness를 작성한다.**

Task 4 fixture의 literal 좌표는 root_length=16, step 0 query=[4,6), step 1 query=[10,15)다. V group은 7 query rows, clean ancestor cutoffs는 4와 10이다. Tensor visibility 검사는 아래 수동 기대값을 사용한다.

```python
def test_context_visibility_has_only_own_ancestors_and_branch_past():
    model,batch = two_step_case()
    plan = sidecar.build_context_sidecar_plan(clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],spans=batch["reasoning_step_spans"])
    for group,blocks in zip(plan.groups, [((2,4),(8,10)),((0,1),(5,7))]):
        expected = torch.zeros(7,23,dtype=torch.bool)
        for lo,hi,ancestor,blocked in [(0,2,4,blocks[0]),(2,7,10,blocks[1])]:
            for row in range(lo,hi):
                expected[row,:ancestor] = True
                expected[row,blocked[0]:blocked[1]] = False
                expected[row,16+lo:16+row+1] = True
        actual = sidecar.sidecar_dense_visibility(16,group,device="cpu")
        assert torch.equal(actual,expected)
        mask = sidecar.build_sidecar_flex_block_mask(16,group,"cpu")
        probe = mask.mask_mod(torch.tensor(0),torch.tensor(0),
                             torch.arange(7)[:,None],torch.arange(23)[None,:])
        assert torch.equal(probe,expected)
```

실행 중 KV/position identity는 model.model에 hook을 설치해 확인한다. Clean capture 결과와 V/T 호출에 전달된 bank가 같은 값을 가지며 두 group이 같은 bank storage를 사용하는지 검사한다.

```python
def test_runtime_prefix_bank_identity_and_mrope_reuse():
    model,batch = two_step_case()
    clean_banks,side_banks,side_positions = [],[],[]
    def observe(module,args,kwargs,out):
        if kwargs.get("stage2_capture_kv"):
            clean_banks.append(out.stage2_key_values)
        if kwargs.get("stage2_prefix_key_values") is not None:
            side_banks.append(kwargs["stage2_prefix_key_values"])
            side_positions.append(kwargs["position_ids"])
    handle = model.model.register_forward_hook(observe,with_kwargs=True)
    try:
        _trainer(context_config()).compute_loss(model,batch)
    finally:
        handle.remove()
    assert len(clean_banks) == 1 and len(side_banks) == 2
    for layer,clean_pair in enumerate(clean_banks[0]):
        for i,tensor in enumerate(clean_pair):
            for bank in side_banks:
                torch.testing.assert_close(bank[layer][i],tensor)
                assert bank[layer][i].requires_grad
            assert side_banks[0][layer][i].data_ptr() == side_banks[1][layer][i].data_ptr()
    positions = batch["position_ids"][:,:,[4,5,10,11,12,13,14]]
    for actual in side_positions:
        assert torch.equal(actual,positions)
```

First-token perturbation과 no propagation은 step 0 V 차단만 해제한 두 실행을 비교한다. Dropout은 0으로 유지한다.

```python
def test_first_prediction_changes_without_propagating_to_clean_or_other_sidecars():
    model,batch = two_step_case()
    plan = sidecar.build_context_sidecar_plan(clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],spans=batch["reasoning_step_spans"])
    first = replace(plan.groups[0][0],blocked_start=0,blocked_end=0)
    altered = replace(plan,groups=((first,*plan.groups[0][1:]),plan.groups[1]))
    outputs,projected = [],[]
    for current_plan in (plan,altered):
        heads = []
        handle = model.lm_head.register_forward_hook(lambda m,a,out: heads.append(out))
        try:
            outputs.append(model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],use_cache=False,
                stage2_shared_prefix_inputs=dict(plan=current_plan,backend="dense",checkpoint=False)))
        finally:
            handle.remove()
        projected.append(heads)
    # Existing model order scores V and T heads before the clean output head.
    assert torch.linalg.vector_norm(projected[0][0][0,0]-projected[1][0][0,0]) > 1e-8
    torch.testing.assert_close(outputs[0].logits,outputs[1].logits)
    left = {r["name"]:r for r in outputs[0].stage2_sidecar_results}
    right = {r["name"]:r for r in outputs[1].stage2_sidecar_results}
    for name in left:
        if name != "step-0-visual-off":
            torch.testing.assert_close(left[name]["mean_logp"],right[name]["mean_logp"])
            torch.testing.assert_close(left[name]["entropies"],right[name]["entropies"])
```

LM-head call order는 source comment만으로 가정하지 않는다. Hook에서 group의 총 target rows와 head rows를 함께 검사하고, Task 2의 return_outputs 경로에서 관찰한 순서를 기록한다. First token의 V/T 양쪽 검증을 위해 위 테스트를 modality별로 parameterize하고 text 변경 시 group index=1, 제외 name=step-0-textual-off로 한다. 두 번째 group의 rows offset은 독립 group이므로 첫 row는 동일하게 0이다.

- [ ] **Step 5: 모든 새 CPU witness와 최소 baseline을 실행하고 커밋한다.**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q -s tests/step_preference/test_context_preference.py tests/step_preference/test_context_sidecars.py tests/stage2/test_stage2_modeling.py tests/step_preference/test_ablation.py > "logs/$ILVR_CONTEXT_RUN_ID/task4-green.log" 2>&1
git add tests/step_preference/test_context_preference.py tests/step_preference/test_context_sidecars.py
git commit -m "test: verify context preference locality and shared-prefix gradients"
```

Production causal fix가 있었으면 해당 파일만 명시적으로 추가하고 커밋 메시지를 fix 범위에 맞춘다. Full suite를 이 단계에서 반복하지 않는다.

## Task 5: 작은 GPU/PEFT 검증, 문서화와 최종 handoff

**Files:**
- Modify: tests/step_preference/test_context_sidecars.py — opt-in actual CUDA Flex parity.
- Modify: tests/stage2/integration/shared_prefix_smoke.py — context objective, span-only batch, per-microbatch counters.
- Modify: tests/stage2/integration/test_shared_prefix_distributed.py — result assertions.
- Modify: README.md — new CLI, scientific/counter scope.
- Write results only: results/$ILVR_CONTEXT_RUN_ID/summary.md and runtime records.
- No production model/optimizer artifacts outside checkpoints.

**Interfaces:**
- Consumes: Task 3 default route/config, Task 4 two_step_case, existing RuntimeCounters and existing DoRA/ZeRO2 smoke.
- Produces: per-microbatch observations with policy_calls_forward, decoder_calls_forward, latent_decoder_calls_forward, scoring_decoder_calls_forward, separately labeled backward recomputation and observed forbidden-call counts.
- Reuses: RUN_SHARED_PREFIX_DDP=1 and SHARED_PREFIX_RUN_ID from the existing smoke. Add RUN_CONTEXT_CUDA=1 only as a test opt-in; it is not a new training option.
- Reporting must distinguish actual hooks from logical scoring groups and passed tests from skipped tests.

- [ ] **Step 1: 새 CUDA parity 테스트를 추가한다.**

```python
import os

@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_CONTEXT_CUDA") != "1" or not torch.cuda.is_available(),
                    reason="explicit context CUDA check")
def test_context_cuda_flex_matches_dense_scores_gradients_and_calls():
    base,batch = two_step_case()
    batch = {k:(v.to("cuda:0") if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
    base = base.to("cuda:0")
    records = []
    for backend in ("dense","flex"):
        model = copy.deepcopy(base)
        config = replace(context_config(),sidecar_attention_backend=backend)
        trainer = _trainer(config)
        with RuntimeCounters(model) as counters:
            loss,out = trainer.compute_loss(model,batch,return_outputs=True)
            before = counters.snapshot()
            counters.phase = "backward"
            loss.backward()
            counts = counters.snapshot()
        assert before["decoder_calls_forward"] - before["latent_decoder_calls_forward"] == 3
        records.append((loss.detach(),out.stage2_sidecar_results,
            {n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad},counts))
    torch.testing.assert_close(records[0][0],records[1][0],atol=1e-5,rtol=1e-4)
    for left,right in zip(records[0][1],records[1][1]):
        assert left["name"] == right["name"]
        for key in ("mean_logp","entropies"):
            torch.testing.assert_close(left[key],right[key],atol=1e-5,rtol=1e-4)
    for name in records[0][2]:
        torch.testing.assert_close(records[0][2][name],records[1][2][name],
                                   atol=1e-5,rtol=1e-4,msg=name)
    print({"context_cuda_dense":records[0][3],"context_cuda_flex":records[1][3]})
```

실제 Flex kernel 사용 여부는 기존 test_shared_prefix_flex.py의 integration.compile_friendly_flex_attention witness를 같은 방식으로 이 테스트에 적용한다. Record actual calls and rectangular K length=root_length+query_length; do not accept an unnoticed dense fallback.

- [ ] **Step 2: 기존 distributed harness를 새 기본 objective로 바꾸는 RED를 추가한다.**

test_two_rank_shared_flex_dora_zero2_accumulation의 기존 결과 검사에 다음을 추가한다.

```python
assert witness["objective"] == "context_preference"
for rank in witness["ranks"]:
    assert len(rank["scoring_microbatches"]) == 2
    for batch in rank["scoring_microbatches"]:
        assert batch["policy_calls_forward"] == 1
        assert batch["scoring_decoder_calls_forward"] == 3
        assert batch["latent_decoder_calls_forward"] == 4
    assert rank["legacy_route_calls"] == 0
```

유휴 GPU가 확인된 뒤에만 기존 smoke를 실행해 RED를 확인한다. Resources가 없으면 GPU 테스트를 실행한 것처럼 기록하지 않는다. CPU 수치 검증 결과를 먼저 보고하고 GPU 미검증을 남긴다.

- [ ] **Step 3: DoRA/ZeRO2 smoke의 config와 batch만 교체한다.**

```python
from step_preference.config import StepPreferenceConfig
config = StepPreferenceConfig(enabled=True,mode="stage2b",
    stage2a_checkpoint_path="tiny-stage2a",attention_implementation="sdpa",
    sidecar_attention_backend="flex")
# Immediately after tiny_measurement_case:
batch["step_preference"] = {"steps": tuple({"span":s} for s in batch["reasoning_step_spans"])}
```

기존 실제 Stage2Trainer, optimizer-step, accumulation, DoRA dropout 0.05, checkpointing, save_strategy=no를 유지한다. 기존 RuntimeCounters context 안에서 microbatch별 delta를 수집한다.

```python
scoring_microbatches = []
legacy_route_calls = 0
original_compute = trainer.compute_loss
original_legacy = trainer._compute_stage2b_loss
def reject_legacy(*args,**kwargs):
    nonlocal legacy_route_calls
    legacy_route_calls += 1
    raise AssertionError("default context smoke entered legacy Stage 2B")
def observed_compute(*args,**kwargs):
    before = counters.snapshot()
    value = original_compute(*args,**kwargs)
    after = counters.snapshot()
    delta = {key:after.get(key,0)-before.get(key,0) for key in
        ("policy_calls_forward","decoder_calls_forward","latent_decoder_calls_forward")}
    delta["scoring_decoder_calls_forward"] = delta["decoder_calls_forward"]-delta["latent_decoder_calls_forward"]
    scoring_microbatches.append(delta)
    return value
trainer.compute_loss = observed_compute
trainer._compute_stage2b_loss = reject_legacy
try:
    trained = trainer.train()
finally:
    trainer.compute_loss = original_compute
    trainer._compute_stage2b_loss = original_legacy
```

위 code는 기존 observed_backward/accelerator.backward 교체의 try/finally 안에 넣어 backward phase 분리를 보존한다. Trainer.train을 두 번 호출하지 않는다. 기존 rank record에 scoring_microbatches와 legacy_route_calls를 추가하고 최상위 witness에 objective="context_preference"를 추가한다. 테스트의 기존 forward/backward parameter-update assertions도 보존한다. 이 smoke는 한 optimizer step이며 학습 품질 평가가 아니다.

- [ ] **Step 4: README에 기본 명령과 제한을 명문화한다.**

아래 실행 예시는 사용자가 실제 Stage2A checkpoint와 dataset 경로를 제공할 때 사용하는 형식이다. 계획 실행 중 이 명령으로 본 학습을 시작하지 않는다.

```bash
STEP_PREFERENCE_ENABLED=true \
STEP_PREFERENCE_MODE=stage2b \
STEP_PREFERENCE_OBJECTIVE=context_preference \
STEP_PREFERENCE_BETA=0.1 \
STEP_PREFERENCE_WEIGHT=1.0 \
STEP_PREFERENCE_MARGIN=0.0 \
STEP_ENTROPY_WEIGHT=0.1 \
STEP_ENTROPY_MARGIN=0.1 \
STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT" \
STAGE2A_CHECKPOINT_PATH="$STAGE2A_CHECKPOINT_PATH" \
DATA_PATH="$DATA_PATH" IMAGE_ROOT="$IMAGE_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash run_stage2_training.sh
```

README 설명에는 기존 clean latent decoder를 제외한 clean/V/T scoring ≤3, reference/negative=0, 균등 step reduction, 메모리 부족 시 sidecar chunk 금지, ordinary Stage2 enabled=false 유지, legacy_dpo 명시 선택과 과거 negative 파일의 별도 필요성을 포함한다. OUTPUT_DIR은 현재 worktree checkpoints의 서로 다른 run directory로 지정한다고 명시한다.

- [ ] **Step 5: GPU resource preflight 후 작은 checks만 실행한다.**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

다른 프로세스를 종료하거나 신호를 보내지 않는다. 유휴 GPU가 두 개면 해당 index를 CUDA_VISIBLE_DEVICES로 지정한다. 다음 예시는 0,1이 모두 유휴일 때만 사용한다. 하나만 유휴라면 CUDA parity만 실행하고 two-rank smoke는 미실행으로 보고한다. 기존 harness가 전체 host process 유무를 검사해 거부하면 그 안전 검사를 무력화하지 않는다.

```bash
RUN_CONTEXT_CUDA=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q -s tests/step_preference/test_context_sidecars.py -k cuda > "logs/$ILVR_CONTEXT_RUN_ID/cuda-parity.log" 2>&1
RUN_SHARED_PREFIX_DDP=1 SHARED_PREFIX_RUN_ID="${ILVR_CONTEXT_RUN_ID}-ddp-red" CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/stage2/integration/test_shared_prefix_distributed.py > "logs/$ILVR_CONTEXT_RUN_ID/ddp-red.log" 2>&1
```

DDP RED 명령은 Step 2 시점에, GREEN은 Step 3 후 서로 다른 fresh run ID로 실행한다:

```bash
RUN_SHARED_PREFIX_DDP=1 SHARED_PREFIX_RUN_ID="${ILVR_CONTEXT_RUN_ID}-ddp-green" CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q tests/stage2/integration/test_shared_prefix_distributed.py > "logs/$ILVR_CONTEXT_RUN_ID/ddp-green.log" 2>&1
```

실패한 run logs를 보존한다. 수정이 필요하면 원인별 regression 1개와 가장 작은 수정 후 fresh run ID로 재시도한다. GPU 실행은 unified exec로 시작한 뒤 session ID로 추적하고 60초 이상 사용자 업데이트를 비우지 않는다.

- [ ] **Step 6: 변경 review 뒤 최종 검증 command를 한 번 실행한다.**

requesting-code-review를 사용해 task 결과와 spec/plan을 reviewer에게 전달한다. Root만 paperthin hook을 사용할 수 있으며 구현자/리뷰어에게 hook이나 추가 paperthin agent 실행을 지시하지 않는다. Critical/Important finding은 원인에 맞는 테스트와 최소 수정으로 해결한다.

완료 후 branch handoff가 바로 이어지므로 verification-before-completion과 finishing-a-development-branch를 함께 적용한다. 다음 CPU full-suite를 두 gate의 겹치는 test-pass 주장에 한 번만 사용한다. GPU assertions는 위 별도 명령 결과와 연결한다.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ./venv/ilvr-venv/bin/pytest -q > "logs/$ILVR_CONTEXT_RUN_ID/full-suite.log" 2>&1
bash -n run_stage2_training.sh
git diff --check
```

Full-suite 전체 출력과 exit status를 읽는다. GPU tests가 skipped인 CPU suite를 “GPU 포함 전체 검증 완료”라고 부르지 않는다. 동일 suite를 바로 다시 돌리지 않는다. 새로운 fix가 있으면 영향을 받는 witness와 최종 claim에 필요한 검증만 재실행한다.

- [ ] **Step 7: 결과와 제한을 즉시 기록하고 마지막 수정들을 커밋한다.**

results/$ILVR_CONTEXT_RUN_ID/summary.md에 다음 필드를 실제 관측값으로 작성한다:

| 보고 항목 | 증거 |
| --- | --- |
| K=1/4/8/16 실제 outer policy/decoder/latent/scoring 호출 | Task 3/4 K parameterized hook 출력 |
| backward recomputation | phase=backward의 별도 counters |
| reference/negative/rejected 비호출 | first_nail traps와 실제 기본 경로, legacy route trap |
| clean/V/T log-prob 및 entropy sanity | context metric 출력; finite 여부 |
| first-token/locality/no-propagation/prefix identity/mRoPE | Task 4 runtime tests |
| 동일 target/prefix 및 logits reuse | planner target identity, independent oracle, 3-call witness |
| step 균등 reduction | literal unequal-length scalar와 oracle |
| baseline 보존 | disabled-preference 직접 비교 및 full CPU suite |
| GPU Flex / DoRA ZeRO2 | 실제 실행 로그와 rank witness, 또는 명시적인 미실행 사유 |
| 변경 파일과 재사용 구현 | git diff 및 task commits |
| 미검증 범위 | 긴 실입력 메모리, full training 품질, stochastic naive equality |

환경상 GPU witness가 미실행이면 그 제한을 명시하고 검증 완료로 포장하지 않는다. Final response 직전에 ntfy summary hook을 실행한다. 구현이 실제로 끝나고 필수 witness가 충족됐을 때만 success, 그렇지 않으면 partial을 사용한다.

```bash
git add tests/step_preference/test_context_sidecars.py tests/stage2/integration/shared_prefix_smoke.py tests/stage2/integration/test_shared_prefix_distributed.py README.md
git commit -m "test: validate context preference GPU and training integration"
```

results/logs를 강제로 git add하지 않는다. Branch ours1은 기존 사용자 worktree이므로 임의 삭제/정리하지 않는다. Merge/PR/keep 결정은 finishing skill의 사용자 handoff로 남기며, 어떤 선택도 자동으로 실행하지 않는다.

## Spec Coverage와 실행 중 체크

| Spec 계약 | Task / witness |
| --- | --- |
| §1 확정된 방법/값 | Task 1 loss, Task 3 config와 reducer |
| §2 최소 교정 및 새 기본 경로 | Tasks 2–3; 기존 helper 유지, main/launcher 포함 |
| §3 실제 count ≤3, latent/backward 제외, K/empty | Tasks 2–4 CPU hooks, Task 5 per-microbatch smoke |
| §4 local V/T, 첫 predictor, valid spans | Task 2 existing planner validation, Task 4 literal visibility/runtime perturbation |
| §5 공유 KV/gradient/position/격리 | Task 4 bank identity/all-gradient oracle/no-propagation |
| §6 같은 logits와 두 단계 reduction | Tasks 1,3 reducer, Task 4 unequal lengths + independent oracle |
| §7 negative/reference 기본 의존성 제거 | Task 3 preflight→collator→trainer first_nail + shell test |
| §8/§9 완료 증거/불확실성/first_nail | Tasks 3–5, 실제 로그 기반 summary |
| §10 lifecycle/venv/storage | 준비 절, 각 task commands, Task 5 finish |

계획 작성 self-review는 root가 완료했다. 위 표로 spec coverage를 대조하고,
미정 placeholder가 없는지 확인했으며 task 간 type/signature/metric key를 대조했다.
그 과정에서 tiny vision fixture의 spatial_merge_size=2와 기존 legacy fixture의 추가 전환 대상을 보완했다.
이 검토를 subagent에게 넘기지 않았다.
별도 shower reviewer는 plan/spec 사본 전체를 읽고 stands on its own으로 판정했다.
실행을 막는 모호함은 지적되지 않았다. 이는 문서 이해 가능성 검토이며 구현 검증이 아니다.

## Execution Handoff

AGENTS.md는 사용 가능한 subagent-driven-development 경로를 기본으로 지정한다. 승인된 spec과 이 plan을 작업자에게 함께 전달하고 task별로 구현 및 두 단계 review를 수행한다. Task 1과 2의 의존성은 분리되어 있지만 기본값 전환인 Task 3은 둘이 통과한 뒤 진행한다. Task 4–5는 새 기본 경로가 동작한 뒤 진행한다.

Writing-plans handoff에서 제시할 선택:
1. Subagent-Driven (recommended): task별 fresh implementer와 review.
2. Inline Execution: 사용자가 이 경로를 선택하면 executing-plans로 같은 plan을 이 세션에서 실행.

구현 재개 전 execution 방식 선택을 사용자에게 전달한다. 새로 해결해야 할 의미상 모호함이 생기면 사용자 답변 전에 dependent work를 진행하지 않는다.
