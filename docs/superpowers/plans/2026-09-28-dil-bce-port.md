# DIL-BCE Variant Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 별도 ILVR worktree에서 원본 8개 DIL-BCE variant와 clean/voff entropy 선택을 학습·저장·resume·평가 경로에 연결한다.

**Architecture:** 원본 objective primitives를 보존하고 검증된 context-preference trainer에 선택적으로 연결한다. Top-k 계산은 기존 sidecar checkpoint 안에서 수행하고 token별 scalar 통계를 반환한다. 기존 coordinator와 PEFT artifact 경로를 재사용한다.

**Tech Stack:** Python 3.11 repository venv, PyTorch 2.7.1/CUDA12.8, vendored Transformers Qwen2.5-VL, PEFT DoRA, DeepSpeed ZeRO-2, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-28-dil-bce-port-design.md`, 승인 revision `6127aa5d4c81af1b830a01f5bdea750f4c087808`, SHA-256 `3d4ca6df758a67e3ab2b6ce6f4366d7d9f2680cbcbe51b3b089b9d519203f38b`. 사용자가 문서 제시 뒤 `승인`이라고 답했다. 명세의 '검토 대기' 문구는 작성 당시 상태이며 실제 승인 기록은 `results/dil-bce-port-workflow/spec-review-record.json`에 있다.

## Global Constraints

- Worktree `/root/workspace/ILVR/.worktrees/dil-bce-port`, branch `feat/dil-bce-port`; 기반 `51e11733084275bb2b1b2079924368d3cd636d9b`.
- 원본 `/raid/workspace/temp2/ILVR/.worktrees/ours1-dil-bce`는 read-only; source identity manifest를 확인하고 원본 trainer 전체를 복사하지 않는다.
- selector `dpo | dil_bce_rf | dil_bce_rf_topk`; 기본 `dpo`, 기존 경로 수치·gradient·CLI 호환성 유지.
- 새 BCE 지원: Stage2B context_preference, V-only, clean_prefix, independent, shared-prefix ON, latent-body mask, START/END 유지·supervision, original image 접근, reference-free, CE1/entropy0/margin0.
- 기본 학습: 2 nodes × 2 GPUs = 4 ranks, microbatch1, accumulation4, global batch16, LR5e-5, seed42, warmup10, BF16, beta0.5/preference1.
- 기본 target immediate_text; until_next_latent 명시 선택 지원. reference_random/concatenated-v1/crossing-token 제외와 partial-tail 분모 보존.
- DoRA r32/alpha64/dropout0.05/latent8, q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj; decoder SDPA/vision FA2/sidecar Flex1024, native checkpoint OFF, latent·기존 sidecar checkpoint ON, ZeRO-2, sequence8192, workers0.
- Original base + reproduced Stage1 checkpoint-431 사용. 3379 retained의 전체 노출은 212 updates; 실제 retained/sampler로 확인하며 기존 phase 의미 유지.
- top-k negative에서 q 양쪽 gradient 유지, clean mass target만 detach, mass coefficient는 preference coefficient와 독립, token 평균 후 sample-local step 평균.
- Entropy는 full vocabulary, sample별 ceil quota, stable token-order ties; 미선택 step도 분모에 유지; OFF/100% parity; clean CE 유지.
- 원본 과거 training checkpoint migration/weights·results 복사/전체 재학습/latent-causal 도구는 범위 밖.
- Python·tests는 worktree `venv/ilvr-venv`, vendored PYTHONPATH 사용; pip reinstall 금지. 모델 상태·tiny CPU checkpoint fixture도 `checkpoints/`, dataset은 `data/`, metadata는 `results/`, 로그는 `logs/` 사용.
- 실제 GPU 학습·준비·평가는 이 계획 승인에 포함되지 않는다. Task 6에서 구체적인 별도 smoke 제안을 작성한다. CPU tiny fixture의 저장·reload는 승인된 개발 테스트 범위다.
- 구현 방법은 아직 미선택. 선택된 방법의 implementer/fixer와 독립 review gates를 따른다. 모든 테스트·review 결과는 실제 revision과 연결한다.

## Review Focus

1. Empty target 또는 entropy가 한 step에 몰린 sample: CE 유지, finite zero와 원래 step 분모 유지 — Tasks 1/3의 empty-step 테스트.
2. 반복 token·같은 길이의 다른 step·checkpoint backward 재실행: shape만 일치하는 잘못된 clean/OFF 결합을 거부 — Task 3의 permuted-identity/gradient oracle.
3. 설정 source 후 stale k/entropy 또는 wrapper의 beta alias: 실제 양 노드 argv가 요청 조합과 일치 — Tasks 2/5의 parser와 poisoned-environment dry-run.
4. checkpoint의 config 누락/손상 또는 BCE에서 DPO로 resume: optimizer 진행 전에 실패하고 Stage1 fresh load는 허용 — Task 4의 resume matrix.
5. rank별 step 수 불일치와 마지막 3-real+1-padding update: sample 균등 가중 및 global16 분모, padding RNG·metrics 보존 — Task 3의 실제 Trainer 기반 tail witness.

---

## File Structure and Shared Test Setup

| 소유 작업 | 생성/수정 파일 | 책임 |
|---|---|---|
| 1 | `src/step_preference/dil_bce_loss.py`, `dil_topk_loss.py`, `entropy_selection.py` (생성) | 수식, token 통계, scalar entropy mask |
| 2 | `src/step_preference/dil_config.py` (생성), `config.py`, `src/utils_deepseed.py`, `src/main.py` | 새 필드, 지원 조합, 실제 vocab 검증 |
| 3 | `src/step_preference/dil_reduction.py` (생성), `src/stage2/trainer.py`, vendored Qwen `stage2_shared_prefix.py`, `modeling_qwen2_5_vl.py` | checkpoint-local 계산과 trainer reduction |
| 4 | `src/step_preference/dil_identity.py` (생성), `src/stage2/trainer.py`, `src/stage2/peft.py` | config identity 저장·resume·inference 복원 |
| 5 | `run_stage2_training.sh`, `src/multinode_runtime.py`, `scripts/stage2/dil_bce_profile.sh`와 `docs/dil-bce.md` (생성), `AGENTS.md` | 옵션 전달, 명시적 프로파일, 8개 직접 실행 예시 |
| 6 | `tests/step_preference/compare_dil_bce_source.py` (생성), workflow results/logs | source 비교 증거, 최종 검증, GPU smoke 제안 |

Task 의존성: 1 → 2 → 3 → 4 → 5 → 6. Task 4/5는 공통 trainer/설정 변경이
완료된 뒤 수행한다. 구현 방법과 무관하게 기존 사용자 파일을 덮어쓰지 않는다.

각 테스트 shell은 다음 환경을 먼저 설정한다. 아래에서 `pytest`라고 적은 명령은
항상 `./venv/ilvr-venv/bin/python -m pytest -p no:cacheprovider`로 실행한다.

```bash
export PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs/dil-bce-port-workflow results/dil-bce-port-workflow
```

테스트별 임시 model fixture는 `--basetemp="$PWD/checkpoints/dil-bce-port-cpu/task-N"`에
둔다. 이 디렉터리는 테스트 전용이며 pytest가 재사용 시 지우는 영역이므로 다른
run을 넣지 않는다. RED/GREEN의 stdout/stderr와 exit status는 worktree logs에
각각 보존한다. shell pipeline이면 `set -o pipefail`을 사용한다.

각 task의 commit 전 해당 GREEN와 `git diff --check`를 확인한다. 선택된 실행
skill의 task review가 요구되면 그 verdict와 수정 루프를 마친 뒤 다음 작업으로
넘어간다. Native의 독립 whole-branch review는 Task 6에서 수행한다.

## Task 1: Loss primitives and entropy masks

**Files:** 위 표의 Task 1 파일 생성; 생성 테스트
`tests/step_preference/test_dil_bce_loss.py`, `test_dil_topk_loss.py`,
`test_dil_entropy_selection.py`.

**Interfaces:**
- `dil_bce_positive(clean_scores: Tensor, beta: float) -> Tensor` 및
  `dil_bce_loss(clean_scores: Tensor, off_scores: Tensor, beta: float, *, stage2_debugging: bool=True) -> DilBceResult`.
- `DilBceResult`: 원본의 loss/pair_losses/positive_losses/negative_losses/policy_margins.
- `topk_token_terms(clean_logits: Tensor, off_logits: Tensor, target_ids: Tensor, *, beta: float, k: int, compute_mass: bool=True, chunk_size: int=128, stage2_debugging: bool=True) -> TopkTokenResult`.
- `TopkTokenResult`: negative_loss, mass_loss, entropy, clean_mass, off_mass,
  mass_abs_gap, gt_coverage의 7개 FP32 `[T]` Tensor를 가진 NamedTuple.
- 원본 `topk_negative_and_mass(...) -> TopkStepResult` signature를 유지하는
  wrapper는 위 7개 token 배열을 각각 평균한다. token helper는 nonempty `[T,V]` 입력.
- `select_entropy_masks(step_entropies: Sequence[Tensor], percent: float) -> tuple[Tensor, ...]`:
  한 sample의 원래 token 순서 `[T_j]` entropy들을 받아 동일 shape bool masks 반환.
  모든 입력이 비었으면 빈 masks를 반환한다. 호출자는 sample을 섞지 않는다.

- [ ] **1.1 RED tests 작성.** `test_zero_score_is_two_log_two_and_signed_gradients`에서 beta=.5, clean=off=0의 loss=2log2, gradient=(-.25,+.25).
  `test_topk_live_q_and_detached_clean_mass_match_independent_oracle`는 float64 독립
  수식에서 clean mass detach/off complement gradient와 q 양쪽 gradient를 검사한다.
  `test_mask_quota_and_ties`는 T=25/P=28의 선택 수7, 모든 entropy 동일 시 처음7개,
  극소 양수 P는1개, sample별 별도 호출의 선택이 독립임을 검사한다.

```python
clean = torch.zeros(1, requires_grad=True)
off = torch.zeros(1, requires_grad=True)
loss = dil_bce_loss(clean, off, .5).loss
assert loss.item() == pytest.approx(2 * math.log(2))
loss.backward()
torch.testing.assert_close(clean.grad, torch.tensor([-.25]))
torch.testing.assert_close(off.grad, torch.tensor([.25]))
mask, = select_entropy_masks([torch.ones(25)], 28.)
assert torch.equal(mask, torch.arange(25) < 7)
```

- [ ] **1.2 RED 실행.** `pytest -q tests/step_preference/test_dil_bce_loss.py tests/step_preference/test_dil_topk_loss.py tests/step_preference/test_dil_entropy_selection.py`.
  예상 실패는 새 모듈/함수 부재이며 환경 import 오류와 구분한다.
- [ ] **1.3 최소 구현.** source identity를 확인한 뒤 BCE와 안정적인 group-gap/mass
  계산을 재사용한다. Top-k 내부를 token 결과로 분리하고 기존 wrapper 의미를 유지한다.
  entropy selector는 `Fraction(str(percent))/100`, stable descending argsort,
  detached ranking을 사용한다. 선택 support는 OFF.detach().topk의 indices이며 값은 live다.
- [ ] **1.4 경계 검증 추가.** uniform top-k negative floor `log(1+k**(-.5))`,
  equal group gap, finite extreme FP32, FP64→FP32 overflow, zero probability,
  BF16, 부분 chunk, k=1/k=V/non-integer, NaN/inf beta, empty BCE differentiable zero,
  shape/device/target-range 오류를 검증한다. 원본 step 평균과 token helper의 평균을 비교한다.
- [ ] **1.5 GREEN 실행.** 1.2의 세 파일 실행, 실패0 확인. float32 ordinary oracle은
  atol1e-6/rtol1e-5, gradients는 atol1e-5/rtol1e-4; 극값과 BF16은 별도 수학적
  허용오차를 테스트에 명시하고 일반 오차를 넓혀 숨기지 않는다.
- [ ] **1.6 Commit.** 해당 세 production/세 test 파일만 stage하여
  `feat: port DIL-BCE primitives and entropy selection`.

## Task 2: Configuration and entry validation

**Files:** `src/step_preference/dil_config.py` 생성; `src/step_preference/config.py`
`StepPreferenceConfig.__post_init__/from_args`, `src/utils_deepseed.py` parser,
`src/main.py` model/config resolution 수정. 생성 `tests/step_preference/test_dil_config.py`.

**Interfaces:**
- Config 필드: `preference_loss_type: str='dpo'`, `preference_topk: int|None=None`,
  `lambda_topk_mass: float=0.0`, `preference_entropy_top_percent: float|None=None`,
  `preference_entropy_source: str='clean'`.
- `validate_dil_config(config: StepPreferenceConfig) -> None`와
  `validate_dil_runtime(config: StepPreferenceConfig, *, vocab_size: int, supervise_latent_boundaries: bool) -> None`.
  TYPE_CHECKING import로 config/helper 순환 import를 피한다.
- `uses_entropy_selection(config: StepPreferenceConfig) -> bool`: top-k + 비율이
  존재하고 100 미만일 때만 true. 이후 task는 이 판단을 재사용한다.

- [ ] **2.1 RED tests 작성.** `test_real_parser_resolves_all_dil_fields`는 실제
  parser→from_args를 거쳐 beta=.5/weight1/k8/mass1/percent20/sourcevoff를 검사한다.
  beta option의 실제 destination은 기존 `step_dpo_beta`임을 검사한다.
  omitted vs explicit dpo의 기존 fields가 같아야 한다.

```python
# args는 위 실제 parser 입력으로 생성한다.
config = StepPreferenceConfig.from_args(args)
assert config.beta == .5 and config.lambda_preference == 1.
assert (config.preference_topk, config.lambda_topk_mass) == (8, 1.)
assert (config.preference_entropy_top_percent, config.preference_entropy_source) == (20., 'voff')
with pytest.raises(ValueError):
    validate_dil_runtime(config, vocab_size=8, supervise_latent_boundaries=True)
```

- [ ] **2.2 RED 실행.** `pytest -q tests/step_preference/test_dil_config.py`.
  미등록 CLI/필드 또는 검증 누락을 관측한다.
- [ ] **2.3 필드·검증 연결.** 명세의 다섯 CLI, 대응 필드 및 지원 조합 검사 추가.
  BCE가 disabled/Stage2A/legacy/T-off/다른 prefix/boundary suppression/image 차단/
  CE!=1/entropy!=0/margin!=0/shared-prefix OFF일 때 실패한다. finite numeric에서
  bool을 수치로 허용하지 않는다. DPO의 기존 지원 조합은 바꾸지 않는다.
- [ ] **2.4 실제 runtime 조기 검사.** resized output vocabulary와 boundary supervision을
  확인한 뒤 모델 첫 forward 전에 runtime validator를 실행하도록 main과 이후
  trainer 경로에 연결할 contract를 정한다. 빈-target batch도 이 검사를 우회하지 않는다.
- [ ] **2.5 GREEN matrix.** P=None/100/20, clean/voff, immediate/until, mass0/1,
  lambdaP0을 포함한다. k>=V, 과거 DPO에 stale k, sourcevoff+P=None,
  finite 아닌 coefficient를 거부한다.
  `pytest -q tests/step_preference/test_dil_config.py tests/step_preference/test_config.py tests/step_preference/test_target_scope.py`.
- [ ] **2.6 Commit.** Task 2 파일을 stage하여 `feat: validate DIL-BCE objective configuration`.

## Task 3: Checkpoint-local sidecar statistics and trainer integration

**Files:** `src/step_preference/dil_reduction.py` 생성; `src/stage2/trainer.py`의
`_compute_context_preference_loss/_reduce_context_objective`, vendored
`transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py`의
`validate_shared_inputs/_score_chunk/score_sidecars`, 같은 경로
`modeling_qwen2_5_vl.py`의 clean-before-sidecar 호출부 수정.
생성 `tests/step_preference/test_dil_sidecar.py`, `test_dil_trainer.py`,
`test_dil_entropy_integration.py`, `tests/stage2/test_dil_reference_tail.py`.

**Interfaces:**
- shared-prefix options에 optional `dil_objective` mapping 추가:
  `{'loss_type': 'dil_bce_rf_topk', 'beta': float, 'k': int}`. 외부 caller가
  arbitrary clean logits를 options로 주입하지 못한다. 기본 BCE/DPO는 기존 options 사용.
- `score_sidecars(...existing positional args..., *, dil_clean_logits: Tensor|None=None)`:
  모델이 자신의 clean selected logits `[1,T,V]`를 넘긴다. plan의 root positions,
  step IDs, target IDs로 branch 행을 맞추고 detached `[T_j,V]`를 checkpoint의
  명시적인 tensor 입력으로 전달한다. clean CE/positive의 live tensor는 유지한다.
- Top-k sidecar result의 기존 name/kind/step_id/mean_logp/entropies 유지;
  추가 `dil_tokens: TopkTokenResult`, `target_logps: Tensor[T_j]`,
  `target_ids: Tensor[T_j]`, `predictor_positions: tuple[int,...]`.
  checkpoint callable은 NamedTuple을 구성하기 전 flat tensor tuple로 반환한다.
  `dil_tokens`와 logp/entropy 외에 `[T,V]` 출력은 없다.
- `reduce_dil_context_loss(clean_ce: Tensor, clean_token_logps: Sequence[Tensor], off_token_logps: Sequence[Tensor], *, config: StepPreferenceConfig, topk_terms: Sequence[TopkTokenResult]|None=None, selection_entropies: Sequence[Tensor]|None=None) -> tuple[Tensor, dict[str,Tensor]]`.
  기본 BCE에서는 off scalar mean을 length1 tensor로 받아도 되고 top-k에서는 정확한
  per-token 배열을 요구한다. caller가 항상 실제 step 순서로 전달한다.

- [ ] **3.1 RED model tests 작성.** existing `_tiny_model`, `tiny_measurement_case`,
  `_trainer`를 재사용하여 실제 model 입력 두 unequal steps의 BCE/top-k/filtered
  전체 loss를 독립 full-logit oracle과 비교한다. `test_sidecar_rejects_permuted_step_identity`
  는 길이가 같아도 다른 step IDs/targets를 섞으면 실패한다. 반환 scalar는 detached가 아니어야 한다.

```python
# actual/oracle은 동일 tiny forward에서 얻은 logits의 두 독립 loss 계산이다.
torch.testing.assert_close(actual, oracle, atol=1e-6, rtol=1e-5)
for got, want in zip(actual_parameter_grads, oracle_parameter_grads):
    if want is not None:
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
assert all(t.ndim == 1 for result in sidecars for t in result['dil_tokens'])
assert torch.count_nonzero(observed_live_kv_gradient) > 0
```

- [ ] **3.2 RED 실행.** `pytest -q tests/step_preference/test_dil_sidecar.py tests/step_preference/test_dil_trainer.py tests/step_preference/test_dil_entropy_integration.py`.
  새 dispatch/result contract 부재를 확인한다.
- [ ] **3.3 Sidecar 연결.** 기존 clean projection/CE 순서를 유지한 채 selected logits를
  scorer에 전달한다. top-k checkpoint 안의 기존 OFF LM-head 결과에 Task 1 kernel을
  적용하고 branch별 token 통계를 분리한다. clean target만 detach하고 live KV와
  independent group embedding lookup은 변경하지 않는다. 기존 DPO callable/output 유지.
- [ ] **3.4 Trainer reduction.** config selector dispatch, actual resized vocab/boundary
  validation, clean GT vector 추출, Task 1 mask, positive(mean selected logp), selected
  negative/mass token 평균→원래 step 평균을 연결한다. no-selected-step와 no-step은
  그래프 연결된 zero이며 CE는 유지한다. mass-only도 sidecar가 실행되어야 한다.
  diagnostics는 positive/negative/mass, clean/off score, q entropy/GT coverage,
  mass gap, eligible/selected counts와 source/percent를 기존 numeric reduction과
  string logging 경로에 맞춰 기록한다. 필터 OFF/100은 selection helper를 호출하지 않는다.
- [ ] **3.5 Gradient/cost regression.** `test_dil_checkpoint_replay_preserves_gradients`는
  checkpoint on/off의 loss·parameter gradients를 비교하고 live KV/latent의 nonzero
  gradient를 hooks로 관측한다. `test_dil_adds_no_forward_calls`는 동일 tiny fixture의
  DPO/BCE/top-k/clean20/voff20 forward decoder·LM-head 횟수 차이가0임을 검사한다.
  backward 재계산 횟수는 forward count와 별도 기록한다. `test_dpo_unchanged`와
  `test_entropy_100_matches_disabled`는 loss·logits·parameter gradients의 exact parity를 검사한다.
- [ ] **3.6 Tail/empty tests.** 실제 trainer compute_loss/training_step 경로로 기본 BCE,
  mass-only, filtered mode를 테스트한다. 서로 다른 1/2/3-step sample의 loss를 sample별
  평균한 oracle과 비교하며 3-real+padding 기여는 `(g1+g2+g3)/16`이다. padding은
  loss/gradient0, RNG 복원, metric 증가0을 검사한다. 기존 real Trainer partial-step
  test fixture를 활용해 configured GAS4가 마지막 step에도 적용되는지 확인한다.
  sample의 한 step에만 선택이 집중되면 총 step 수로 나누며, empty sample은 CE만 남는다.
- [ ] **3.7 GREEN 실행.** 위 네 신규 파일과
  `tests/step_preference/test_context_preference.py`, `test_clean_first_order.py`,
  `test_context_embedding_alignment.py`, `tests/stage2/test_reference_random_sampler.py`,
  `test_reference_sampler_resume_runtime.py`, `test_reproduction_checkpointing.py`.
  모두 failure0이어야 한다. 추가 decoder/head, 큰 OFF 반환 tensor, detach가 있으면 수정한다.
- [ ] **3.8 Commit.** Task 3 production/test 파일만 stage하여
  `feat: integrate checkpointed DIL-BCE context losses`.

## Task 4: Objective identity, resume and PEFT inference round-trip

**Files:** `src/step_preference/dil_identity.py` 생성;
`src/stage2/trainer.py`의 `train/_save`, `src/stage2/peft.py` inference loader 수정.
생성 `tests/step_preference/test_dil_identity.py`, `tests/stage2/test_dil_peft_roundtrip.py`;
기존 `tests/multinode/test_artifacts.py`에 새 config namespace 전달 case 추가.

**Interfaces:**
- `build_dil_identity(config: StepPreferenceConfig) -> dict[str,object]|None` (DPO이면 None).
- `read_dil_identity(config_payload: Mapping[str,object]) -> dict[str,object]|None`:
  `ilvr_context_preference` namespace의 strict key/type/version 검사.
- `validate_dil_resume(checkpoint: Path, requested: StepPreferenceConfig) -> None`:
  checkpoint config와 요청 identity 비교; 둘 다 None인 historical DPO만 기존 처리.
- `attach_dil_identity(model, config: StepPreferenceConfig) -> None`:
  underlying Qwen config에 identity 저장, stale incompatible namespace를 조용히 남기지 않는다.
- Identity fixed keys: schema_version=1, loss_type, formula_version, beta,
  lambda_preference, lambda_ce, lambda_entropy, lambda_topk_mass, preference_margin,
  topk, entropy_top_percent, entropy_source, target_scope, prefix_mode, branch_mode,
  visual_only, mask='latent_body_attention', boundaries='visible_supervised',
  block_original_image=false, shared_prefix=true, sidecar_attention_backend,
  sidecar_query_budget, sidecar_checkpoint=true,
  reduction='token_mean_then_sample_step_mean_v1',
  entropy_selection='sample_full_vocab_stable_ceil_original_step_denominator_v1'.
  formula_version은 명세의 BCE/topk 문자열. OFF percentage는 null이며 k 미사용은 null.

- [ ] **4.1 RED tests 작성.** `test_identity_roundtrip_and_mutation_rejection`은 모든
  semantic field를 하나씩 바꿔 resume를 거부한다. `test_bce_resume_missing_or_unknown_identity`
  는 missing/unknown field/version/wrong type을 거부한다. `test_bce_to_dpo_resume_rejected`
  는 BCE namespace를 DPO 요청으로 덮어쓰지 못하게 한다.

```python
payload = {'ilvr_context_preference': build_dil_identity(config)}
assert read_dil_identity(payload) == payload['ilvr_context_preference']
# checkpoint fixture는 위 payload를 config.json으로 실제 저장한다.
validate_dil_resume(checkpoint, config)
with pytest.raises(ValueError):
    validate_dil_resume(checkpoint, dataclasses.replace(config, beta=1.))
```

- [ ] **4.2 RED 실행.** `pytest -q tests/step_preference/test_dil_identity.py tests/stage2/test_dil_peft_roundtrip.py`.
- [ ] **4.3 Save/resume 연결.** `train`이 True/False/path resume를 정규화한 뒤
  super.train/optimizer 복원 전 identity 검증. `_save`에서 PEFT/full 모두 config
  저장 전에 identity 부착. 기존 runtime/sampler/ZeRO/RNG 검사 유지. Stage1 fresh
  initialization에는 resume 검사를 실행하지 않는다. 실제 runtime vocab/mask 검증을 통과한
  config만 저장하게 한다.
- [ ] **4.4 Inference 연결.** `load_stage2_model_for_inference`가 artifact의 namespace를
  읽어 strict 검증하고 로드한 Qwen config에 보존한다. training loss를 호출하지 않는다.
  기존 PEFT/sparse-token/base 검사와 config.json 전달 allowlist를 재사용한다.
  metadata namespace 때문에 generation이 바뀌지 않아야 한다.
- [ ] **4.5 GREEN round-trip.** 기존 `write_sparse_stage2_artifact` fixture로 실제 tiny
  adapter/config를 저장하고 original tiny base와 재로드한다. identity 보존, token rows와
  고정-input logits/latent states 일치를 확인한다. collector/inference manifest가
  config.json을 보존하며 optimizer state는 inference 복사 대상이 아닌지 검사한다.
  malformed namespace는 큰 model load 전에 거부한다. historical DPO와 fresh Stage1
  입력은 기존과 같이 동작한다.
  Run: 신규 두 파일 + `tests/stage2/test_peft_inference.py tests/stage2/test_peft_checkpointing.py tests/multinode/test_artifacts.py`.
- [ ] **4.6 Commit.** `feat: persist DIL-BCE objective identity across checkpoints`.

## Task 5: Coordinator forwarding and eight runnable examples

**Files:** `run_stage2_training.sh`의 env→CLI 조립, `src/multinode_runtime.py` allowlist,
`scripts/stage2/dil_bce_profile.sh`와 `docs/dil-bce.md` 생성, `AGENTS.md`의 과거 batch 문구
정리. 생성 `tests/multinode/test_dil_forwarding.py`, `tests/step_preference/test_dil_examples.py`.

**Interfaces:**
- 신규 다섯 환경 변수→명세의 다섯 CLI, 기존 beta alias/weight/scope 재사용.
- `scripts/stage2/dil_bce_profile.sh`는 source-only common profile이며 process 실행 없음.
  외부 run env source 다음에 호출하여 전체 canonical fresh-run profile을 명시하고,
  DIL selector/k/mass/percent/source를 neutral 값으로 reset한다. RESUME_CHECKPOINT가
  비어 있지 않으면 fresh profile 적용을 거부한다. resume는 저장된 실행 설정을 사용한다.
- 문서는 8개 explicit `env ... ./run_multinode_stage2_training.sh --dry-run ...`
  명령과 voff override, 기존 prepare-inference/evaluation 사용법을 제공한다.
  원본 V=151668 수치는 예시일 뿐 실제 k는 original-base/Stage1 provenance가 검증한
  resized vocabulary를 읽어 resolve한다. CLI는 정수 k만 받는다.

- [ ] **5.1 RED tests 작성.** `test_dil_settings_reach_both_real_worker_commands`는
  existing coordinator builder와 worker shell dry-run JSON을 실행하여 양 machine rank의
  모든 신규 값·topology4·GAS4·beta.5·weight1을 검사한다.
  `test_fresh_profile_overrides_stale_source`는 k/percent/entropy/old beta/GAS/workers를
  잘못 설정한 env를 profile로 reset한 뒤 variant별 resolved 값만 남는지 검사한다.

```python
# 기존 builder의 node_specs 각각에서 실제 worker shell JSON argv를 얻는다.
assert argv[argv.index('--num_processes') + 1] == '4'
assert argv[argv.index('--gradient_accumulation_steps') + 1] == '4'
assert float(argv[argv.index('--step_preference_beta') + 1]) == .5
assert float(argv[argv.index('--step_preference_weight') + 1]) == 1.
assert argv[argv.index('--step_preference_loss_type') + 1] == expected_selector
```

- [ ] **5.2 RED 실행.** `pytest -q tests/multinode/test_dil_forwarding.py tests/step_preference/test_dil_examples.py`.
- [ ] **5.3 옵션 전달 구현.** allowlist→worker env→CLI까지 새 필드를 연결하고 빈
  optional k/P는 argv에서 생략한다. 잘못된 조합은 worker dry-run에서도 명확히 실패한다.
  기존 coordinator rank ownership/preflight/remote path를 변경하지 않는다.
- [ ] **5.4 프로파일·문서 작성.** 명세 section7 모든 fresh defaults, AdamW/linear,
  weight_decay.01/max_grad_norm1, original base/Stage1, A/B allocation 의미를 명시한다.
  eight examples는 common source 블록 뒤 자기 selector/k/mass/percent를 설정한다.
  실제 resized vocabulary는 `stage1_peft_metadata.json`의 `resized_vocab_size`를
  기존 Stage1 metadata loader로 검증해 읽는다. `pretrained_model_vocab_size`나
  `base_vocab_size`를 대신 쓰지 않는다. 모델 준비 시 tokenizer/LM-head와 일치하지
  않으면 실패한다. k 계산만을 위해 model을 로드하지 않는다. 1%/10%는 정수 floor.
  source 비율/resolve k를 dry-run 기록에 남긴다. AGENTS는 canonical batch16 우선과
  충돌하는 역사적 문구만 정리한다.
- [ ] **5.5 GREEN matrix.** 실제 문서의 bash 블록을 추출해 no-network/no-GPU dry-run으로
  실행하는 test를 작성한다. 8개 preset 및 entropy20-voff가 양 노드에 일치하고
  V151668 fixture의 k가1516/15166인지 검사한다. profile은 checkpoint/results 실행
  디렉터리를 생성하지 않아야 한다. path에 공백, stale empty option, resume env도 검사한다.
  신규 두 파일과 기존 `tests/multinode/test_stage2b_forwarding.py`,
  `test_evidence_gain_forwarding.py`, `test_insertion_forwarding.py`를 실행한다.
  metadata fixture에는 `resized_vocab_size=151668`, `pretrained_model_vocab_size=151936`,
  `base_vocab_size=151665`를 함께 넣어 잘못된 필드 선택을 검출한다.
- [ ] **5.6 Commit.** `feat: expose DIL-BCE variants through multinode launchers`.

## Task 6: Comparative evidence, independent review and handoff

**Files:** `tests/step_preference/compare_dil_bce_source.py` 생성;
`results/dil-bce-port-workflow/` evidence, `logs/dil-bce-port-workflow/` logs.
제품 수정이 필요하면 반드시 앞선 소유 task와 선택된 implementer/fixer로 돌린다.

**Interfaces:**
- 비교 도구 CLI: `--source-root PATH --identity-manifest PATH --output PATH`.
  identity가 맞는 source pure modules만 file path로 임시 로드한다. source trainer,
  source dependencies 설치, checkpoint/model 실행은 하지 않는다. ordinary pytest
  collection/runtime는 이 도구를 import하거나 `/raid` 존재를 요구하지 않는다.

- [ ] **6.1 RED comparator test.** `tests/step_preference/test_dil_source_comparison.py` 생성.
  tmp source pure fixture의 hash 변조는 비교 전에 실패; 다른 gradient 결과는
  failure verdict/exit nonzero이며 output에는 input case·max error가 있어야 한다.
  `pytest -q tests/step_preference/test_dil_source_comparison.py`로 RED 관측.
- [ ] **6.2 도구 최소 구현과 GREEN.** seed42의 fixed logits/unequal lengths,
  k5/8/50/1516/15166(각 k보다 큰 V), mass0/1, clean/voff entropy20, 100%,
  boundary quota를 source와 port에 동일하게 입력한다. BCE/top-k mean 및
  selected-step reduction, clean/off gradients, entropy masks를 비교한다.
  source helper의 dataclass module registration을 지원하고 runtime import는 고립한다.
  일반 case 허용오차는 Task1과 동일, 같은 source primitive 비교는 가능한 exact를
  기록한다. 큰 k case는 소수 token만 사용한다. CPU만 사용하고 artifact는 JSON뿐이다.
- [ ] **6.3 실제 source 비교 실행.**
  `./venv/ilvr-venv/bin/python tests/step_preference/compare_dil_bce_source.py --source-root /raid/workspace/temp2/ILVR/.worktrees/ours1-dil-bce --identity-manifest results/dil-bce-port-workflow/source-identity.json --output results/dil-bce-port-workflow/source-comparison.json`.
  Expected exit0와 모든 case pass. identity mismatch는 새 source로 조용히 바꾸지 않는다.
  도구와 테스트를 commit: `test: witness DIL-BCE source objective parity`.
- [ ] **6.4 독립 review.** 선택된 실행 skill의 필수 task/final review를 실행한다.
  reviewer는 implementer와 별개이며 approved spec, 원본 identity/비교, diff/commit,
  gradient·call-order·tail·round-trip 증거를 보고 specification과 quality를 평가한다.
  수정은 선택된 방법으로 수행하고 해당 원인 회귀와 필요한 integration만 재검증한다.
  reviewed-but-unresolved findings를 clean verdict로 바꾸지 않는다.
- [ ] **6.5 최종 command 결정.** verification-before-completion과 적용되는
  finishing-a-development-branch를 읽고 announce한다. code handoff의 full-suite
  command를 아래 하나로 정한다. review 이후 수정이 없는 revision에서 한 번 수행한다.

```bash
set -o pipefail
./venv/ilvr-venv/bin/python -m pytest -q -p no:cacheprovider tests -m 'not gpu' \
  --basetemp="$PWD/checkpoints/dil-bce-port-cpu/final" \
  --junitxml=results/dil-bce-port-workflow/final.xml \
  2>&1 | tee logs/dil-bce-port-workflow/final.log
```

위 명령 전에 Shared Test Setup 환경을 적용한다. model-bearing fixture는
checkpoint tree 아래에 있고 결과와 로그는 분리한다. Expected exit0/failures0;
skip/deselection은 이유를 보고한다. failures가 있으면 finishing Step2로 넘어가지 않는다.

- [ ] **6.6 다른 주장 검증.** `bash -n run_stage2_training.sh scripts/stage2/dil_bce_profile.sh run_multinode_stage2_training.sh`; `git diff --check`.
  source parity가 최종 production revision 이후 유효한지 확인하고 변경되었을 때만
  비교 재실행. 8개 dry-run 증거는 full-suite의 실제 test 실행을 재사용한다.
  새 production 변경 없이 동일 broad suite를 연속 재실행하지 않는다.
- [ ] **6.7 GPU smoke proposal 작성.** 실제 현재 source/profile에 맞춘 command와
  fresh run IDs·inputs/sample IDs·update 수·save/resume·PEFT preparation·소수 evaluation
  명령을 `results/dil-bce-port-workflow/gpu-smoke-proposal.md`에 작성한다.
  BCE/top-k/entropy clean/voff의 실행 coverage, 예상 자원, 네-rank 관측 방법과
  양 node code/environment 일치 preflight를 구체화한다. 아직 실행하지 않는다.
  full TEST400나 전체8개 재학습은 제안에 추가하지 않는다.
- [ ] **6.8 증거·handoff.** actual method/agent IDs/reviews/fixes/commit/hashes/commands를
  workflow record에 기록한다. 코드 검증과 GPU pending을 구분하여 사용자에게 보고하고
  적용되는 finishing 선택지를 제시한다. merge/PR/keep 결정을 대행하지 않는다.
  GPU 승인이 오면 기존 reviewed 코드와 proposal을 사용해 다음 실행 단계로 이어간다.

## Plan Self-Review and Execution Selection

명세 section1–3은 Tasks2/5, section4–5는 Tasks1/3, section6은 Task4,
section7은 Task5, section8은 Tasks3/4/6, section9의 변경 경계는 전체 file map에
대응한다. Review Focus 다섯 항목의 테스트 소유자를 위에 지정했다.
계획의 signature/필드명과 source 비교 wrapper, sidecar token 결과, identity schema를
대조했다. 승인된 범위를 넘어서는 source checkpoint migration이나 GPU 실행은 없다.

계획 검토와 구현 방법 선택은 대기 상태다. 권장 방식은 **Subagent-driven**:
수식·checkpoint autograd·resume·양 노드 전달을 별도 검토 가능한 여섯 작업으로
나누었고 잘못된 gradient를 학습 후에 발견하면 실험 비용이 크기 때문이다.
대안 **Native**는 한 agent가 구현하고 마지막 독립 whole-branch review를 받는다.
사용자의 명시적인 선택 전에는 둘 중 어느 것도 시작하지 않는다.
