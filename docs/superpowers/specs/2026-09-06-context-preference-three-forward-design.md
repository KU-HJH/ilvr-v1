# Stage 2B Context Preference + Entropy — 3-forward correction

Status: 승인된 design commitment v1의 서면 명세. 서면 검토 대기이며 implementation plan은 아직 작성하지 않았다.
작업 위치: `/root/workspace/ILVR/.worktrees/ours1`.
코드 조사 기준: `a69b9c0`.
Understood as: 기존 ours1 Stage 2B의 shared-prefix 계산을 보존하면서 새 기본 학습 경로를 clean target의 local context preference로 교정한다. 계획 전 모호한 사항은 사용자에게 확인하며, spec 검토 → plan → 구현 순서를 따른다.

## 1. 근거와 확정된 결정

기준 문서는 primary checkout의
`/root/workspace/ILVR/documents/TODO-ours1-20260906.md`이다.
이 명세는 해당 TODO와 다음 사용자 확인을 함께 적용한다.

- 기존 ILVR Stage 2의 clean latent 생성은 유지하고 3-forward 제한에서 제외한다.
- preference margin `gamma_P=0`.
- entropy는 각 step 내부에서 토큰 평균을 구한 뒤 V/T 모든 step을 동일 가중치로 평균한다.
- 채팅의 design commitment v1은 사용자의 “진행해”로 승인되었다.
- 여러 구성요소에 걸친 변경을 bounded로 분류했던 절차를 바로잡아 이 서면 spec과 후속 plan을 작성한다. 이는 승인된 학습 방법이나 작업 범위를 변경하지 않는다.

이전 `2026-09-06-shared-prefix-sidecars-design.md`는 historical legacy DPO 설계다.
그 문서의 rejected/reference 유지 및 sidecar chunking 허용은 새 기본 경로의 요구사항이 아니다.

## 2. 문제, 범위와 접근

현재 기본 Stage 2B는 negative metadata와 reference score를 요구한다.
shared-prefix 경로도 rejected branch를 구성하고, query budget에 따라 decoder 호출을 나눈다.
따라서 기존 logical group 수나 바깥 policy 호출 수만으로 새 요구사항 준수를 주장할 수 없다.

선택한 접근은 기존 differentiable shared-prefix KV와 V/T suffix replay를 교정하는 것이다.
이미 구현된 clean CE, ReasoningStepSpan, attention mask, mRoPE, entropy 계산을 재사용하며,
모듈을 삭제하고 다시 구현하거나 외부 trainer로 교체하지 않는다.

대안과 배제 이유:
- step별 full-prefix 재실행은 clean-prefix 공유와 forward 상한을 위반하므로 채택하지 않는다.
- 기존 monolithic tree 경로는 legacy 비교 구현으로 보존할 수 있지만,
  승인된 clean/V/T grouped replay를 교체하는 추가 구조 변경은 이번 범위에 넣지 않는다.

포함: 실행 진입점부터 설정 검증, 데이터 준비, collator, trainer, 모델 scoring까지의 기본 경로 교정과 correctness 검증.
제외: 본 학습 캠페인, 방법론의 성능 향상 주장, 새 cache/provenance/recovery 시스템, Stage 1/ordinary Stage 2 알고리즘 변경.

## 3. Forward와 rollout의 정확한 계측 경계

실제 바깥 policy forward와 내부 language decoder forward를 각각 계측한다.
상한은 wrapper 호출 수만으로 판정하지 않고, 기존 clean latent 생성이 끝난 이후의
실제 scoring decoder invocation에 적용한다.

| 구분 | 상한 포함 | 처리 |
| --- | --- | --- |
| 기존 clean latent 생성 decoder | 아니오 | 생성 방식과 gradient를 보존하고 호출 수를 별도 보고 |
| 전체 clean 최종 scoring decoder | 예 | 1회; clean CE, 모든 target logits, 공유 KV 확보 |
| 모든 step의 V-off scoring decoder | 예 | 하나의 그룹으로 최대 1회 |
| 모든 step의 T-off scoring decoder | 예 | 하나의 그룹으로 최대 1회 |
| backward checkpoint recomputation | 아니오 | forward와 분리해 계측 |
| reference forward / negative generation / preference rollout | 허용하지 않음 | 새 기본 경로에서 각각 0 |

K=1/4/8/16 모두 scoring decoder 호출은 최대 3회다.
모든 유효 step을 포함하고 random step sampling, modality group chunking,
추가 score/entropy/cache-preparation forward를 금지한다.
유효 step이 없는 입력은 기존 지원 범위에서 clean CE만 계산할 수 있으며 상한은 여전히 적용한다.
현재 collator와 span mapper의 local microbatch size=1 제약을 유지한다.
서로 다른 example을 한 microbatch에 넣는 기능이나 example 간 새 가중 방식은 추가하지 않는다.
기존 gradient accumulation과 distributed training의 batch 처리 의미는 보존한다.
유효 step이 0개라면 preference/entropy loss와 해당 step 통계는 0,
scored target token 수는 0으로 보고하고 sidecar는 실행하지 않는다. Clean CE는 그대로 계산한다.

기존 clean latent 생성만 제외한다. sidecar를 latent 생성으로 분류하거나,
branch별 clean latent 재생성을 추가하는 방식으로 호출 수를 숨기지 않는다.
메모리 부족 시 blocker와 사용한 입력 크기를 보고한다.
명시적인 batch/sequence 설정 축소는 가능하지만 sidecar chunking이나 조용한 step 제거로 우회하지 않는다.

## 4. 데이터와 local context 의미

각 transition은 `T_(k-1) → Z_k → T_k`다.
세 context 모두 같은 clean target T_k와 같은 teacher-forced target prefix를 사용한다.

- Clean: question, input image, 이전 clean history, T_(k-1), Z_k.
- V-off: 현재 T_k의 prediction query에서 현재 Z_k의 KV 접근만 차단.
- T-off: 현재 T_k의 prediction query에서 바로 이전 T_(k-1)의 KV 접근만 차단.

이전 clean history는 유지한다. 모든 latent를 동시에 손상시키는 global ablation은 하지 않는다.
Sidecar 결과는 clean trajectory, 다음 step, 다른 sidecar에 전파하지 않는다.
기존 clean latent나 ancestor에 이미 매개된 정보까지 제거하는 total causal intervention은 목표가 아니다.

ReasoningStepSpan과 기존 target/label construction을 재사용한다.
Target span이 [s,e)이면 실제 scoring query는 [s-1,e-1)이다.
첫 target token의 predictor인 latent-end 등 boundary input도 sidecar 안에서 재계산한다.
Control token은 predictor input으로 사용할 수 있지만 preference/entropy의 target token에는 넣지 않는다.
기존 source 준비의 require_reasoning_units=True를 유지한다. 각 latent unit에는
직전/직후의 비어 있지 않은 text block이 있어야 하며, 첫 latent도 예외가 아니다.
이 조건을 만족하지 않는 latent를 T-off 대상에서 조용히 빼지 않고 기존 validation error로 거부한다.
Token span은 순서가 맞는 비어 있지 않은 범위여야 하고 step ID는 0부터 연속적이어야 한다.
Latent unit 자체가 없는 clean 입력의 빈 step 목록과, 잘못 구성된 latent transition은 구별한다.

## 5. Shared-prefix 및 gradient 계약

Clean 최종 decoder가 만든 layer별 KV를 동일 microbatch의 모든 local sidecar가 공유한다.
각 sidecar는 자신의 replay 시작점보다 앞선 clean ancestor와 자기 branch의 causal 과거만 볼 수 있다.
Clean future와 다른 sidecar는 볼 수 없다.

기존 suffix 입력 구성, dense/Flex visibility, logical mRoPE 위치 처리를 재사용한다.
Boundary predictor는 clean hidden/logit을 복사해 대신하지 않는다.
공유 KV와 clean latent의 autograd 연결을 보존하며, 편의를 위한 detach나 inference cache 전환을 금지한다.
Forward에서 저장한 tensor를 backward에서 재사용하는 기존 checkpoint 의미도 보존한다.

## 6. Objective와 reduction

각 context의 block score는 T_k target token log-prob의 평균이다.
모달리티 m에 대해 `delta_k^m = s_k^clean - s_k^off_m`으로 정의한다.

`pref_k^m = -logsigmoid(beta * (delta_k^m - gamma_P))`

각 step에 sigmoid를 적용한 뒤 V/T의 2K개 loss를 동일 가중치로 평균한다.
Margin을 먼저 합산하거나 평균한 뒤 sigmoid를 적용하지 않는다.
`dpo_loss.py`에 `context_preference_loss(clean_logps, ablated_logps, beta, margin)`을 추가한다.
이 함수는 동일 shape의 step score들에 개별 식을 적용한 후 평균한 scalar를 반환하고 양쪽 gradient를 유지한다.

`entropy_k^m = mean_j relu(gamma_H - (H_off[k,j] - H_clean[k,j]))`

Entropy 역시 step 내부 토큰 평균 후 V/T의 2K개 값을 동일 가중치로 평균한다.
긴 step에 더 큰 가중치를 주는 전체 토큰 평균은 새 objective에 사용하지 않는다.
모든 target token에서 전체 vocabulary entropy를 계산하며 첫 몇 token만 고르는 근사는 사용하지 않는다.

`total = clean_CE + lambda_P * context_preference + lambda_H * entropy`

초기 기본값: beta=0.1, gamma_P=0, lambda_P=1, lambda_H=0.1, gamma_H=0.1.
Clean CE 계수는 1이다. 기존 clean CE의 supervised label construction을 보존한다.
CE, preference, entropy의 prediction source는 같은 clean/V/T 세 종류의 logits뿐이다.
Preference와 entropy를 위해 별도의 forward를 추가하지 않는다.

## 7. 기본 경로와 파일 책임

| 기존 파일 | 교정 책임 |
| --- | --- |
| run_stage2_training.sh | 새 기본 Stage 2B 실행에서 negative 파일 요구/전달 의존성 제거; preference 설정 연결 |
| src/utils_deepseed.py, src/step_preference/config.py | context preference 설정과 검증; 기존 Stage 2A 초기화 유지; 기본 경로의 legacy 의존성 제거 |
| src/main.py | source/checkpoint 검증을 유지하면서 기본 preflight의 negative manifest/store/reference score 준비를 사용 중단 |
| src/stage2/data.py | 기존 span mapping을 보존하고 negative join 없이 trainer가 사용할 step metadata 전달 |
| src/step_preference/sidecar.py | 기존 local chosen replay를 재사용해 V/T target branch만 구성 |
| transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py | modality마다 전체 step을 한 decoder 호출로 실행; 기존 prefix gradient/position 처리 재사용 |
| src/stage2/trainer.py | 새 기본 경로의 clean/V/T score 및 entropy를 재사용해 objective와 로그 구성 |
| src/step_preference/dpo_loss.py | context_preference_loss 추가, 기존 helper 보존 |
| tests/step_preference, tests/stage2 | 아래 계약의 실패 증거와 runtime 검증 |

기존 Stage 2A checkpoint는 policy 초기화 용도로 유지한다. Reference model로 로드하지 않는다.
새 기본 경로는 negative cache/loader, reference log-prob cache, rejected scoring,
online/offline autoregressive preference rollout을 호출하지 않는다.
Legacy 파일이나 실험 helper의 보존은 허용하지만, negative 인자의 존재 등을 이유로
새 기본 경로가 자동으로 legacy 계산으로 바뀌어서는 안 된다.
명시적인 legacy 실험 선택을 남기는 경우에만 그 경로에서 과거 동작을 유지한다.

로그에는 CE/context preference/V 및 T별 preference/entropy, context별 log-prob와 entropy,
실제 forward 계측 범위와 결과를 구분해 남긴다.
새 objective를 DPO loss라는 이름으로 보고하지 않는다.

## 8. 검증과 완료 조건

첫 검증은 negative artifact 없이 기본 설정 → preflight → collator → trainer를 통과하는 작은 통합 경로다.
Legacy loader, generation, reference, rejected entrypoint를 호출 즉시 실패하도록 감시한다.
Model forward/backward가 finite이고 실제 scoring decoder 수가 최대 3인지 확인한다.
무거운 model loading만 작은 모델로 대체할 수 있으며, 이를 실제 pretrained/full-training 검증으로 표현하지 않는다.

필수 witness:
- K=1/4/8/16에서 실제 policy/decoder entrypoint hook 계측; clean latent 및 backward recomputation 별도 보고.
- V-off와 T-off가 첫 target predictor부터 해당 local KV 접근만 차단하는 runtime 비교.
- Sidecar의 clean ancestor identity, branch 간 격리, clean future 차단, no propagation.
- Clean/V/T target 및 teacher-forced prefix identity; 세 logits의 CE/preference/entropy 공동 사용.
- 서로 다른 target 길이를 가진 step으로 step 균등 reduction 검증.
- Planner가 생성한 mask를 정답으로 재사용하지 않는 독립 dense oracle과 loss/gradient 비교.
- FP32/dropout-off에서 기존 프로젝트 수준의 atol=1e-5, rtol=1e-4로 수치 비교하고 checkpointing off/on을 검사.
- 기존 dense/Flex 실행 및 PEFT 등 실제 지원 경로에 맞춘 작은 integration check; 미실행 환경은 명시.
- step_preference.enabled=false에서 Stage 2의 forward/loss 동작 보존.

작은 수치 검증은 stochastic dropout 경로의 완전 동등성이나 긴 실입력 메모리 적합성을 증명하지 않는다.
최종 검증의 구체 명령과 요구사항별 연결은 후속 plan에 기록한다.
중복 broad test/review 반복은 하지 않는다.

완료 보고는 TODO 15절을 따른다: 변경 파일, 재사용 구현, legacy 비호출 증거,
K별 호출 수, log-prob/entropy sanity, first-token/locality/no-propagation/baseline 결과.
새 경로에서 reference forward와 negative rollout이 각각 0이고 scoring decoder가 최대 3이어야 완료다.
이 명세의 3~8절은 원문 TODO의 연산/테스트/완료 계약을 명문화한 것이며,
원문 13절 A~H의 추가 항목을 생략하는 축약 보고가 아니다.

## 9. Design commitment v1 검토 기록

동결 proposal v1에서 commitment v1으로 학습 방법/범위 변경은 없었다.
이 문서는 승인된 commitment의 서면 표현이며, 구현 성공 보고가 아니다.

- hate의 root: trainer만 고치면 상위 기본 실행 경로에 negative/reference 준비 의존성이 남을 수 있다.
- first_nail: negative artifact가 없는 작은 기본 경로 통합 실행에서 legacy 호출을 trap하고 finite forward/backward 및 scoring 호출 상한을 검증한다.
- first_nail 상태: 새 경로가 미구현이므로 아직 통과하지 않았다. 아래 실패 테스트는 현재 의존성을 보여주는 RED 증거이며 완료 증거가 아니다.
- macrothink: 방법과 범위가 TODO와 사용자 확인으로 지정되어 있어 방향 재탐색의 추가 비용을 정당화하지 못하므로 생략했다.
- prism: 실행 경로 도달성과 수치/gradient 정확성 관점 모두 runtime witness 전에는 unclear이다. 서로 상충하는 설계 권고는 없었으며, 의견 일치는 검증 증거가 아니다.
- 미검증 전제: 새 경로의 end-to-end 도달성, loss/gradient 정확성, chunking 없는 긴 실입력 메모리 적합성.
- 위험 면제: 없음. 실패한 완료 조건을 승인으로 대체하지 않는다.

서면 shower 검토는 문서 사본 전체를 읽은 별도 context-free reviewer가 수행했다.
판정은 minor gaps였으며, local batch size=1, 빈 step의 loss/로그,
유효 transition 기준과 원문 요구사항의 포함 범위를 본문에 명시해 보완했다.
Batch/transition 기준은 기존 collator, span mapper, planner 코드로 확인했다.
이는 이해 가능성 보완이며 학습 방법/범위의 변경이나 runtime correctness 증거가 아니다.

## 10. 현재 증거와 다음 단계

기존 코드 진단 결과(K=1/4/8/16, tiny CPU Qwen, step당 latent slot 2개):
전체 decoder 호출은 5/11/19/35, 그중 clean latent 생성은 2/8/16/32였다.
따라서 제외 후 scoring 호출은 각각 3이었다.
이는 legacy rejected/reference가 포함된 기존 코드의 계측이며 새 objective 준수 증거가 아니다.
결과: `results/contextpref-preplanning-20260906-01/call-audit.json`.

관련 baseline 36개 통과:
`logs/contextpref-implementation-20260906-01/baseline.log`.
새 테스트 6개는 context preference 함수 부재와 negative 필수 설정 때문에 실패:
`logs/contextpref-implementation-20260906-01/red-core.log`.
추가된 파일은 `tests/step_preference/test_context_preference.py`이며 production 코드 변경은 아직 없다.

작업 순서:
서면 spec 검토 → writing-plans → 승인 범위의 subagent-driven-development →
TDD와 최소 교정 → requesting-code-review → verification-before-completion → 해당 시 branch handoff.
TODO 14절의 receiving-code-review/systematic-debugging 조사 결과와 기존 RED 증거를 이어 사용하며,
새 방법을 설계하거나 root cause 확인 없이 재구현하지 않는다.

Python과 테스트는 worktree의 `./venv/ilvr-venv`만 사용한다.
결과는 worktree `results/`, 로그는 `logs/`, model-bearing artifact는 기존 `checkpoints/` symlink 아래에만 둔다.
본 학습 실행은 이번 구현 검증에 포함하지 않는다.
