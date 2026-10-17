# GT-conditioned DIL-BCE: shared OFF Top-k union GT

Status: user authorized writing-plans after presentation of this written spec
at `1c3eae3` by explicitly invoking `$superpowers:writing-plans`. This approves
the transition to planning, not implementation. Written-plan review and
execution-method selection remain required.

## 1. 목적과 승인된 의도

사용자가 원하는 rejected는 OFF 분포의 top-k 전체에 대한 가중 loss가 아니라,
`OFF Top-k ∪ GT` 안에서 정규화한 GT 확률이다. Rejected와 mass의 후보 집합은
위치별로 한 번 선정한 동일한 token ID 집합이어야 한다. Clean에서 top-k를
별도로 뽑지 않는다. 기존 top-k 및 reverse 구현은 옵션으로 보존한다.

사용자는 `유지로 진행해`로 token log-probability 평균 후 softplus 순서를
선택했고, 두 mass와 rejected의 동일 집합 사용 설명에 `그렇게 진행해라.`로
진행을 승인했다. Mass 합집합은 이 의도의 일부이며 별도 실험 제안이 아니다.

성공 기준은 수식뿐 아니라 실제 trainer → shared-prefix sidecar → token
통계 → sample reduction 경로가 아래 loss와 gradient를 계산하고, CLI 및
resume에서도 다른 목적함수로 오인되지 않는 것이다. 학습 launch, 기존 run의
설정 변경, 결과 개선 주장, merge/push는 이번 구현 범위가 아니다.

## 2. 선택지와 채택 방향

1. **새 selector 추가 — 채택:** `dil_bce_rf_topk_gt`로 새 의미를 명시한다.
   기존 selector 및 저장된 formula identity는 변경하지 않는다.
2. 기존 `dil_bce_rf_topk`를 교체하면 역사적 checkpoint와 같은 이름의 loss가
   달라진다. 기존 구현 보존 요구에 어긋나므로 채택하지 않는다.
3. rejected와 mass support를 각각 임의 조합하는 추가 flags는 승인된 목적에
   필요하지 않고 잘못된 집합 조합을 허용하므로 추가하지 않는다.

기반 코드: `.worktrees/dil-loss-defaults-20261001`, revision
`2f48c51985c395c16bea80a89847d7586728f9cd`. 다른 worktree를 사용하게 되면 이
기반의 알고리즘/계수 수정을 포함해야 한다.

## 3. 위치별 support와 gradient 계약

Target 위치 t의 GT ID를 y_t, 전체 output vocabulary를 V, OFF/clean logits를
각각 z^O_t, z^C_t라 한다. 기존 next-token predictor/target 정렬을 유지한다.

```
K_t = TopK(stopgrad(z^O_t), k).indices
S_t = unique(K_t union {y_t})
ell_t = z^O_t[y_t] - logsumexp(z^O_t[S_t])
q_GT,t = exp(ell_t)
```

- GT가 K_t 안에 있으면 |S_t|=k, 밖이면 |S_t|=k+1이다. GT 때문에 기존
  top-k 후보를 버리거나 GT를 중복해서 분모에 더하지 않는다.
- S_t를 rejected 분모, OFF mass, clean mass가 공유한다. OFF와 clean의
  top-k가 달라도 clean target은 S_t에 대해서만 계산한다.
- 인덱스 선택만 gradient가 없고, 선택된 OFF logits의 분자와 분모는 live다.
  GT 확률을 다시 가중치로 곱하지 않으며 top-1이나 후보별 BCE 평균도 아니다.
- 이 rejected는 GT의 집합 내부 비율을 낮춘다. OFF 전체 확률에서 집합 mass가
  바뀔 수 있으므로 GT 절대확률 감소를 매 update의 보장으로 주장하지 않는다.

## 4. Loss와 reduction

J는 sample의 원래 eligible step 수, T_j는 기존 entropy selection을 통과한
step j의 target 위치 집합이다. c_t는 clean full-vocabulary GT log-probability다.

```
positive_j = softplus(-beta * mean_{t in T_j}(c_t))
negative_j = softplus( beta * mean_{t in T_j}(ell_t))

p^C_t = softmax(z^C_t over full V)
p^O_t = softmax(z^O_t over full V)
a_t = stopgrad(sum_{v in S_t} p^C_t[v])
b_t = sum_{v in S_t} p^O_t[v]
mass_t = a_t*log(a_t/b_t) + (1-a_t)*log((1-a_t)/(1-b_t))
mass_j = mean_{t in T_j}(mass_t)

L_sample = lambda_CE * clean_CE
         + lambda_P * sum_j(positive_j + negative_j) / J
         + lambda_M * sum_j(mass_j) / J
```

- T_j가 비면 그 step의 세 항은 모두 0이다. `softplus(0)`를 넣지 않는다.
  분모 J에서 그 step을 제거하지 않는다. J=0이면 preference/mass는 0이다.
- Chosen은 전체 어휘 기준 clean GT 확률을 높이는 기존 original DIL 집계다.
  Reverse selector의 tokenwise positive를 가져오지 않는다.
- Mass는 `KL(Bernoulli(clean detached) || Bernoulli(OFF live))`다. 집합 전체의
  확률을 clean에 맞추고 내부 분배를 직접 지정하지 않는다. OFF의 집합 안과
  바깥 logits 모두 full-vocabulary mass 계산에 필요한 gradient를 유지한다.
- Lambda_M은 lambda_P와 독립이다. Lambda_P=0이어도 lambda_M>0이면 mass가
  작동한다. 두 항은 공유 logits/parameter에서 상호작용할 수 있다.
- 선택 source(clean 또는 voff)의 full-vocabulary entropy, sample 단위 stable
  top-percent 선택, 원래 step 분모, sampler/padding/partial accumulation은
  기존 동작을 유지한다. GT-union entropy로 선택 기준을 바꾸지 않는다.

## 5. 수치 안정성과 입력 검증

기존 `2 <= k < |V|`, finite beta>0, finite nonnegative weights, target ID,
shape/device, nonempty token input, 실제 output-head vocabulary 검증을 유지한다.
BF16 입력도 loss 통계는 FP32에서 계산하고 logsumexp/log-softmax를 사용한다.
확률이 underflow한 뒤 log를 취하는 방식으로 ell_t를 만들지 않는다.

Mass는 기존 안정적인 selected/complement group 계산을 재사용할 수 있지만,
S_t를 기준으로 구성해야 한다. `k=|V|-1`이고 GT가 K_t 밖이면 S_t=V가 된다.
이 경우 두 mass는 정확히 1, KL과 그 gradient는 0이다. Empty complement를
기존 `_group_gap`에 전달하지 않는다. 하나의 batch 안에서 full support와
partial support가 섞여도 독립적으로 처리한다. GT-union 때문에 k 허용 범위를
임의로 좁히거나 기존 옵션의 수치 처리를 바꾸지 않는다.

## 6. 코드 경계와 데이터 흐름

- `src/step_preference/dil_topk_loss.py`: 기존 primitive를 보존하고 GT 방식의
  위치별 통계를 추가한다. 기존 q-weighted loss와 새 ell_t는 의미가 다르므로
  별도 명명된 GT 통계(`gt_logq`)로 전달한다. Existing `negative_loss` 필드에
  log-probability를 숨겨 넣지 않는다. Support 선정은 같은 호출에서 공유한다.
- `transformers/.../qwen2_5_vl/stage2_shared_prefix.py`: selector 검증과
  checkpoint-local scoring dispatch를 추가한다. GT token IDs 및 clean/OFF
  위치 정렬을 유지하고 필요한 scalar token 통계만 반환한다. 전체 vocabulary
  logits를 step reduction까지 추가로 보관하는 경로를 만들지 않는다.
- `src/stage2/trainer.py`, `src/step_preference/dil_reduction.py`: 새 통계의
  전달과 위 reduction을 연결한다. 기존 full-vocabulary GT 통계를 conditional
  GT score로 오해해서 사용하지 않는다. 기존 DPO/DIL 경로는 그대로 보존한다.
- `config.py`, `dil_config.py`, `utils_deepseed.py`, `run_stage2_training.sh`,
  coordinator forwarding 및 DIL resume/planned-stop dispatch에 새 selector를
  연결한다. 새 launcher나 별도 실행 framework는 만들지 않는다.
- `dil_identity.py`: `dil_bce_rf_topk_gt_v1` formula identity를 추가한다.
  같은 selector의 identity roundtrip과 resume은 지원하고, 기존 selector와
  새 selector 간 resume은 optimizer 복원 전에 거부한다.
- `docs/dil-bce.md`: 선택법, 수식, 같은 support 계약, 기존 옵션과의 차이를
  기록한다. 기본 selector를 바꾸지 않는다.

## 7. Coefficients, diagnostics, compatibility

새 DIL 실행에서 beta=1, margin=0, preference weight=1, CE weight=0.1의 standing
defaults를 적용한다. CE는 한 번만 가중한다. 기존 run/resume 계수는 보존한다.
Mass는 기존 `--step_preference_lambda_topk_mass` / 환경 변수로 설정하며 global
fallback(현재 0)을 이 기능 변경만으로 바꾸지 않는다. Mass가 켜진 예시에서는
명시적으로 1을 전달한다. 실제 학습 설정과 launch 승인은 별도 작업이다.

`loss/dil_negative`는 실제 step negative를, `loss/topk_mass`는 실제 union mass
KL을 기록한다. `stats/logp_visual_off`의 기존 full-vocabulary GT 의미를
유지하고 conditional GT score는 `stats/topk_gt_logq`로 기록한다. 이 metric은
선택 token 평균 후 원래 step 수로 평균하며 빈 step은 0으로 기여한다.
GT coverage는 union 이후 항상 1인 값이 아니라 원래 K_t에 GT가 있었는지를
뜻한다. 새 selector의 q entropy와 mass diagnostics는 S_t 기준임을 명시한다.

구현이 수식을 따른다는 검증과 multi-node 학습/GPU 실행 또는 성능 개선 증거는
구별한다. 이 작업의 CPU 검증을 새 training run의 검증으로 보고하지 않는다.

## 8. 검증 계약

TDD로 각 동작의 실패를 먼저 확인하고 구현한다. 독립 수치 oracle과 실제
tiny-model trainer 경로를 사용하며, 선언된 selector 목록만으로 실행을 추정하지
않는다. 다음 인과관계를 검증한다.

| 검증 대상 | 잡아야 할 잘못된 구현 |
| --- | --- |
| GT가 top-k 안/밖에 있는 mixed rows | 중복 GT, GT 누락, 후보 교체, 고정 k+1 중복 분모 |
| Clean/OFF top-k가 의도적으로 다른 입력 | clean에서 support 재선정, mass에 옛 K만 사용 |
| 직접 수식과 loss/gradient 비교 | q 가중 기대값 재사용, GT/분모 detach, clean mass live |
| 선택 token 수가 다른 steps와 empty selection | tokenwise softplus, token-weighted steps, 분모 변경 |
| 독립 lambda_P/lambda_M 및 CE | mass를 preference weight로 재가중, CE 이중 가중 |
| S=V와 partial support 혼합, extreme/BF16 | empty-complement NaN, underflow log, 불안정한 KL |
| 실제 tiny trainer/sidecar 및 checkpoint 재계산 | 기존 loss dispatch, 잘못된 scalar unpacking/target 정렬 |
| CLI/coordinator defaults와 resume | 새 selector 누락, beta10 오적용, cross-objective resume |
| 기존 original/reverse 및 DPO 회귀 | 기존 수식, gradient, formula identity 의미 변경 |

구현 후 독립 specification/quality review를 받고 결함을 수정한다. 최종 검증은
reviewed revision에 대해 repository venv의 project full-suite command를 한 번
실행하여 verification/branch-finishing gate에 공통 사용한다. 추가 증거는 별개
주장을 입증하는 최소한의 검사만 수행한다. 실패/skip과 GPU·학습 미실행 범위를
그대로 보고하며, suite 통과만으로 수식 요구 충족을 대신하지 않는다.

## 9. Workflow와 다음 gate

현재: 대화에서 승인된 설계 → written spec self-review → 사용자 writing-plans
호출로 계획 단계 진행 승인. `superpowers:writing-plans`로 구체적 구현 plan을
작성한다. 그 plan의 검토 및 사용자 구현 방식 선택 후에
TDD/구현/독립 review/verification/branch finishing을 진행한다.

직접 구현 또는 subagent 구현 방식은 아직 선택되지 않았다. Spec 작성은
controller control artifact이며 이 선택을 대신하지 않는다. 승인/실행/agent/
revision/verification 증거는 기존 task 기록에 이어 남긴다.
