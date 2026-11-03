# 이미지 없는 V-off에서 분기하는 preference — 설계 검토본

상태: **design proposal v1 (1–6절 동결), design commitment v1은 8절**. 7절은 중간 검토 당시의 기록이며, 그때의 OFF latent 질문은 이후 사용자 정정으로 해소되었다. 계산 흐름의 확인과 최종 문서에 대한 구현 승인을 구분한다. 코드는 아직 변경하지 않았다.

승인 기록: 최종 문서 제시 후 사용자가 “진행해라”로 design commitment v1의 구현을 승인했다. 이제 구현·집중 검증을 진행한다. 학습 시작 승인은 포함하지 않는다.

Understood as: 이미지 입력과 이미지에 조건화된 latent/KV를 받지 않은 V-off를 실제 실행하고, 그 실행에서 ON을 분기한다. `independent`는 매번 원래 V-off에서 시작하고, `cumulative`는 앞선 ON의 이미지 삽입과 계산 이력을 이어간다. 기존 `clean_prefix` 등은 보존한다.

## 1. 무엇을 바로잡는가

| 항목 | 기존 `v_off_prefix` | 새 `v_off_insertion` 제안 |
|---|---|---|
| preference의 출발점 | clean 실행으로 만든 latent를 포함한 입력 재계산 | 이미지 없이 별도로 실행한 V-off |
| OFF의 이미지 유래 latent | 남아 있을 수 있음 | 받지 않음 |
| ON의 latent | clean latent 재사용 | 삽입한 이미지와 해당 분기의 문맥으로 새로 생성 |
| 분기 이전 KV | clean KV는 재사용하지 않지만 clean latent는 재사용 | 실제 V-off 실행에서 얻은 KV만 사용 |
| 기존 구현과 결과 | 그대로 보존 | 별도 옵션·별도 실험으로 구분 |

기존 구현 확인 지점: `modeling_qwen2_5_vl.py`의 `_forward_stage2`는 `rollout.final_inputs_embeds`를 기존 sidecar scorer에 전달한다. `stage2_shared_prefix.py`의 evidence scorer가 이를 사용한다. 이는 정적 호출 경로 확인이며 새 구현의 실행 증거는 아니다.

## 2. 실제 입력은 어떻게 구성하는가

표기: `Q`는 기존 시스템·사용자 메시지의 **텍스트 토큰**, `T_i`는 데이터의 reasoning 텍스트, `S/E`는 `<latent_start>/<latent_end>`, `I`는 해당 샘플의 원본 `image_input`을 Stage2와 같은 방식으로 인코딩한 이미지 토큰 블록, `Z_i`는 모델이 생성하는 연속 latent이다. 여러 원본 이미지가 있으면 기존 순서의 전체 이미지 블록을 사용한다.

두 번째 시점까지의 구체적인 입력 예시는 다음과 같다. 각 행의 마지막 `T_2`에서 동일한 target token ID들의 예측 확률을 비교한다.

| 조건 | 두 번째 시점까지의 입력 |
|---|---|
| OFF | `Q T_0 S E T_1 S E T_2` |
| ON, `independent` | `Q T_0 S E T_1 I S Z_2 E T_2` |
| ON, `cumulative` | `Q T_0 I S Z_1 E T_1 I S Z_2 E T_2` |

- OFF에서는 초기 원본 이미지 블록의 `vision_start/image_pad/vision_end`와 모든 latent pad 슬롯을 제거한다. `S/E`와 기존 텍스트 토큰은 유지한다. 빈 `S E` 쌍에서 latent를 생성하지 않는다.
- ON에서는 대상 시점의 `S` **직전**에 원본 이미지 블록을 삽입하고, 그 `S/E` 사이에 기존 `latent_size`만큼 latent를 생성한다. 현재 구성의 8을 새로 하드코딩하지 않는다.
- `independent`는 이전의 이미지 삽입, 이전 ON latent, 이전 ON KV를 다음 분기에 넘기지 않는다. 매 시점 OFF의 해당 위치 이전 KV에서 새로 시작한다.
- `cumulative`는 최초 삽입 전까지의 OFF KV에서 시작한 하나의 ON 이력을 계속 계산한다. 앞서 삽입한 이미지, 생성한 latent, 그에 조건화된 텍스트 KV를 유지한다. OFF에는 이 이력이 역으로 들어가지 않는다.
- ON의 이미지 접근은 latent 생성 때만 허용하는 것이 아니다. 삽입 뒤 텍스트 채점에서도 일반적인 causal attention으로 접근한다. **추가 이미지 차단은 넣지 않는다.** 따라서 이 실험은 latent만의 기여를 분리하는 실험이 아니다.
- 이미지 없는 OFF와 ON은 기존 데이터의 동일한 텍스트를 teacher forcing한다. 텍스트를 새로 샘플링하거나 이미지 설명을 텍스트에서 삭제하지 않는다. 여기서 “이미지 없음”은 시각 입력과 그 계산 이력이 없다는 뜻이지, 주어진 정답 텍스트가 이미지 내용을 언급하지 않는다는 뜻이 아니다.

**Stage2 대비 명시적인 변경:** 기존에는 원본 이미지가 초기 사용자 메시지에 들어간다. 새 preference 경로는 그 이미지를 OFF에서 빼고, ON에서는 중간 삽입 지점에 넣는다. 이 입력 배치 변경과 independent/cumulative 분기 외에 latent 생성법·텍스트 지도·CE 경로를 바꾸지 않는다. `helper_image`를 직접 넣거나 별도 pooling/선택기를 만들지 않는다.

## 3. 실행 경로와 책임 분리

권장안은 기존 differentiable KV 분기와 Stage2 latent rollout을 재사용하는 별도 scorer이다. 처음부터 각 ON 전체를 다시 계산하는 방식은 테스트 기준 구현으로만 사용한다. clean 실행을 먼저 만들고 일부를 가리는 방식은 요구한 V-off 출발점이 아니므로 새 경로의 대안으로 채택하지 않는다.

1. **입력 계획기:** 원본 token ID와 기존 step/target 매핑에서 OFF 토큰 배열 및 ON 삽입 위치를 만든다. 토큰 삭제·삽입 후 target ID와 새 predictor 위치를 명시적으로 다시 매핑한다. 토큰 문자열을 다시 토크나이즈해서 target을 바꾸지 않는다.
2. **OFF 실행:** OFF token ID를 embedding lookup하고 일반 decoder를 실제 호출한다. 이미지 feature, clean rollout embedding, clean KV를 인자로 받지 않는다. OFF target 점수와 differentiable KV를 얻는다. 모든 시점의 점수는 이 이미지 없는 causal 실행에서 얻는다.
3. **ON 실행:** 실제 OFF KV를 현재 `S` 전까지 잘라 사용한다. `independent`는 시점별 분기, `cumulative`는 첫 삽입부터 이어지는 한 분기이다. 이미지 삽입과 branch-local latent 생성 이후의 텍스트는 ON에서 계산한다. 미래 target 및 미래 이미지의 KV를 prefix로 넘기지 않는다.
4. **latent 생성:** 기존 `LatentRolloutEngine.rollout`의 “앞 문맥의 마지막 hidden state를 다음 latent embedding으로 사용”하는 recurrence를 그대로 호출한다. KV prefix를 묶은 decoder callback이 suffix용 직사각형 causal mask를 전달한다. 새 detach/no_grad를 넣지 않는다.
5. **위치 번호:** OFF는 이미지 없는 배열로, ON은 실제 삽입된 이미지 배열과 `image_grid_thw`로 위치를 다시 계산한다. ON 분기 이전 위치는 OFF와 같아야 한다. clean의 전체 위치 번호를 그대로 복사하지 않는다.
6. **채점과 loss:** 기존 target 단위, mean log-prob와 context-preference reducer를 재사용하되 ON 점수를 clean CE 점수로 잘못 연결하지 않는다. clean CE는 기존 Stage2 경로 그대로 별도 계산한다.

원본 이미지 encoder 결과는 ON끼리 공유할 수 있으나, clean decoder에서 생성한 latent/hidden/KV는 공유하지 않는다. 이미지 encoding의 공유 여부는 frozen/trainable 설정을 바꾸지 않으며 gradient 경로를 임의로 끊지 않는다.

OFF의 빈 `S E`를 현재의 엄격한 Stage2 latent span parser에 넣지 않는다. OFF는 일반 decoder 입력이며, ON의 실제 latent span만 기존 rollout에 전달한다. 기존 parser의 검증 규칙은 완화하지 않는다.

검토한 기존 연결 지점:

| 파일 | 기존 기능 / 새 변경 범위 |
|---|---|
| `src/step_preference/config.py`, `src/utils_deepseed.py` | 새 선택값과 분기 옵션 검증 |
| `run_stage2_training.sh`, `src/multinode_runtime.py` | 두 노드에 동일한 옵션 전달 |
| `src/step_preference/`의 새 insertion 계획기 | OFF 배열, 분기·target 위치 매핑 |
| `src/stage2/trainer.py` | 새 scorer 요청과 ON/OFF 결과를 기존 reducer에 연결 |
| `transformers/.../modeling_qwen2_5_vl.py` | raw 입력에서 새 경로 호출; 기존 clean CE 및 기존 scorer 호출 보존 |
| `transformers/.../`의 새 insertion scorer | 실제 OFF forward, KV 분기, Stage2 rollout, 채점 |
| `stage2_shared_prefix.py`, `stage2_latent_rollout.py` | 기존 기능 재사용; legacy 의미 변경이나 recurrence 재설계 금지 |

decoder의 `stage2_capture_kv`와 `stage2_prefix_key_values` 지원은 소스에서 확인했다. **새 조합의 gradient·위치·수치 동등성은 아직 실행 검증하지 않았다.** 구현 후 작은 비교 probe로 확인한다.

## 4. 옵션과 loss

기존 `prefix_mode=clean_prefix` 기본값과 기존 `v_off_prefix` 의미는 유지한다. 새 방식은 명시적으로 선택한다.

| 새 경로 설정 | 값 / 의미 |
|---|---|
| `step_preference_prefix_mode` | `v_off_insertion` 추가 |
| `step_preference_branch_mode` | `independent` 기본; `cumulative` 선택 가능 |
| `step_preference_target_scope` | 이번 비교는 `immediate_text`; 기존 `until_next_latent` 의미도 보존 |
| `step_visual_only_context_preference` | `true` |
| `step_visual_off_suppress_latent_boundaries` | `false`를 명시, 양쪽 START/END 유지 |
| `step_preference_block_original_image` | 새 경로에서는 `false`; `true` 조합은 오류. OFF의 이미지 제거를 끄는 옵션이 아님 |
| 실행 경로 | 기존 Stage2B shared-prefix 경로, microbatch 1; legacy packing과 혼합하지 않음 |

`cumulative`를 기존 prefix mode에 지정하면 오류로 알린다. `independent` 기본값은 기존 경로의 동작을 바꾸지 않는다. CLI·launcher·원격 전달·실험 설정 기록에서 같은 값을 사용한다.

`immediate_text`는 해당 latent 바로 다음 **데이터 reasoning entry 하나**다. 한 토큰이나 마침표로 나눈 문장이 아니다. `until_next_latent`는 다음 latent 직전까지 기존 매핑이 선택한 텍스트 token들이다. 어떤 target scope든 ON/OFF의 target token ID 목록은 같아야 한다.

$$
s_i^b=\frac{1}{|Y_i|}\sum_{t=1}^{|Y_i|}\log p_\theta(y_{i,t}\mid C_i^b,y_{i,<t}),\quad b\in\{\mathrm{ON},\mathrm{OFF}\}
$$

$$
\mathcal L_{\mathrm{pref}}=\frac{1}{N}\sum_{i=1}^{N}-\log\sigma\left(\beta[s_i^{\mathrm{ON}}-s_i^{\mathrm{OFF}}-m]\right).
$$

새 reference model, OFF stop-gradient, 다른 loss 정규화는 추가하지 않는다. 기존 context preference의 수식을 유지하고 비교 문맥만 바꾼다. 다음 단기 비교를 승인받아 실행할 때에는 이전 진단과 같은 `CE + 10 × preference`, `beta=0.1`, `m=0`, entropy weight=0을 명시한다. 이는 **실험 설정**이지 전역 기본값 변경이 아니다. 이 설계 승인만으로 학습을 시작하지 않는다.

## 5. 구현 후 무엇으로 확인하는가

| 확인할 주장 | 필요한 실행 증거 |
|---|---|
| OFF가 정말 이미지 없는 실행이다 | dropout=0에서 원본 이미지 값·해상도 및 clean latent를 바꿔도 OFF token ID/position/KV/logit이 동일; OFF 호출 인자에 이미지 입력·clean 상태가 없음을 기록 |
| ON이 실제 OFF에서 분기한다 | decoder 호출 trace에서 OFF가 먼저 실행되고, ON prefix KV가 정확히 OFF의 삽입 직전 구간임을 확인 |
| 두 분기 모드가 다르다 | 2개 이상 삽입점에서 이전 삽입 이력이 independent에는 없고 cumulative에는 있음을 입력·KV trace로 확인; 첫 삽입점은 두 모드가 동일 |
| Stage2 recurrence를 보존한다 | 작은 dropout=0 모델에서 KV 방식과 전체-prefix 재계산의 latent/logit 및 parameter gradient 비교; ON/OFF 양쪽에서 finite gradient 확인 |
| target 미래 정보를 보지 않는다 | predictor/target 위치 매핑 검사 및 미래 token 변경에 대한 이전 점수 불변성 검사 |
| 기존 옵션을 깨지 않는다 | 기존 관련 회귀 테스트와 새 옵션의 두 노드 전달 테스트 |

집중 테스트 한 세트와 가장 작은 핵심 통합 검사를 사용한다. KV 접합의 실행 증거 없이 “Stage2와 동일”이라고 주장하지 않는다. 성능·학습 속도·preference 정체 해소는 이후 실제 다중 노드 실험으로 따로 확인해야 한다. 단위 테스트 통과나 loss 감소만으로 정답률 개선을 주장하지 않는다.

## 6. 경계와 미검증 사항

- ON은 raw 이미지와 latent 모두에 접근한다. 새 결과를 “latent만의 효과” 또는 “정체 원인 규명”으로 해석하지 않는다.
- cumulative는 같은 원본 이미지 블록을 각 삽입점에서 다시 넣으므로 시퀀스가 길어진다. 임의 truncation, 이미지 축소, latent 개수 감소로 처리하지 않는다. 입력 길이와 실제 peak memory를 확인한 뒤 학습 예산을 정한다.
- KV를 잘라 재사용하는 새 경로의 autograd, gradient checkpointing, 위치 정합성은 아직 미검증이다. 기존 인터페이스가 존재한다는 사실만으로 실행 가능성과 수치 동등성을 보증하지 않는다.
- 기존 코드와 기존 실험은 삭제·재명명하지 않는다. 새 결과에는 정확한 prefix/branch 옵션을 표시한다.
- 구현 과정에서 입력 배치·이미지 출처·teacher forcing·loss 의미를 바꿀 필요가 생기면 이 설계로 승인받은 범위를 넘어가므로 다시 확인한다.

## 7. 제안 v1 검토 기록 — 구현 승인 요청 전 중단

**현재 결론: 이 제안은 구현 가능한 최종 합의가 아니다.** 2절의 OFF latent 슬롯 제거는 사용자 확인을 받은 사항이 아니라 초안의 가정이었다. 이미지에 조건화된 latent를 금지하는 것과, 이미지 없이 새로 생성하는 latent까지 없애는 것은 다른 요구다. 사용자에게 이 차이를 확인하기 전 구현하거나 학습하지 않는다.

- `hate.root`: OFF가 clean/image 유래 상태 또는 위치 번호에 의존하면 실험 전체가 다시 잘못된 개입을 비교하게 된다.
- `hate.first_nail`: 작은 두 삽입점 샘플에서 이미지 값·해상도와 clean latent를 바꾸고, 실제 OFF 호출 인자·token ID·위치·KV·logit을 비교한다. 시각 상태의 유입이나 값 변화가 있으면 가정을 반증한다. **미실행**: 새 경로가 아직 없고 구현 승인을 받지 않았다. 제안된 시험은 실행 증거가 아니다.
- `macrothink`, fresh A — **divergent-incompatible**: 이미지 없는 baseline도 자기 문맥에서 연속 latent를 생성해야 한다는 해석. 초안의 모든 latent 슬롯 제거와 충돌한다.
- `macrothink`, fresh B — **divergent-compatible**: 이미지 없이 생성된 latent는 적법한 baseline 상태이며, 금지해야 할 것은 이미지에 조건화된 상태라는 지적. 삽입점·반복 이미지·loss pairing도 명시해야 한다는 지적은 2–4절과 대조했다.
- 공통 쟁점: “이미지 없는 OFF”가 “latent 계산까지 없는 OFF”를 뜻하는지 확정되지 않았다. 독립 검토의 유사한 지적은 사용자 의도의 증명이 아니며 다수결로 채택하지 않는다.
- `prism`과 최종 handoff cold-read는 **보류**: 이 핵심 요구를 먼저 확인하고 수정 제안을 새 버전으로 동결한 뒤 필요한 검토를 진행한다. 최종 design commitment와 구현 승인은 아직 없다.

이번 작업에서는 이 검토 문서만 추가했다. 모델·loss·launcher 코드는 수정하지 않았고 학습도 시작하지 않았다.

## 8. Design commitment v1 — 최종 구현 승인용

### 합의한 문제와 범위

기존 `v_off_prefix`의 clean-generated latent 재사용은 사용자가 요구한 이미지 없는 OFF가 아니다. 이를 기존 옵션에서 몰래 바꾸지 않고, 새 `v_off_insertion` 경로를 추가한다. 그 안에서 `independent`와 `cumulative`를 선택한다. 기존 `clean_prefix`, `v_off_prefix`, 기본값, 실험 기록은 보존한다.

### 최종 설계

- **OFF:** 원본 이미지 블록과 visual latent 슬롯을 제거한 토큰 배열을 실제 실행한다. `latent_start/end`와 기존 텍스트는 유지한다. 텍스트-only latent를 생성하거나 대체 벡터를 넣지 않는다. clean latent·hidden·KV를 재사용하지 않는다.
- **ON:** OFF의 삽입 직전 KV에서 분기하여 현재 `latent_start` 직전에 원본 `image_input` 블록을 넣는다. 기존 Stage2 rollout으로 그 분기의 visual latent를 새로 생성하고 같은 target 텍스트를 채점한다. helper feature 직접 주입이나 새 latent 생성법은 추가하지 않는다. 삽입 이미지는 텍스트 채점 때도 접근 가능하며 별도 차단하지 않는다.
- **independent:** 각 시점에 원래 OFF에서 새로 분기한다. 이전 ON의 이미지·latent·KV는 가져오지 않는다.
- **cumulative:** 첫 삽입 전 OFF에서 출발한 ON 실행을 이어간다. 이후 삽입점마다 같은 원본 이미지 블록을 다시 넣고, 앞선 이미지·latent·텍스트 KV를 유지한다. 비교하는 OFF는 계속 이미지 없는 실행이다.
- **학습 방식:** 기존 teacher forcing과 target token ID 목록, Stage2 latent recurrence, context-preference loss 수식, clean CE를 유지한다. 이번 비교의 target은 `immediate_text`이며 기존 다른 target 옵션은 제거하지 않는다. 새 reference model이나 OFF stop-gradient는 넣지 않는다.
- **변경 범위:** 새 입력/분기 scorer와 설정 전달만 추가하고 기존 KV 분기·latent rollout·loss reducer를 재사용한다. 새 prefix에서 `branch_mode=independent`가 기본이며 `cumulative`는 선택이다. START/END 유지, V-only, 기존 shared-prefix 실행 경로를 명시하고 충돌 설정은 오류 처리한다. 일반적인 retry·recovery·새 실험 관리 시스템은 만들지 않는다.

입력·위치 매핑, 기존 연결 지점, 수식과 옵션은 2–5절에 구체화되어 있다. 원본 이미지의 중간 삽입은 이 새 preference 경로에만 적용한다. 기존 clean CE의 Stage2 입력 배치는 바꾸지 않는다.

### 비평에서 달라진 점과 해소한 이견

제안 v1의 계산 흐름은 유지한다. 사용자 정정으로 **텍스트-only latent 생성이라는 대안은 폐기**했다. fresh A의 “OFF도 latent를 생성해야 한다”는 해석은 사용자 요구와 맞지 않아 채택하지 않는다. fresh B가 제시한 가능성 역시 새로운 옵션으로 만들지 않는다. 이견은 다수결이 아니라 사용자의 ILVR visual latent 정의와 후속 지시로 해소했다. 독립 검토의 일치 여부는 구현 증거가 아니다.

`prism`의 두 관점은 다음과 같다.

| 관점 | 판정과 가장 중요한 이유 |
|---|---|
| 요구 충족 | 설계 수준 pass: OFF에서 모든 visual latent를 빼고 텍스트-only 대체를 금지하므로 사용자 정정을 반영함 |
| 실행 가능성 | unclear: 새 KV/rollout 조합의 gradient와 cumulative 메모리를 아직 실행하지 않음 |

두 관점의 판정 차이를 해소할 다음 질문은 **“승인 후 작은 실제 모델 비교 probe에서 OFF 분리, KV/전체 재계산의 gradient 정합성, 대표 입력의 메모리 적합성이 확인되는가?”**다. 답이 나오기 전 구현의 정확성이나 학습 가능성을 확정하지 않는다. 요구 충족 판단은 실행 검증을 대신하지 않는다.

### 첫 검증과 남은 불확실성

`hate.first_nail`은 7절의 실제 OFF 호출 추적과 이미지/clean latent 교란 검사다. **아직 실행하지 않았다.** 새 구현이 없기 때문이며, 그 결과를 가정하지 않는다. 구현 승인 후 집중 테스트의 첫 기준으로 삼고 실패하면 학습하지 않는다. 이어서 두 분기 이력, target/predictor 정합성, 기존 Stage2 recurrence를 작은 모델의 전체 재계산과 비교한다. 관련 회귀 테스트 한 세트와 최소 통합 검사만 수행한다.

실제 GPU 메모리, 실행 시간, preference loss 정체 해소 및 정확도 개선은 모두 미검증이다. 구현 승인은 이 효과를 인정하는 승인이 아니며, **장시간 학습·평가 시작도 포함하지 않는다.** 새로운 비용 큰 실험이나 설계 의미 변경에는 별도 확인이 필요하다. 위험을 면제받았다고 간주한 항목은 없다.

최종 문서 검토 후 구현 승인을 받으면 구현 계획과 테스트로 진행한다. 이미 정정된 latent 정의를 다시 질문하거나 텍스트-only 옵션을 추가하지 않는다.

### 문서 검토 증거

`shower`의 독립 cold-read는 최종 commitment와 참조한 설계 내용을 읽고 **stands on own**으로 보고했다. 처음 전달한 commitment 발췌만으로는 2–5절이 빠져 이해가 막혔으며, 해당 내용을 같은 검토에 추가 제공한 뒤 그 차단점이 해소되었다. 수치 허용 오차와 작은 모델의 구체적 크기는 후속 구현 계획에서 정한다. 이 결과는 문서 이해도 검토일 뿐, 코드 정확성이나 학습 성능의 검증이 아니다.
