# Stage 1 vision_gain v2 구현 설계

상태: 대화상 설계 승인 완료, **이 문서의 사용자 검토 대기**. 구현·학습·분산 실행 검증은 수행하지 않았다.

## 1. 목적과 승인 범위

기존 Stage 1의 teacher-forced CE/alignment 학습에, teacher-selected visual latent가 바로 다음 GT text item 예측에 기여하도록 보조 parameter displacement를 추가한다. 보조 방향에서 masked score를 낮추는 일차 성분을 projection으로 제거한다.

알고리즘 기준은 canonical `documents/vision_gain_v2.md`다. 읽은 파일의 SHA256은 `bd3a1ec535c81bdeebde033e4f5d97920e768790b1010c66c1cd8bdcffb79065`다. 이 문서는 해당 수식을 실제 Stage 1의 데이터·Trainer·DeepSpeed lifecycle에 연결한다.

사용자가 승인한 선택은 다음과 같다.

- 독립된 worktree `/root/workspace/ILVR/.worktrees/stage_1_vision_gain`, branch `stage_1_vision_gain`에서 진행한다.
- 초기 활성화 경로는 **DoRA + ZeRO-1/2**다. 기본 Stage 1 실행은 ZeRO-1을 유지한다.
- 각 블록의 masked 경로는 전체 forward를 독립적으로 재계산한다.
- `alpha_vis` 기본값은 0이고, 활성화 실행에서는 양수 값을 명시한다. 검증된 효능 계수는 없다.
- 실제 OOM이 확인되면 그 원인에 맞춰 구현을 조정하되 목적함수·scoring 범위·전역 배치를 유지한다.
- 인접 text/text 경계를 가로지르는 whole token은 보조 scoring에서만 제외한다.
- 기존 DeepSpeed 0.18.7 호환 및 Stage 1 dropout context 연결 수정을 필요한 범위에서 재사용한다.

teacher feature 선택, teacher target의 stop-gradient, 원본 이미지 입력, 기존 alignment 정의·가중치, optimizer 종류와 기본 업데이트, EMA 규칙, 모델 구조, Stage 2 목적함수는 유지한다. Self-generated latent 학습, reference teacher 추가, calibration, margin/softplus, 새 warm-up, 전체 optimizer 이동량 projection은 포함하지 않는다.

## 2. 기준 코드와 확인한 실행 제약

기준 커밋은 `2f48c51985c395c16bea80a89847d7586728f9cd`다. 새 loss 기본 정책을 포함하는 `dil-loss-defaults-20261001`에서 분기했다. 이 계보 자체가 Stage 1의 실제 실행 가능성을 보장하지는 않는다.

현재 코드에서 확인한 사항:

1. `src/stage1/trainer.py::compute_loss`는 teacher target을 만들고 `ce_weight * CE + sim_weight * alignment`를 계산한다.
2. Stage 1 metadata에는 latent span은 있으나 바로 다음 text item의 token 좌표는 없다.
3. 설치된 Accelerate의 `DeepSpeedEngineWrapper.backward`는 누적 경계에서 `engine.step()`도 호출한다. 일반 Trainer의 optimizer callback만으로 실제 이동 시점을 추정하면 안 된다.
4. ZeRO-1/2는 모델 weight 외에 FP32 master partition을 보유한다. 보조 이동량은 master와 모델 양쪽에 반영되어야 다음 step과 저장 후에도 유지된다.
5. 4D attention mask 사용 시 Qwen의 자동 multimodal position 계산 경로를 사용할 수 없다. 원래 2D mask로 계산한 position ID를 명시적으로 전달해야 한다.
6. Stage 1의 sample-stable dropout context는 한 original 호출과 최대 한 checkpoint replay를 추적한다. 여러 forward를 하나의 context에 단순 삽입하지 않는다.

현재 로컬 환경 metadata는 PyTorch `2.7.1+cu128`, DeepSpeed `0.18.7`, Accelerate `1.10.1`, TRL `0.15.2`다. 기준 trainer는 DeepSpeed `0.18.0`을 요구하고 학습 step의 dropout context 연결도 누락되어 있다. 기존 수정 `2b4d06cf9b25711a58b8685ded386a5575a9b856`와 `5644748ca0ee9eadb5ed01840c4ead2defd122b9`의 관련 변경을 검토해 재사용한다. 단순 버전 문자열 변경만으로 호환성을 입증했다고 처리하지 않는다.

보조 기능 OFF의 비교 기준은 위 필수 실행 수정을 적용한 Stage 1이다. 이 수정과 vision_gain 효과를 구분해 검증한다.

## 3. 설정과 지원 경계

학습 CLI는 `--stage1_vision_gain_alpha`, Stage 1 coordinator는 `--vision-gain-alpha`, shell 환경은 `STAGE1_VISION_GAIN_ALPHA`를 사용한다. 모두 하나의 resolved `alpha_vis`로 연결한다.

- fresh run의 생략값은 0이다. 명시값은 유한한 실수이며 0 이상이어야 한다.
- `alpha_vis > 0`일 때만 DoRA, ZeRO-1/2, BF16, decoder SDPA를 활성화 경로의 지원 조합으로 검사한다. vision attention과 기존 Stage 1 non-reentrant checkpointing 설정은 유지한다.
- 비활성 경로에 새 tuning/ZeRO 제한을 적용하지 않는다. 활성 상태로 지원하지 않는 조합을 요청하면 training 전에 명확히 실패한다.
- coefficient는 scheduler LR과 독립적인 고정 parameter-displacement 계수다. 암묵적인 LR 곱, accumulation 나눗셈, gradient clipping을 추가하지 않는다.
- fresh Stage 1은 CE weight 0.1과 기존 alignment weight를 사용한다. 문서의 CE 기호를 CE weight 1로 재해석하지 않는다.
- tokenizer mapping, runtime mapping, coefficient, model topology 오류는 조용한 fallback 없이 중단한다.

원래 optimizer는 base gradient에 대해 기존 clipping·AdamW·weight decay·linear schedule을 수행한다. 보조 방향을 base `.grad`에 더하거나 AdamW에 재입력하지 않는다.

## 4. 데이터와 GT scoring 계약

`Stage1SampleMetadata`에 블록별 immediate text target 좌표를 추가한다. source item의 경계를 보존하고, 실제 processor가 만든 token sequence와 latent replacement 이후 좌표에 연결한다. latent normalization에 따른 위치 이동도 반영한다.

기존 render 문자열, token ID, labels, image 순서와 padding을 변경하지 않는다. auxiliary metadata 생성 전후 이 값들이 일치해야 한다. token ID를 별도 tokenization 결과로 교체하지 않는다.

블록 `b`의 scoring 집합 `W_b`는 **바로 다음 source text item**에 온전히 속하고 label이 유효한 target 위치다. padding, `-100`, latent/marker 등 non-text target은 제외한다. 바로 다음 item이 없거나 text가 아니거나 유효 token이 없으면 해당 블록은 scoring-invalid다. 빈 text를 건너뛰어 다음 item으로 확장하지 않으며 pre text를 요구하지 않는다.

인접 text/text 경계에 걸친 token은 whole token을 auxiliary 집합에서 제외한다. 입력과 CE supervision에는 그대로 남긴다. latent/control 경계 등 그 밖의 경계 불일치나 좌표 mapping 실패는 source ID와 함께 오류로 보고한다. 임의로 token을 이웃 item에 배정하지 않는다.

각 target `i`는 predictor `i-1`의 logit을 사용한다. 선택한 predictor row를 FP32로 바꾸고 **전체 vocabulary**의 log-softmax에서 GT ID를 gather한다. token chunk로 계산할 수 있지만 chunk별 평균을 평균하지 않는다. token 합계를 `|W_b|`로 나누어 블록 평균을 계산한다.

`s_b^F = mean_{i in W_b} log p_F(y_i)` 및 `s_b^M = mean_{i in W_b} log p_Mb(y_i)`에서 두 경로의 target ID, 좌표, 블록 집합은 같다. 전체 update의 유효 블록 수를 `B`라 할 때 `S^F = sum_b s_b^F / B`, `S^M = sum_b s_b^M / B`, `G = S^F - S^M`이다. 샘플 평균, rank 평균, microbatch 평균으로 대체하지 않는다.

## 5. Forward와 메모리 수명

각 microbatch에서 teacher feature를 한 번 만들고 detach한 동일 값을 모든 경로의 latent 입력 및 base alignment target으로 사용한다. 원본 입력과 2D padding mask로 multimodal position ID를 계산하여 공유한다.

블록의 start/end marker가 `l_b`, `r_b`일 때 masked 경로는 기존 causal/padding mask에 `q >= r_b`이고 `l_b < k < r_b`인 edge 차단을 추가한다. 모든 관련 decoder layer에서 적용하며 원본 image key와 start/end marker는 남긴다. end marker와 이후 상태를 다시 계산하고, clean downstream KV cache를 재사용하지 않는다.

기본 실행 순서는 메모리 수명을 제한하기 위해 다음과 같이 정한다.

1. 유효 블록이 있으면 auxiliary clean forward에서 전체 clean score 합계의 gradient를 계산한 뒤 graph를 해제한다.
2. 유효 블록별 masked forward와 score differentiation을 순차 실행한다. 각 블록 처리가 끝나면 graph/logit을 해제한다.
3. 동일 teacher target과 확률적 조건으로 기존 base Stage 1 forward/backward를 수행한다.

즉, 활성 경로에서는 base forward와 별도로 clean score forward를 한 번 더 수행한다. 모든 masked graph와 base graph를 동시에 보관하지 않는다. 모든 gradient는 optimizer update 전 파라미터에서 계산된다.

각 forward/differentiation 쌍은 **새 context 객체**를 사용하되 sample identity, optimizer step, seed, stochastic site 값은 동일하게 둔다. checkpoint replay까지 해당 context 수명을 유지한다. auxiliary 계산이 base의 global RNG 진행을 바꾸지 않도록 RNG 상태를 보존·복원하며, teacher는 기존 eval 상태를 유지한다. context 검증을 끄거나 dropout을 0으로 바꿔 우회하지 않는다.

auxiliary forward는 CE/alignment를 계산하지 않도록 labels를 넘기지 않는다. 원래 Qwen forward는 여전히 전체 sequence vocabulary logits를 만들기 때문에 이것만으로 메모리 적합성을 주장하지 않는다.

## 6. Gradient 집계와 projection

학습 가능 student parameter 전체를 이름·순서·shape가 고정된 목록으로 만든다. DoRA adapter뿐 아니라 기존 학습 가능한 token row 등도 포함한다. frozen vision/base parameter와 teacher는 제외한다. unused gradient는 같은 parameter 공간의 0으로 취급한다.

auxiliary score는 `torch.autograd.grad`로 미분해 반환값을 별도 FP32 buffer에 누적하는 방식을 사용한다. `create_graph=False`이며 score를 DeepSpeed loss scaling이나 base backward에 넣지 않는다. clean gradient 합계 `C`, masked gradient 합계 `H`, valid block 수 `B_local`을 accumulation 동안 모은다.

누적 경계에서 모든 rank가 같은 순서로 SUM reduction에 참여한다. local valid block이 0이어도 빠지지 않는다. 전역 `B`로 나누어 `h = H/B`, `g = (C-H)/B`를 만들고 한 번 projection한다. 마지막 partial accumulation에서도 실제 전역 유효 블록 수가 분모이며 configured accumulation 길이로 추가 나누지 않는다. base gradient의 기존 정규화는 별도로 유지한다.

내적과 제곱 norm의 scalar reduction은 FP64로 수행한다. 이미 동일한 전역 벡터를 가진 rank 사이에서 scalar를 다시 SUM해 중복 계수화하지 않는다.

`h2 = h·h`, `hg = h·g`에 대해:

- `h2 == 0`: `u = g`.
- `0 < h2 < 1e-30`: auxiliary update를 생략하고 `tiny_h_norm`을 기록한다. 이 threshold는 수치 보호 기준이며 효능 계수가 아니다. checkpoint identity에 포함한다.
- 그 외: `u = g - min(0, hg) / h2 * h`.

projection 계수와 곱셈·뺄셈은 parameter chunk별 FP64 중간값으로 계산해 FP32 buffer에 저장한다. denominator에 epsilon을 더하지 않는다. score, gradient, projection 결과에 NaN/Inf가 있으면 base step 전에 실패한다. 단순 tiny-denominator skip과 nonfinite 오류를 구분한다.

`B == 0`이면 score gradient/projection/auxiliary displacement를 생략한다. `alpha_vis == 0`이면 auxiliary forward, gradient buffer, 집계를 생성하지 않고 base 경로를 사용한다.

## 7. 실제 ZeRO update와 EMA

Accelerate의 마지막 base backward가 DeepSpeed step을 호출하기 전에 auxiliary 집계를 준비한다. 기존 `_DeepSpeedStepAdapter`의 실제 mutation 경계를 사용하며 callback 호출 순서만으로 성공을 추정하지 않는다.

실행 순서는 다음과 같다.

1. base gradient만으로 기존 optimizer step을 수행한다.
2. 기존 applied-step witness로 실제 성공 여부를 확인한다.
3. 성공했고 유효한 `u`가 있으면 FP32 master parameter에 `alpha_vis * u`를 더한다. DeepSpeed의 기존 full-parameter get/set mapping을 사용하고 optimizer moment 항목을 수정하지 않는다. parameter별 순차 처리로 전체 master 복사본 추가 생성을 피한다.
4. 기존 ZeRO-1/2의 low-precision parameter 동기화 경로로 master를 모델 replica에 반영한다.
5. 기존 규칙대로 EMA를 갱신한다. `ema_tau=1`의 frozen teacher 동작도 유지한다.
6. auxiliary buffer와 update-local count를 초기화한다.

overflow 등으로 base step이 skip되면 auxiliary 이동과 EMA도 skip하고 해당 update의 buffer를 버린다. application/synchronization 오류 후에는 학습을 계속하거나 checkpoint를 성공 산출물로 발행하지 않는다. rollback/retry framework는 추가하지 않는다.

FP32 master에서 비교하는 수식은 `theta_new = theta_old + Delta_S1 + alpha_vis*u`다. BF16 모델 weight는 이 master의 cast 결과이므로 BF16 자체에서 실수 수식의 bitwise 일치를 요구하지 않는다. projection은 보조 방향만의 일차 성질이며 전체 finite update의 score 개선을 보장하지 않는다.

## 8. 실행 전제의 증거와 남은 불확실성

로컬 PyTorch의 `autograd.grad`는 `.grad`에 누적하지 않는 반환형 differentiation으로 정의되어 있다. 설치된 DeepSpeed `utils/torch.py`는 이 PyTorch 버전에서 post-accumulate hook을 등록한다. 이 서로 다른 runtime source는 별도 score-gradient 경로의 근거다.

그러나 실제 ZeRO-1/2 engine, non-reentrant checkpoint, DoRA, rank별 다른 블록 수를 함께 실행한 증거는 없다. `autograd.grad`가 reducer state까지 완전히 보존하는지, master 갱신이 다음 step과 save/resume에 유지되는지는 **구현 승인 전에 해결할 실행 전제**다.

따라서 implementation plan을 확정하기 전에 다음 제한된 premise probe를 별도로 제시해 명시적 실행 승인을 받는다. 이 문서 승인은 probe 실행 승인을 대신하지 않는다.

- 설치된 실제 PyTorch/DeepSpeed/Accelerate 조합, 작은 decoder와 DoRA 및 동일 checkpoint context를 사용한다.
- 기존 coordinator를 통해 2노드·4rank에서 ZeRO-1과 ZeRO-2를 순차 검사한다. 최대 2 optimizer update, microbatch 1, accumulation 4를 사용한다.
- auxiliary differentiation 전후 base `.grad`/reducer 상태, 순수 base 대비 AdamW moment, FP32 이동량, 다음 step의 보조 이동 유지 여부를 확인한다.
- 정상 경로, rank별 불균등 block 수와 0-block rank를 포함한다. full-model 메모리 적합성을 이 작은 probe로 주장하지 않는다.
- 실패하면 원인을 기록하고 설계를 수정한다. runtime 증거 없이 source-level 가능성을 완전 호환으로 승격하지 않는다. probe를 생략하려면 남은 불확실성에 대한 사용자 위험 수용을 별도로 받는다.

probe 코드가 필요한 경우에도 사용자가 선택한 구현 방식과 해당 실험 코드의 TDD·독립 검토 요구를 적용한다. 실행 방식은 아직 선택되지 않았다.

## 9. Checkpoint와 resume

기존 checkpoint metadata 저장 경로에 `stage1_vision_gain` 설정 identity를 추가한다. identity는 schema/formula version `stage1-vision-gain-projection-v2`, resolved alpha, immediate-text window 규칙, text-join crossing-token 제외 규칙, latent-body mask 규칙, `min_h_norm_sq=1e-30`, reduction 및 precision 정책을 포함한다.

- 새 fresh run은 생략값 0을 저장한다.
- resume CLI의 생략은 저장값 상속이다. CLI에서 명시한 다른 값은 training 전에 거부한다. parser가 생략과 명시적 0을 구분해야 한다.
- legacy checkpoint에 identity가 없으면 보조 기능 OFF로만 resume한다. 기존 coefficient/formula를 바꾸는 resume는 허용하지 않는다.
- 기존 checkpoint weight를 initializer로 사용하는 **fresh run**은 optimizer·scheduler·global step을 resume하지 않는 기존 경로로 구분한다.
- checkpoint는 optimizer boundary에서만 저장한다. update-local gradient buffer를 모델 weight처럼 저장하거나 mid-accumulation resume를 새로 지원하지 않는다.
- 기존 EMA/optimizer/scheduler/RNG/ZeRO state와 저장 정책은 유지한다. 보조 master 이동 후 저장하고, 저장값이 실제 재로딩 모델에 반영되어야 한다.

## 10. OOM 대응과 로그

첫 구현은 독립 full recomputation과 순차 graph 해제를 사용한다. rank별 peak allocated/reserved memory와 clean-score, masked-score, base-backward, auxiliary-apply 중 어느 단계에서 실패했는지 기록한다.

OOM이 확인되면 실패 로그와 run identity를 보존하고 원인이 확인된 할당만 줄인다. 선택지는 activation checkpointing/recomputation, 필요한 predictor row만 LM head에 통과시키는 경로, 전체 vocabulary를 유지한 token-chunk scoring이다. 전체 logits가 이미 생성된 뒤 scoring만 chunk하는 수정은 upstream 할당을 줄였다고 주장하지 않는다.

각 조정은 작은 입력에서 원래 score와 gradient에 대한 비교 검증을 거친다. 실패 원인에 대한 회귀검사 하나와 필요한 통합검사를 수행한 뒤 새 run ID로 재실행한다. 승인된 전체 재계산 구조를 prefix-cache 방식으로 바꾸는 등 material design 변경은 재승인을 받는다.

OOM 대응으로 target/window 축소, block sampling, masked detach, teacher 변경, global batch 축소, single-node training, 임의 dependency downgrade를 하지 않는다. speculative retry/offload framework는 만들지 않는다.

update 단위로 `S^F`, `S^M`, `G`, 전역 유효 블록 수, `h·g`, `h·u`, `g·u`, `||u||²`, projection 적용 여부, zero-direction 여부, tiny-denominator skip과 base-step skip 이유, alpha 및 rank별 peak memory를 기록한다. `B=0`의 점수는 0으로 위장하지 않고 unavailable로 표현한다. 비율은 대상 update 수를 분모로 명시한다.

## 11. 검증과 완료 조건

| 범주 | 필요한 증거 |
| --- | --- |
| 데이터/좌표 | 인접 text/text, 빈 text, 연속 latent, pre 없음, padding, -100, marker, truncation·replacement 좌표 사례; 입력/CE 불변 |
| Mask/scoring | `i=r_b+1`인 첫 text target이 masked end-marker logit을 사용함; 모든 관련 layer 차단; 원본 이미지·marker 보존; 동일 position/dropout; masked student gradient 존재 |
| Projection | FP64 reference 대비 FP32 구현; h=0, g=0, 반대/평행/일반 벡터, tiny h, nonfinite; `h·u >= 0`, `g·u = ||u||²`, `(g+h)·u >= 0`의 scale-aware 오차 |
| 집계 | 불균등 블록/토큰 수, 0-block rank, microbatch 분할과 마지막 partial; 합친 reference의 score/gradient/projection과 비교 |
| 업데이트 | base-only와 동일한 초기 상태에서 FP32 master delta가 base delta + alpha*u; 같은 step의 AdamW moments 동일; 다음 step 및 save/resume에 이동량 유지 |
| OFF/skip | alpha=0에서 auxiliary 호출/할당 없음; global B=0 복귀; base step skip 시 auxiliary/EMA skip; 기존 Stage 1 결과 비교 |
| 실행 조합 | 실제 ZeRO-1/2 각각의 premise probe와 real-model 2노드·4rank coordinator smoke; configured world size 대신 모든 rank의 실행 evidence |
| 메모리 | retained 입력 중 긴 sequence와 많은 visual block을 가진 사례를 포함한 real-model peak 측정; OOM 발생 시 조정 후 동일 의미 검증 |
| 저장/설정 | alpha 생략·명시 0 구별, legacy OFF, mismatch 사전 거부, resolved CLI/env와 checkpoint identity 일치 |

수학적 CPU reference는 float64에서 `rtol=1e-10, atol=1e-12`를 사용한다. FP32 projection은 `rtol=1e-5`와 `1e-7 * max(1, ||h||||u||, ||g||||u||, ||u||²)`의 scalar 절대 오차를 사용하며 raw residual도 기록한다. 같은 logits를 사용한 chunked scoring 비교는 `rtol=1e-5, atol=1e-6`이다. BF16 모델 경로를 재계산해 비교할 때 score는 `rtol=1e-3, atol=1e-3`, parameter별 gradient는 `rtol=1e-2, atol=1e-5`를 사용하고 global relative L2 error도 1% 이하여야 한다. reference gradient norm이 0이면 상대 오차 대신 절대 오차를 적용한다. OFF 경로의 동일 설정 반복, auxiliary 전후 base gradient/reducer 불변, 같은 base step의 AdamW state 비교는 가능한 항목에 bitwise equality를 요구한다. FP32 master update는 동일 연산 순서의 reference와 `rtol=1e-6, atol=1e-7`로 비교하며, 보조 이동 자체가 관측되도록 FP32 rounding보다 큰 test signal을 사용한다. 오차 기준은 실패를 통과시키기 위해 사후 확대하지 않는다.

독립 reviewer는 specification compliance와 quality를 모두 확인한다. CPU 통과, premise probe, real-model smoke, full training, 평가 정확도를 따로 보고한다. 학습 효과의 성공은 gain 증가만으로 판정하지 않는다.

전체 학습이나 TEST400 평가는 이 설계 문서 작성 단계의 실행 범위가 아니다. 향후 학습은 canonical `AGENTS.md`, `documents/training-defaults.md`, `docs/multinode-training.md`에 따라 4rank, microbatch 1, accumulation 4, global batch 16, LR 5e-5, seed 42, warmup 10, CE 0.1 및 Stage 1 고유 설정을 명시한다. update budget은 retained 수와 실제 sampler/Trainer remainder로 계산하며 역사적 431을 복사하지 않는다.

## 12. 구성 경계와 변경 범위

- `src/stage1/data.py`, `types.py`: immediate-text metadata와 원래 token 좌표 대응.
- Stage 1 전용 vision-gain 모듈: scoring, mask, gradient buffer/reduction, projection, 작은 수치 helper.
- `src/stage1/trainer.py`: teacher-target 재사용, branch별 context, 실제 update 경계와 diagnostics 연결.
- `src/stage1/composition.py`, 기존 CLI/coordinator/worker: alpha 전달, 지원 조합 검사와 resolved 설정.
- 기존 Stage 1 checkpoint metadata 경로: objective identity 및 resume 검증.
- `tests/stage1` 및 기존 multinode contract tests: 요구사항별 검증.

기존 Stage 2 helper는 semantics가 동일한 부분만 참조·재사용한다. Stage 2의 pre-window 요구나 target/query 범위를 Stage 1에 그대로 적용하지 않는다. OOM 증거로 LM-head row 선택이 필요해지면 기존 Qwen forward에 backward-compatible 선택 경로를 추가하고 OFF/Stage 2 회귀를 검증한다. 관련 없는 refactor는 포함하지 않는다.

모든 model-bearing artifact는 worktree의 `checkpoints/`, 데이터·부분집합은 `data/`, 결과·workflow 기록은 `results/`, 로그는 `logs/`에 둔다. 기존 symlink를 유지한다.

## 13. 현재 증거와 다음 승인 단계

이미 실행한 baseline CPU 명령은 다음과 같으며 **120 passed**였다. 실제 GPU runtime 검증은 아니다.

```bash
PYTHONPATH=transformers/src:src CUDA_VISIBLE_DEVICES='' \
  ./venv/ilvr-venv/bin/python -m pytest \
  tests/stage1/test_data.py tests/stage1/test_targets.py \
  tests/stage1/test_trainer.py -m 'not gpu and not distributed' -q
```

기록은 `results/stage_1_vision_gain/workflow/intake.txt`, 로그는 `logs/stage_1_vision_gain/baseline-cpu-02.log`에 있다. 첫 실행의 import-path 누락 실패도 별도 로그로 보존했다. 이 문서 작성 중 제품 코드는 변경하지 않았다.

사용자가 이 문서를 검토·승인하면 architectural 경로의 다음 skill은 `superpowers:writing-plans`다. 실행 전제 probe의 별도 승인·증거 또는 명시적 위험 수용을 해결한 뒤 구현 계획을 확정하고 사용자가 실행 방식을 선택한다. 그 후 TDD, 선택한 구현 방식, 독립 검토, fresh verification, 해당 시 branch finishing 순서를 따른다.
