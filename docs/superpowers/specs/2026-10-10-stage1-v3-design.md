# Stage1-v3 설계 — design proposal v1

작성일: 2026-10-10. 상태: 대화형 설계 A/B/C 확인 후 작성한 제안서.
**서면 설계 최종 승인, 구현 계획 승인, 구현 방식 선택 및 학습 실행 승인은 아직 없다.**
이 파일은 자체 검토 후 동결하며, 비판 결과와 최종 승인 대상은 별도 commitment 문서에 기록한다.

## 1. 목적과 승인 범위

사용자는 `stage1-v3` worktree를 만들고 `documents/text_only_gain_v1.md`와
`documents/text_only_gain_v1/`를 참고하여 구현을 준비하도록 요청했다.
목적은 선행 assistant 텍스트의 직접 attention 접근을 제한한 상황에서 현재 visual latent가
바로 다음 GT 텍스트 예측에 기여하도록 학습하고, 제한된 예측이 clean student에서 지나치게
벗어나지 않도록 하는 것이다. 생성한 rejected 응답을 학습하는 방식은 아니다.

사용자가 확인한 사항:

- 학습 프로필: 저장소 기본값을 적용하고 첨부 자료는 알고리즘 참고로 사용한다.
- A: 기존 Stage1 entrypoint/coordinator와 CE·alignment·helper 경로에 한정된 v3 모듈을 연결한다.
- B: 동일 helper snapshot, clean/shared-prefix/packed replay, text-only query별 mask 계약.
- C: gradient·optimizer 순서, 저장/resume, 검증 계약과 이 설계서 작성.

위 확인은 이 문서를 작성할 권한이다. 존재하지 않았던 서면 설계나 구현 계획의 승인이 아니다.
구현 준비의 성공 기준은 재현 가능한 알고리즘 계약, 수정 범위, 검증 기준과 불확실성을
명시하는 것이다. 향후 구현 성공은 아래 기계적 검증으로, 성능은 별도 평가로 판단한다.
참고 자료의 54.25% 성능을 이번 프로필의 예상 결과로 약속하지 않는다.

## 2. 작업 기반과 근거

- Branch `stage1-v3`; native worktree `/root/.codex/worktrees/541e/ILVR`;
  접근 alias `/root/workspace/ILVR/.worktrees/stage1-v3`.
- 코드 기반 `60ff1df410635610bd41f28ca240e4817ca84cab` (`stage_1_vision_gain`).
  objective-aware CE 기본값 계보를 포함하나, 이전 부분 accumulation resume 실패가 미해결이다.
- Canonical `AGENTS.md`, `documents/training-defaults.md`와 이번 세션의 사용자 workflow가 적용된다.
- 참고 주 문서 SHA256:
  `40a77d9062be3067a334824c241a7109f6141568506421de6565c092354f31e4`.
  bundle manifest 52개 파일의 크기·해시를 확인했다. TRAIN 3,450 / TEST 400.
- Bundle은 완결된 실행 checkout이 아니다. 빠진 의존성이 있고 현재 dropout과도 차이가 있다.
  참고 모듈 전체를 덮어쓰거나 bundle을 runtime dependency로 설치하지 않는다.
- 기존 CPU baseline: CLI/targets/trainer/EMA 153 passed. 전체 suite, GPU 동등성 또는
  분산 실행 검증으로 확장 해석하지 않는다. 명령/원본 로그/저장 경로 위반과 복구 기록은
  `logs/stage1-v3-preparation/workflow.md`에 있다.

## 3. 통합 방식과 경계

선택한 방식은 opt-in v3 objective/profile을 기존 실행 경로에 연결하는 것이다.
reference Trainer/Qwen 전체 복사는 초기 작업량은 작지만 기존 수정과 의존성·dropout 계약을
훼손한다. 전체 별도 Trainer는 helper/PEFT/분산/checkpoint 경로를 중복하므로 선택하지 않았다.

| 경계 | 책임과 연결 지점 |
|---|---|
| CLI/composition/coordinator | 명시적 v3 선택, 지원 프로필 검증, 초기화·dry-run·resume 설정 해석 |
| data/types/targets | 기존 렌더링을 보존하면서 source item→token metadata, 즉시 다음 텍스트 span, helper snapshot 제공 |
| prefix selection 모듈 | detached attention·selector KL로 query별 text key 집합 계산; 모델/optimizer를 변경하지 않음 |
| Qwen trace/replay 모듈 | clean prefix KV, 실제 attention 통계, 원본 위치와 dropout을 보존한 packed R/RV 실행 |
| Gain/Text-KL execution 모듈 | alias graph의 score/VJP, 실제 optimizer parameter로 Text-KL gradient 전달 |
| update/trainer 연결 | global block gradient 합산·projection·실제 optimizer boundary의 추가 displacement |
| checkpointing | 목적함수/스케줄/teacher/RNG와 완료된 update 상태를 기존 저장 체계로 저장·검증 |

새 모듈은 `src/stage1/`와 기존 Qwen 모델 디렉터리 내부에 한정한다. 필요한 기존 호출부만
수정하며 생성 원본/실행 파일의 저장소 관례를 따른다. 이름과 파일별 작업 순서는 구현 계획에서
정한다. 범위에 없는 Stage2 변경, IIER, preference, Clean-KL, image masking, entropy budget,
일반화된 복구/아카이브/재시도 시스템은 추가하지 않는다.

v3 OFF의 기존 동작은 보존한다. 저장된 다른 목적함수를 v3로 재해석하지 않는다.
정확히 지원할 수 없는 과거 resume는 학습 전에 명확히 거부한다.

## 4. 새 실행의 기본 프로필

| 항목 | Stage1-v3 계약 |
|---|---|
| 초기화 | original Qwen2.5-VL-3B-Instruct pretrained base, fresh run; Stage1/Stage2 결과 가중치 사용 안 함 |
| 분산 | 2 nodes × 2 GPUs = 4 ranks; 기존 Stage1 multinode coordinator |
| 노드 | remote `korea@10.1.102.41` / `llm_kd` rank0, local rank1; rendezvous `10.1.102.41`; Gloo/NCCL `bond1.102` |
| batch | per-device microbatch1 × accumulation4 × ranks4 = global16 |
| optimizer | 기존 AdamW/linear, LR5e-5, warmup10 updates, weight decay0.01, max grad norm1.0 |
| 재현/precision | seed42, BF16, 저장소 `venv/ilvr-venv` |
| PEFT | DoRA r32/alpha64/dropout0.05; q/k/v/o/gate/up/down projection; latent8 |
| teacher / base loss | frozen EMA1.0; CE0.1을 raw CE에 한 번 적용; alignment1; 기존 helper 및 noise0.01 의미 보존 |
| runtime | ZeRO-1, offload 없음, decoder SDPA, 기존 Stage1 vision 로딩 경로, non-reentrant checkpoint와 sample-stable dropout |
| data | 기존 Stage1 max sequence32768, workers0; 필터 후 실제 retained IDs를 기록 |
| exposure | 한 epoch 상당; 실제 sampler padding·dataloader·Trainer 마지막 부분 accumulation 기준으로 budget 계산 |
| Gain | alpha0.001, packed execution, independent blocks, min h norm squared1e-30 |
| Text-KL | max beta0.01, warmup ratio0.05, temperature2 |
| prefix | text rho0.3..0.5, cap0.3, 낮은 attention mass fallback0.1; image cap0 |
| 비활성 목적 | Clean-KL0, preference OFF, IIER OFF |

Stage2의 vision FlashAttention2/Flex/ZeRO2 프로필 전체를 Stage1에 복사하지 않는다.
기존 Stage1 vision backend를 실행 시 실제 module/config에서 확인하고 양 노드 resolved 기록에
남긴다. 현재 코드의 선언만으로 backend 호환성을 단정하지 않는다.

환경 파일을 source한 뒤 모든 applicable 값을 effective CLI/env에서 명시해 stale fallback을
덮어쓴다. 이후 명시적 사용자 override와 saved resume만 예외다. 431/1296/212를 공통 step
상수로 쓰지 않는다. 모든 3,450개가 유지되더라도 실제 shard·padding·tail 처리를 확인하기
전에는 최종 update 수를 확정하지 않는다. teacher와 학습 프로필 변경 때문에 참고 실행의
CE1/EMA0.999/3epochs 결과를 재현했다고 표현하지 않는다.

## 5. 데이터와 forward 계약

원본 문자열, tokenizer 결과, CE label, embedding, image 순서와 position/mRoPE ID를 유지한다.
새 metadata는 기존 좌표를 설명한다. auxiliary mask를 만들기 위해 입력을 삭제하거나
재토큰화하지 않는다. 현재 latent 직후의 **한 text item**만 Gain target이다. 즉시 다음 item이
비텍스트이거나 유효 토큰이 없으면 그 block은 무효이며, 뒤의 다른 text item을 찾지 않는다.
padding, -100, 구조/latent 토큰 및 adjacent text/text 경계를 가로지르는 token을 auxiliary
span에서 제외한다. 원본 CE는 유지한다. 설명되지 않는 mapping 실패는 source ID와 함께 오류다.

microbatch마다 기존 frozen helper teacher로 detached latent target snapshot을 한 번 만들고
모든 경로에 재사용한다. 학습 중 새 autoregressive latent rollout을 추가하지 않는다.

| 경로 | parameter/graph | 역할 |
|---|---|---|
| F | 실제 trainable student | 기존 CE/alignment, detached clean Text-KL target |
| Gain-clean | 실제 값과 같은 detached trainable alias | clean prefix KV와 inline attention 통계, selector student logits |
| frozen reference | frozen teacher, eval/no-grad | 같은 helper snapshot으로 selector reference logits |
| R | Gain alias의 replay | 선택된 text key 제한, 현재 latent body 접근 유지 |
| RV | Gain alias의 replay | 같은 text 제한 + 현재 latent body key 접근 차단 |

유효 block이 있는 microbatch의 원래 core decoder 호출은 F1, Gain-clean1, frozen-reference1,
packed R/RV1이다. helper target 생성, vision encoder와 backward checkpoint 재실행은 이
숫자에서 제외한다. Text-KL은 추가 decoder/head forward와 세 번째 score VJP traversal을
추가하지 않는다. 호출 수는 runtime witness로 검증할 요구사항이며 메모리 상한의 증거는 아니다.

## 6. selector와 attention mask

block i의 후보 C_i는 현재 START보다 앞선 일반 assistant reasoning text token이다.
user/image/current post-text/special/crossing token은 후보가 아니다. 유효 target 위치 t의
예측 query는 t-1이다. END가 첫 post-text token의 query가 될 수 있다.

attention은 후보에 다시 정규화하지 않고 전체 causal key에 대해 softmax한 뒤 heads,
layers 순서로 평균한다. Gain-clean 및 frozen-reference logits에서 FP32 full-vocabulary
`p=softmax(student/2)`, `r=softmax(reference/2)`, `d=4*KL(p||r)`를 구한다.
`u=-log(max(d,0)+1e-6)`이며 같은 sample의 모든 유효 query에서 구한 평균으로 중심화한다.
`rho=0.3+0.2*sigmoid(u-mean(u))`. selector에 사용한 값은 모두 detach한다.

후보 attention 내림차순, 동률이면 원래 token 위치 오름차순으로 정렬한다.
candidate mass의 rho 비율에 도달할 때까지 선택하되 cap=`floor(0.3*N)`을 넘지 않는다.
임계 mass를 넘기는 token은 cap 안이면 포함한다. candidate mass<0.1이면
`min(cap,floor(0.1*N))`개를 선택한다. N=0 또는 cap=0의 빈 mask를 허용한다.

R은 **scored query 행에만** 선택된 text key를 차단한다. RV는 같은 text mask에 더해
**현재 block replay suffix의 모든 행**에서 현재 START<key<END의 latent body 전체를 차단한다.
START/END, 원본 image key는 접근 가능하다. block 간 mask를 누적하지 않는다.

clean prefix KV는 현재 END 직전까지 공유하며 detach하지 않는다. replay는 END에서 마지막
유효 prediction query까지 실행한다. 원본 mRoPE 위치와 원본 token별 dropout 대응을 유지한다.
이는 직접 attention edge를 제한하는 계약이다. 앞선 hidden state에 섞인 정보까지 제거했다는
인과적 주장이나 image 제거 주장은 하지 않는다.

## 7. gradient와 optimizer 계약

일반 optimizer 목적함수는 `L=0.1*CE + alignment + beta(s)*TextKL`이다.
TextKL은 `4*KL(stopgrad(p_F)||p_R)`이며 temperature2, FP32 full vocabulary를 사용한다.
microbatch 안에서 sample별 유효 query 평균을 구한 후 유효 sample 평균을 낸다. 유효 query가
없으면 zero다. 그 다음 기존 Trainer/DeepSpeed의 accumulation·rank reduction을 적용한다.
전체 update의 유효 sample 수로 다시 정규화하지 않는다. 이는 아래 Gain 분모와 다르다.

`beta(s)=0.01*min(1,s/ceil(0.05*S))`이고 S는 양의 실제 예정 update 수,
s는 완료된 Trainer global steps다. 첫 update의 beta는0이다. LR warmup10과 별개이며,
resume는 저장한 S와 warmup 의미를 유지한다.

R/RV score는 block별 유효 GT token log-probability의 평균(temperature1)이다.
accumulation 전체와 네 rank의 **unscaled** FP32 alias gradient 합과 valid block count를 SUM
한 뒤 global block count B로 나눈다. `g=grad(mean(S_R-S_RV))`, `h=grad(mean(S_RV))`.
모든 trainable parameter를 합친 `d=h·g`, `n=||h||²`에 대해:

- `0<n<1e-30`: u=0.
- `n>=1e-30`이고 d<0: `u=g-(d/n)*h`.
- 그 외: u=g. n=0도 여기에 포함한다.
- B=0: displacement 없음. rank별 유효 block이0이어도 필요한 collective에는 참여한다.
- nonfinite gradient/projection은 mutation 전에 오류로 처리한다.

positive R score와 Text-KL은 joint two-lane VJP (`vmap` randomness same), RV는 별도 VJP로
계산한다. Text-KL gradient를 detach한 뒤 custom autograd scalar를 통해 실제 optimizer-owned
parameter로 전달한다. shared-prefix derivative도 포함한다. ordinary beta/accumulation/
rank-reduction/clipping을 정확히 한 번 적용한다. Gain 전용 gradient는 Adam moments에 넣지 않는다.

**dropout bridge 요구사항:** native `nn.Dropout`용 reference 처리를 현재의
`SampleStableStage1Dropout(nn.Module)`에 그대로 사용하지 않는다. sample/optimizer-step/site의
seed 및 원본 token 좌표로 같은 mask를 재현하고, packed row와 joint VJP/checkpoint recompute에
명시적으로 대응시킨다. 기존 original/recompute 호출 검증을 조용히 해제하거나 dropout0으로
우회하지 않는다. 이 bridge의 실행 가능성과 gradient 동등성은 아직 관측되지 않았다.

실제 DeepSpeed update boundary에서:

1. global Gain 방향을 확정·검증한다.
2. 일반 loss의 AdamW step을 수행한다.
3. 성공한 step에만 `theta_FP32_master += 0.001*u`를 적용하고 BF16 model weight를 동기화한다.
4. EMA1.0 teacher는 그대로 유지하고 완료된 update 이후에만 checkpoint를 발행한다.

displacement에 LR을 곱하거나 accumulation으로 다시 나누지 않는다. skipped/failed base step에는
Gain을 적용하지 않는다. base step 후 Gain 적용 실패는 run을 중단하고 유효 checkpoint를
발행하지 않는다. 자동 rollback/recovery 체계를 새로 만들지 않는다.

## 8. tail, resume, 오류와 저장

v3는 실제 retained sample, sampler padding/drop-last, rank별 dataloader 길이와 최종 부분
accumulation을 근거로 한 epoch의 update 수를 산정한다. Stage2 zero-loss padding을 Stage1의
관측된 동작으로 주장하지 않는다. sample order/반복/분모와 tail의 실제 boundary를 기록하고
uninterrupted 실행 및 완료된 boundary에서의 resume가 같은 남은 sample과 update를 수행해야 한다.
기반 코드의 알려진 resume 실패는 v3 경로의 scoped 수정·회귀 검증 대상이다.

기존 checkpoint 체계에 v3 identity, source config, loss/mask/projection 공식 및 계수, frozen
teacher 상태, 예정 step budget/warmup, sampler/remainder identity, optimizer/scheduler/RNG/
global step을 보존한다. 완료된 optimizer boundary에서 동일 objective만 resume한다.
설정 변경은 fresh run이다. supported runtime/metadata가 아니면 학습 전에 거부하며,
오류를 serial execution이나 다른 loss/backend로 바꾸어 숨기지 않는다.

모든 model-bearing artifact와 이를 쓰는 test tmp는 `checkpoints/` 아래에, 입력은 `data/`,
metrics/output은 `results/`, 실행 로그는 `logs/` 아래에 둔다. 기존 symlink를 유지한다.
run ID별 하위 경로를 사용하고 reference 및 과거 checkpoint/result를 덮어쓰지 않는다.
의존성 재설치와 새 orchestration은 범위 밖이다.

## 9. 검증 계약과 한계

| 검증 | 반드시 확인할 주장 |
|---|---|
| CPU contracts | 즉시 다음 item/shift/crossing, 후보·tie·cap·fallback, KL 방향과 reduction, projection·zero/nonfinite |
| 독립 tiny reference 대 packed | 같은 weights/tokens/선택 mask에서 score, parameter gradient, shared-prefix derivative; dropout ON/OFF와 checkpoint recompute |
| gradient 전달 | beta·microbatch·rank reduction이 한 번 적용되고 실제 optimizer parameter에만 ordinary gradient가 들어감 |
| 실제 2 nodes/4 ranks ZeRO1 | 성공/skip/zero-valid rank/tail/save-resume의 호출·상태 전이; FP32 master/BF16 일치와 Adam moments 순서 |
| 비용 계약 | helper/vision/recompute를 구분한 core invocation witness; Text-KL 추가 forward/세 번째 score traversal 없음 |
| 기존 동작 | v3 OFF 및 지원하는 기존 resume 의미 보존; 미지원은 실행 전 reject |
| 성능 | 표준 TEST400 평가 결과를 기계적 검증과 분리 보고 |

tiny 비교는 독립 수식/직접 full computation을 기준으로 한다. reference packed 코드를 그 자체의
oracle로 사용하지 않는다. 고정 seed와 사전 명시한 dtype별 오차 기준을 사용하고, RNG 소비나
분모를 바꾸어 실패를 숨기지 않는다. CPU toy 수학 검증은 실제 3B/BF16/DoRA/DeepSpeed runtime
검증을 대체하지 않는다. 모든 training smoke는 두 노드 네 rank 및 기존 coordinator를 사용한다.

이후 학습 전 양 노드 코드·환경·입력·자원과 resolved profile을 dry-run하고 실제 네 rank
참여를 확인한다. 평가 시 현재 final PEFT artifact+original base, TEST400, greedy temperature0,
top_p1, max new4096, max sequence128000, repetition penalty1, no vision ablation,
`external_full_latent_block`를 사용한다. 새 scorer/evaluation migration은 이 설계 범위 밖이다.
성능 비교가 필요하면 같은 프로필의 control이 필요하며, 역사적 점수만으로 개선을 주장하지 않는다.

## 10. 설계 확정 전 불확실성과 다음 gate

현재 중요한 미검증 전제는 packed alias/shared-prefix/joint-VJP가 sample-stable dropout 및
checkpointing 아래에서도 독립 계산과 같은 gradient를 만들 수 있다는 점이다. 기반 branch의
partial accumulation resume도 실증된 실패가 있다. 정적 계약 또는 기존 153 CPU test는 이
문제들의 해소 증거가 아니다. 참고 성능의 새로운 프로필로의 이전 또한 확인되지 않았다.

자체 검토한 이 proposal v1을 동결한 후 root가 hate의 단일 root/first_nail, macrothink의
독립 read 결과를 수집한다. 같은 proposal을 읽고 중간에 수정하지 않는다. Superpowers가
그 결과를 별도 design commitment로 종합한다. 비용/상태 변경/구현 성격의 전제 실험은
**별도 명시적 승인** 후에만 실행한다. 실행하지 않으면 중요한 잔여 불확실성에 대한
**명시적 risk waiver**가 필요하다. 일반적인 `Ok`를 두 권한에 중복 사용하지 않는다.

최종 승인된 서면 commitment/spec만 `writing-plans`로 이어진다. 구현 계획 검토와 사용자
구현 방식 선택 후 TDD·독립 specification/quality review·fresh verification·해당 finish gate를
수행한다. 이 설계서 작성은 구현/학습 launch 또는 branch merge 승인이 아니다.
