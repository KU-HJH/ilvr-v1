# ILVR Stage 2: Entropy Transition + Counterfactual Vision Gain 구현

현재 열린 ILVR repository에 아래 objective를 구현하라. Superpowers plugin의 실제 설치된 skills를 읽고 활용하라. 연구 수식은 이미 확정된 요구사항이다. 수식을 재설계하거나 다른 loss로 대체하지 말고, 저장소에 맞는 구현 설계를 먼저 제시한 뒤 승인받아 구현·테스트·검증까지 진행하라.

## 0. 목표와 범위

목표는 주어진 interleaved trajectory `T_pre -> V -> T_post`에서 다음 두 동작을 함께 학습하는 것이다.

- 고정 reference 대비 pre-V text entropy 증가와 post-V text entropy 감소의 대비를 강화한다.
- 정상 latent Vision이 ablated latent Vision보다 동일한 post-text 정답을 더 잘 예측하도록 보강한다.

Stage 1은 변경하지 않는다. 새 학습 경로는 Stage 2에만 추가한다. Boundary 탐색, 추가 modality router, adaptive stopping, 방향 2의 preference loss, RL/DPO, entropy ramp/band, 새로운 KL 항은 이번 범위가 아니다. 기존 visual encoder의 freeze 설정과 trainable parameter 정책은 별도 근거 없이 변경하지 않는다.

새 기능은 기본적으로 비활성화한다. 비활성 상태에서는 기존 학습·추론 동작을 보존한다. 새 Stage 2 경로를 활성화하고 auxiliary weight를 모두 0으로 설정한 경우에는, 아래에서 정의하는 self-generated-latent CE-only baseline이 되어야 한다.

## 1. Superpowers 작업 절차

1. repository 및 상위 경로의 `AGENTS.md`, README, 현재 git 상태를 확인한다. 기존 사용자 변경을 덮어쓰지 않는다.
2. `using-superpowers`와 `brainstorming`에 따라 코드·제약을 조사하고 구현 설계를 정리한다. 아래 수식과 연구 목적은 확정 사항이며, 구현상 모순이나 미정 사항만 질문한다.
3. 변경 대상, forward/gradient 경로, ablation 방식, 테스트 범위, backend·메모리 제약을 제시하고 구현 설계 승인을 받는다. 승인 전 production code를 수정하지 않는다.
4. 승인 후 `writing-plans`로 실제 파일·함수·테스트 명령을 포함한 실행 계획을 작성한다. 실행 단계에서는 `using-git-worktrees`에 따라 안전한 격리 작업공간을 사용한다.
5. `test-driven-development`로 실패 테스트 확인 -> 최소 구현 -> 통과 -> 리팩터링 순서를 지킨다. 실행은 설치된 환경에 맞게 `subagent-driven-development` 또는 `executing-plans`를 사용한다.
6. 실패나 예상 밖 동작에는 `systematic-debugging`을 적용한다. 마지막에는 `requesting-code-review`와 `verification-before-completion`으로 수식·gradient·회귀 동작을 검토한다.

Skill 이름을 언급하는 것으로 사용을 대신하지 말고 실제 skill 내용을 읽어라. 없는 도구나 실행하지 않은 subagent를 사용했다고 주장하지 마라. 이미 확정된 연구 목적을 반복 질문하지 말고, 구현 계획에 승인받은 뒤에는 계획 작성만으로 작업을 끝내지 마라. 원격 push, 장시간 전체 학습, 대규모 다운로드는 별도 승인 없이 실행하지 마라.

## 2. 구현 전 저장소 점검

현재 checkout을 기준으로 실제 경로를 확인하라. 아래는 탐색 후보이지, 존재나 동작을 가정해도 되는 확정 인터페이스가 아니다.

- `src/main.py`: stage 선택, trainer 생성, collator, tokenizer, model 초기화.
- `src/trainer.py`: `compute_loss`, CE/alignment 분기, early return, EMA teacher.
- `src/utils_deepseed.py`와 argument parser: label masking, latent marker 처리, CLI 설정.
- repository의 Transformers fork: Qwen2.5-VL forward, Stage 2 latent 재입력, logits/labels shift, attention mask, RoPE, gradient checkpointing.

특히 다음을 확인하고 코드 위치를 보고하라.

- Stage 2에서 teacher/helper-image latent 주입 또는 alignment loss가 남아 있는지.
- `sim_weight == 0`이나 teacher-latent 부재에 따른 early return이 새 loss를 우회하는지.
- 실제 import되는 `transformers` 및 model class의 파일 경로와 버전. 설치된 upstream package와 repository fork를 혼동하지 말 것.
- model 파일이 generated file이라면 실제 수정 원본과 regeneration 정책. 기존 ILVR custom 코드를 덮어쓰는 regeneration을 하지 말 것.
- Stage 2 latent 생성 과정에서 text CE 및 post entropy가 latent 생성 계산 그래프까지 연결되는지.
- 지원되는 attention backend, mixed precision, LoRA, gradient checkpointing, DeepSpeed/ZeRO 설정.
- batch dimension 처리에 `input_ids[0]` 같은 제한이 있는지. 이번 변경에서 일반화하지 못하는 기존 제한은 명시적으로 검증하고 오류 처리할 것.

새 Stage 2 경로에서는 teacher/helper-image latent를 입력하지 않고 모델이 자체 latent를 생성해야 한다. teacher alignment 및 불필요한 EMA teacher 생성·갱신을 이 경로에서 실행하지 않는다. 이 분리와 auxiliary loss 추가를 구분하여, 같은 CE-only 경로를 공정한 비교 baseline으로 제공하라. 기존 legacy 경로를 조용히 수정하지 마라.

## 3. 구현할 수식: 변경 금지

각 유효 block k를 다음과 같이 정의한다.

    C_k -> P_k -> V_k -> Y_k

C_k는 공유 prefix, P_k는 pre-V text, V_k는 현재 latent block, Y_k는 선택된 post-text target이다. F는 정상 branch, M_k는 현재 block k만 ablate한 branch다.

### 3.1 Entropy transition

H_theta_pre[k], H_theta_post_F[k]는 정상 student branch의 entropy다.
H_ref_pre[k], H_ref_post_F[k]는 고정 reference의 정상 Stage 2 branch entropy다.
각 window 내부 entropy를 먼저 평균한 뒤 비율을 계산한다.

    r_pre[k]  = (H_theta_pre[k]    + eps) / (H_ref_pre[k]    + eps)
    r_post[k] = (H_theta_post_F[k] + eps) / (H_ref_post_F[k] + eps)
    D[k]      = r_pre[k] - r_post[k]

    loss_dyn[k] = -sigmoid(D[k] - gamma_H)

여기에는 `-logsigmoid`, softplus, hinge, 추가 temperature를 대신 사용하지 않는다. gamma_H는 sigmoid 기울기의 중심이지 entropy의 상한이나 정지 조건이 아니다.

### 3.2 Counterfactual Vision gain

같은 teacher-forced target Y_k의 유효 text token에 대해 token-mean log-likelihood를 계산한다.

    S_F[k] = mean_j log p_theta(y_kj | C_k, P_k, V_k, y_k,<j)
    S_M[k] = mean_j log p_theta(y_kj | C_k, P_k, Ablate(V_k), y_k,<j)

    G[k] = S_F[k] - stop_gradient(S_M[k])

    loss_vis[k] = softplus(m_V - G[k])
                = -logsigmoid(G[k] - m_V)

`S_M`는 현재 student의 ablated forward 값이며 고정 reference의 score가 아니다. M branch는 backward graph를 만들지 않고 계산하고, loss 경계에서도 detach를 명시하라. F branch와 그 latent 생성 경로에는 gradient가 연결되어야 한다.

S_F/S_M는 기존 CE와 동일한 전체 output vocabulary의 정규화 확률로 계산한다. 아래 entropy용 text-only vocabulary로 재정규화한 확률과 혼동하지 마라.

### 3.3 최종 loss

    loss_total = loss_ce_F
               + lambda_dyn * mean_k(loss_dyn[k])
               + lambda_vis * mean_k(loss_vis[k])

두 mean은 최종 mask 처리를 통과한 동일한 유효 block 집합에서 계산한다. batch 전체 entropy나 score를 먼저 합쳐 sigmoid/softplus 한 번을 적용하면 안 된다.

`loss_ce_F`는 새 Stage 2 정상 forward의 기존 supervised CE이며, 원래 supervised text와 boundary marker의 CE를 유지한다. latent pad 및 기존 ignore-index 규칙을 보존한다. m_V의 기본값은 0으로 한다.

유효 block이 없으면 auxiliary loss는 0이고 CE는 그대로 학습한다. 학습 label 자체가 모두 ignore인 예외도 finite하게 처리하되, 기존 CE의 정상 reduction을 변경하지 않는다.

## 4. Entropy vocabulary와 token alignment

Entropy는 modality-control token, latent pad, image/video placeholder, padding 등 비텍스트 구조 token을 제외한 vocabulary V_T에서 계산한다. tokenizer/config에서 실제 ID를 확인하고 제외 목록을 기록하라. token ID를 하드코딩하거나 알려지지 않은 token을 임의 ID로 대체하지 마라. vocabulary 크기와 tokenizer의 일치도 검증하라.

    log_p_T = log_softmax(selected_logits[..., V_T].float(), dim=-1)
    H_T     = -sum(exp(log_p_T) * log_p_T, dim=-1)

전체 vocabulary entropy에서 special-token 항만 빼는 것은 재정규화가 아니므로 허용하지 않는다. 일반 text token은 임의로 제외하지 마라. CE용 원본 logits를 in-place로 수정하지 않는다. entropy에는 자연로그를 사용한다.

일반적인 causal-LM 정렬에서 logits[t]는 input_ids[t+1]을 예측한다. 실제 fork의 출력 정렬을 테스트로 확인한 뒤 다음 boundary-state 정의를 적용한다.

    s = <|latent_start|>의 위치
    e = <|latent_end|>의 위치

    pre_window=1:  H_pre  = H_T(logits[s-1])
    post_window=1: H_post = H_T(logits[e])

post의 첫 text token이 e+1에 있는 정상 block 기준이다. 예를 들어:

    index:  0   1       2          3   4       5        6   7
    token: p0  p1  latent_start   z0  z1  latent_end   y0  y1

pre entropy는 logits[1], 첫 post entropy 및 y0 likelihood는 logits[5]로 계산한다. logits[s-2]는 마지막 pre token 자체를 예측하는 분포이며 이번 기본 정의와 다르다.

logits[s-1]의 실제 정답은 latent_start다. 이 위치의 marker CE는 유지하면서 entropy만 V_T로 재정규화한다. 따라서 pre mask를 '다음 label이 일반 text인가'로만 만들면 boundary를 잘못 제외하게 된다.

window 확장은 다음처럼 정의한다.

- pre: 같은 assistant의 P_k 안에 있는 마지막 pre_window개 text-state logit 위치. s-1을 포함한다.
- post: 같은 assistant의 다음 text segment에서 첫 post_window개 유효 text target을 선택하고, 각 target 위치 i에 대응하는 logits[i-1]을 사용한다.
- Vision gain의 Y_k는 이 post target 집합과 동일하다. 건너뛴 token이 있더라도 실제 teacher-forced prefix를 삭제하거나 이어 붙이지 않는다.
- 다른 latent block, user/system turn, padding, truncation 경계를 넘어가서 window를 채우지 않는다.

pre_window/post_window 기본값은 각각 1로 두고 설정 가능하게 만든다. 이는 구현 기준을 명확히 하기 위한 기본값이지, 성능상 최적값이라는 뜻이 아니다.

인접 block 사이에서 같은 logit 위치가 pre와 post window에 동시에 들어가면 양쪽 window에서 해당 위치를 제외한다. 그 결과 pre 또는 post가 비는 block은 auxiliary 집합에서 제외하고 이유와 개수를 기록한다. reference에도 동일한 최종 위치 mask를 적용한다. 깨진 marker pair, 중첩 marker, 0-length latent, text가 없는 구간은 조용히 정상 block으로 해석하지 않는다.

## 5. 고정 reference

reference는 Stage 1 완료 checkpoint에서 로드하고, Stage 2 방식으로 자체 latent를 생성하여 평가한다. 학습 중인 student를 따라가는 EMA teacher나 resume 시점 student를 reference로 대체하지 않는다.

필수 조건:

- reference는 `eval()` 및 `no_grad()`로 실행하고 모든 parameter를 freeze한다.
- student와 storage를 공유하지 않고 optimizer에 포함하지 않는다. LoRA가 있다면 reference adapter도 고정한다.
- 동일한 처리 완료 입력 이미지, input_ids, text trajectory, latent 길이, vocabulary, window 정의를 사용한다.
- reference는 자신의 weight로 latent를 생성한다. student가 생성한 latent를 reference에 넣지 않는다.
- tokenizer 확장이나 embedding resize 때문에 초기 student/reference의 marker embedding이 달라지지 않도록 검증한다.
- optimizer update와 checkpoint resume 후에도 reference identity/checksum이 유지되어야 한다. 관련 metadata를 실험 설정에 저장한다.

우선 online reference를 정확성 기준으로 구현한다. reference entropy 캐시는 선택적 최적화다. 캐시를 추가한다면 reference checkpoint, tokenizer/vocabulary mask, 실제 처리된 이미지와 token prefix, latent 설정, window 정의가 일치할 때만 재사용한다. 데이터 augmentation이 달라진 경우 기존 캐시를 쓰면 안 된다.

S_M는 현재 student의 값이므로 학습 전체에 걸친 고정 캐시로 사용하지 않는다.

## 6. M branch: marker/position-preserving local ablation

이번 ablation은 원본 이미지 전체의 제거가 아니라 현재 latent block의 후속 접근 경로 차단이다. 다음 조건을 보존한다.

    input_ids, 원본 image input, shared prefix,
    latent_start/end, latent slot 수, sequence 길이,
    원래 position_ids/RoPE, post target Y_k

기본 intervention은 현재 block k의 latent slot을 decoder의 attention key/value로 읽는 경로를 차단하는 것이다. 원래 causal/padding mask에 다음 restriction을 추가한다.

    query position q >= e_k 이고
    key position j가 s_k < j < e_k인 현재 block의 latent slot이면
    해당 attention edge를 모든 decoder layer/head에서 차단한다.

latent_end query부터 차단해야 한다. post text에서만 차단하고 latent_end가 이미 latent를 읽도록 두면 간접 경로가 남는다. marker 자체의 token/embedding을 제거하지는 않는다.

추가 필수 조건:

- block k마다 독립적인 M_k를 계산한다. 이전 latent block은 정상으로 유지한다. 여러 block을 한꺼번에 ablate한 값을 block별 local gain으로 쓰지 않는다.
- position 계산에 사용하는 원래 padding mask와 ablation용 decoder attention restriction을 분리한다. slot을 padding 처리해 RoPE position을 당기는 구현은 금지한다.
- Stage 2의 prefix 재계산, latent hidden-state 재입력, 마지막 suffix forward에 필요한 restriction이 일관되게 전달되는지 확인한다.
- latent 정보를 이미 읽은 정상 branch의 latent_end/post hidden state나 suffix KV cache를 M에 재사용하지 않는다. 정상 prefix는 causal independence가 확인된 범위까지만 공유한다.
- F/M에서 text target과 그 teacher-forced prefix는 동일하게 유지한다. forward 입력과 cache는 서로 오염시키지 않는다.
- no_grad는 eval과 다르다. F/M의 dropout·augmentation·backend 차이가 gain에 섞이지 않도록 조건을 통제한다. 초기 구현에서 nonzero dropout을 통제하지 못하면 해당 조합을 명시적으로 거부한다.
- gradient checkpointing backward 시 F forward가 ablation mode로 재계산되지 않게 한다. 변경 가능한 전역 flag 대신 forward별 명시적이고 재현 가능한 intervention 인자를 사용한다.

현재 attention backend가 이 restriction을 지원하는지 확인하라. 지원되지 않는 FlashAttention 경로에 mask를 전달하고 작동한다고 가정하지 마라. 먼저 실제 지원되는 eager/SDPA 등 정확성 기준 backend를 명시적으로 선택하고 F/M/reference 조건을 맞춰 검증하라. fallback이나 미지원 조합을 문서화하라.

단순 sequence 삭제, latent embedding을 0으로 치환하는 것, special-token score를 낮추는 것을 위 intervention과 동등하다고 취급하지 않는다. 기본 intervention 구현이 어려우면 범위를 조용히 바꾸지 말고 설계 승인이 필요한 차이로 보고하라.

## 7. Gradient 및 수치 안정성

다음 계산 그래프를 보존하라.

    student F -> CE, H_pre, H_post, S_F -> backward
    current student M_k -> S_M -> detach, no backward
    frozen reference -> H_ref_pre, H_ref_post -> no backward

pre/post entropy나 S_F를 `.item()`, numpy, detach를 통해 loss에 넣지 않는다. 로깅용 detach는 별개다. post loss에서 self-generated latent 계산까지 gradient가 흘러야 한다. 추가 모델/전체 text backbone freeze를 임의로 도입하지 마라.

분모 eps는 numerator에도 같은 값으로 더한다. 선택 위치의 entropy/log-likelihood 연산은 FP32로 수행하고, log_softmax 및 softplus의 안정적인 구현을 사용한다. reference entropy가 매우 작은 위치의 개수를 기록한다. 유효하지 않은 -100 label로 gather하지 않는다.

이상값을 조용히 0으로 바꿔 학습을 계속하지 않는다. block metadata와 원인을 제공하고 명확하게 처리한다. arbitrary D clipping, 새로운 temperature, 추가 margin 변형을 구현하지 않는다.

## 8. 설정, 비용, 분산 학습

다음 의미의 설정을 제공한다. 이름은 repository의 argument 스타일에 맞출 수 있지만 아래 의미와 기본값은 유지하라.

    enable_stage2_objective = false
    lambda_dyn = 0.0
    lambda_vis = 0.0
    reference_model_path = unset
    gamma_H = 0.0
    m_V = 0.0
    entropy_eps = 1e-6
    pre_window = 1
    post_window = 1

새 경로 활성화는 Stage 2에서만 허용한다. 음수 weight, 양수가 아닌 eps/window, dyn 활성화 시 reference 경로 부재를 초기 validation에서 검출한다. eps와 gamma 등의 값은 실험으로 조정할 수 있게 노출하고, 위 값들을 검증된 최적 hyperparameter라고 설명하지 않는다.

lambda_dyn=0이면 불필요한 reference 로딩/forward를 생략한다. lambda_vis=0이면 M forward를 생략한다. 두 값이 0이면 새 경로의 CE-only baseline이 된다. feature 전체가 off이면 legacy 경로와 동일해야 한다.

우선 모든 유효 block에 대해 정확한 objective를 구현한다. block subsampling이나 확률적 ablation scheduling을 조용히 도입하지 마라. block마다 M forward가 필요하다는 계산 비용을 보고하고, prefix 공유나 target 이후 truncation은 결과 동등성을 확인한 뒤 최적화한다. 보조 entropy 때문에 전체 sequence-vocabulary FP32 복사본을 여러 개 추가 생성하지 마라.

DDP/DeepSpeed에서 local block 수가 다를 수 있다. auxiliary의 global block-mean reduction과 gradient averaging의 관계를 정확히 처리하라. zero-valid-block rank도 필요한 collective에 참여해야 한다. gradient accumulation의 reduction 의미도 문서화한다.

특히 ZeRO-3의 parameter-gather 때문에 rank마다 forward 횟수나 latent replay 횟수가 달라질 때 deadlock 위험이 없는지 검토한다. reference 로딩에 단순 GPU deepcopy만 사용해 메모리가 충분하다고 가정하지 마라. 실제 지원하는 multi-forward/ZeRO 구성을 테스트하거나, 미지원 조합은 시작 시 명확하게 차단한다. 검증하지 않은 분산 조합을 지원한다고 주장하지 않는다.

## 9. 필수 테스트: 먼저 작성하고 실패를 확인할 것

### A. 수식과 gradient

- 손으로 계산 가능한 synthetic logits로 text-vocabulary 재정규화와 entropy를 확인한다. 제외된 control logits를 바꿔도 H_T가 달라지지 않아야 한다.
- 같은 student/reference entropy이면 eps를 포함해 r_pre=r_post=1, D=0이어야 한다.
- gamma_H=0이면 D=0에서 loss_dyn=-0.5다.
- 독립적인 entropy scalar 기준으로 loss_dyn의 pre 미분은 음수, post 미분은 양수이며 reference에는 gradient가 없어야 한다.
- S_F=S_M, m_V=0이면 loss_vis=log(2)다. S_F 미분은 -0.5이고 S_M에는 gradient가 없어야 한다.
- D 값이 다른 두 block에서 mean(sigmoid(D_k))를 계산하는지 확인한다. sigmoid(mean(D_k)) 구현을 잡아내라.
- 매우 작은 reference entropy, 작은/큰 logits, bf16 입력에서도 loss와 gradient가 finite해야 한다.

### B. 정렬과 mask

- 위 8-token 예제로 logits[s-1], logits[e], target i와 logits[i-1]의 정렬을 확인한다.
- marker CE는 유지되고 latent pad CE는 기존 규칙대로 제외되어야 한다.
- 다중 block, left/right padding, 짧은 segment, window 중복 제거, assistant 경계, 깨진 marker/truncation, 유효 block 0개를 테스트한다.
- mask helper는 batch>1을 테스트한다. 실제 Stage 2 forward가 batch>1을 지원하지 않으면 silent misuse 대신 explicit error가 나야 한다.

### C. Ablation 정확성

- M에서 input_ids, image input, marker, 길이, position_ids/RoPE와 post target이 보존되는지 확인한다.
- 선택 block만 차단되고 다른 block은 정상인지 확인한다.
- 작은 deterministic causal model에서 선택 latent 값을 바꿔도 M의 post logits가 변하지 않아야 한다. F에서는 그 latent에 의존하도록 구성한 fixture를 사용한다.
- latent_end 경유 leakage, suffix cache 재사용, F/M input mutation을 검출하는 테스트를 작성한다.
- intervention 전 prefix의 결과가 바뀌지 않는지 확인한다.

### D. 통합 및 회귀

- Stage 1과 feature-off legacy 경로의 기존 결과를 보존한다.
- 새 경로의 weight=0 결과가 같은 self-generated-latent CE-only forward와 일치한다.
- entropy-only, vision-only, combined 모드가 각각 의도한 추가 forward만 호출하는지 확인한다.
- model 초기값이 같고 deterministic 평가 조건이 같으면 reference/student에서 D≈0이다. optimizer step 이후 reference weight는 불변이어야 한다.
- combined backward에서 student latent 생성 경로의 gradient를 확인하고, reference와 M branch에 backward가 연결되지 않는지 확인한다.
- gradient checkpointing on/off의 loss와 gradient를 비교하고, M 실행 후 F backward/recompute의 mode 오염을 검출한다.
- 설정 저장·resume 후 reference identity와 objective hyperparameter가 유지되는지 확인한다.

### E. 실제 실행 범위

CPU에서 실행할 수 있는 lightweight unit tests와, repository의 실제 model/trainer 경로를 사용하는 최소 integration test를 분리하라. 대형 checkpoint 다운로드 없이 가능한 테스트는 반드시 실행하라.

GPU 및 checkpoint가 있으면 실제 Stage 2 sample의 forward/backward/optimizer step smoke test를 수행한다. 없으면 실행하지 못한 GPU·ZeRO·실데이터 검증을 정확히 표시하고 재현 명령을 제공한다. mock 테스트만으로 실제 ILVR 학습 성공을 주장하지 마라.

## 10. 로깅과 산출물

다음을 분리해서 기록하라.

    loss_ce, loss_dyn, loss_vis, loss_total
    H_pre, H_post, H_ref_pre, H_ref_post
    r_pre, r_post, D
    S_F, S_M, G
    r_pre>1 비율, r_post<1 비율, G>0 비율
    유효/제외 block 수와 제외 이유, 낮은 reference entropy 수
    실제 추가 forward 횟수와 검증한 backend

로그 평균은 유효 count를 반영한다. D 또는 G 증가만으로 성능 개선을 주장하지 않는다. S_M 저하만으로 gap이 커졌는지와 실제 S_F 상승을 구분할 수 있어야 한다.

산출물:

- 실제 저장소 경로에 맞는 최소 patch와 새 objective 설정.
- 설계 문서 및 실행 계획, 테스트 파일과 실행 결과.
- 같은 Stage 1 checkpoint/data/seed에서 비교할 세 실행 예제: 새 Stage 2 CE-only, CE+entropy, CE+entropy+Vision-gain.
- 경로와 lambda 값은 CLI/환경변수로 입력받는 재현 가능한 실행 script. 존재하지 않는 model/data 경로나 검증된 것처럼 보이는 성능 수치를 하드코딩하지 말 것.
- 수식, boundary/window 정의, reference provenance, ablation 범위, gradient, 계산 비용, 지원/미지원 구성을 설명하는 문서.

문서에는 다음 한계를 정확히 적어라. D의 증가는 pre 상승과 post 하강을 각각 보장하지 않는다. stop-gradient는 M 성능 보존을 보장하지 않는다. 이 ablation은 현재 latent 경로 접근의 기여를 비교하는 것이며, 시각적 의미의 정확성이나 학습 안정성·novelty를 자동으로 증명하지 않는다. auxiliary loss 자체의 감소를 task 성능 개선과 혼동하지 않는다.

최종 보고에는 변경 파일, 핵심 설계, 실제 실행한 테스트 명령과 결과, 실행하지 못한 항목, 재현 가능한 학습 명령, 남은 제한을 포함하라. '완료' 또는 '지원'이라는 표현은 실행 근거가 있는 범위에만 사용하라.

먼저 저장소를 조사하고, 확인한 학습 경로·수정 지점·구현 설계·검증 계획을 제시하라. 위 수식이나 ablation 의미를 변경해야 할 사유가 없다면 이미 합의된 연구 설계는 다시 열지 마라. 구현 설계 승인을 받은 후 실제 구현을 진행하라.
