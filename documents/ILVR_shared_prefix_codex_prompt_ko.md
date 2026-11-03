# ILVR Stage 2B의 shared-prefix sidecar 최적화 — Codex 작업 프롬프트

현재 workspace의 ILVR 기반 구현을 수정하라. Stage 1/2와 Step-wise DPO + entropy 학습은 기존 구현을 우선 사용한다. 이번 작업은 **방법론을 바꾸지 않고, step-local V-off/T-off 및 DPO rejected scoring에서 반복되는 clean-prefix 계산을 제거하는 것**이다.

Random step/modality sampling으로 계산을 줄이지 않는다. 모든 valid reasoning step을 처리하되, clean backbone을 공유하고 각 step의 짧은 branch만 병렬 계산하라.

## 1. Superpowers 작업 절차

`using-superpowers`에 따라 관련 skill을 실제로 읽고 적용하라. 먼저 `AGENTS.md`, git 상태, 기존 테스트와 아래 문서를 확인하라. 파일은 workspace 또는 제공된 첨부에서 실제 경로를 찾아 사용하고, 없는 파일이나 구현을 있다고 가정하지 마라.

- `ILVR_stepwise_codex_implementation_spec_ko.md`
- `ILVR_stepwise_modality_dpo_spec_ko.md`
- `ILVR_stepwise_method_problem_spec_ko.md`

`brainstorming`으로 **계산 구조 변경에 한정한 설계**를 작성하고, 필요한 설계 승인 후 `writing-plans`로 실제 수정 파일·인터페이스·테스트를 연결한 계획을 작성하라. 아래의 고정된 연구 방향을 다시 선택하게 하지 마라. 기존 구현과 충돌하거나 correctness를 결정할 수 없는 지점만 질문하라.

구현은 `test-driven-development`로 진행하고, 실패 원인이 불분명하면 `systematic-debugging`을 사용하라. 코드가 독립적으로 분리되는 작업에만 `subagent-driven-development`를 적용하라. 마지막에는 `requesting-code-review`와 `verification-before-completion`을 거쳐 실행한 테스트와 결과를 보고하라. 사용자의 기존 변경사항을 덮어쓰지 마라.

## 2. 보존할 방법론과 loss

Reasoning unit은 다음과 같다.

$$
T_{k-1}\rightarrow V_k=(I_k^{\mathrm{helper}},Z_k)\rightarrow T_k.
$$

$T_{k-1}$은 직전 text block, $Z_k$는 직전 visual latent block, $T_k$는 scoring target이다. $Q$는 질문, $I_0$는 initial image, $P_{k-1}$은 그보다 이전 clean history이다. Helper image의 기존 학습·metadata 역할을 유지하고 새로운 online image input을 추가하지 마라.

$$
C_k^+=(Q,I_0,P_{k-1},T_{k-1},Z_k).
$$

V-off는 현재 step의 prediction에서 $Z_k$에 대한 직접 attention 접근을 차단한다. T-off는 $T_{k-1}$에 대한 직접 접근을 차단한다. $Q$, $I_0$, 이전 clean history와 다른 경로로 이미 전달된 정보는 유지한다. 이 작업은 **local marginal intervention**이며 total information removal이 아니다.

기존 최종 objective를 보존하라.

$$
\mathcal L=\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}
+\lambda_D\mathcal L_{\mathrm{StepDPO}}
+\lambda_H\mathcal L_{\mathrm{Entropy}}.
$$

각 step의 DPO sigmoid는 독립적으로 유지한다. 여러 step의 margin을 먼저 합쳐 하나의 sigmoid에 넣지 마라. 기존 score normalization, valid-pair filtering, entropy margin, loss reduction 및 gradient 정책도 임의로 변경하지 마라.

DPO rejected는 이미 offline 생성된 $T_{k,V}^{-},T_{k,T}^{-}$를 사용하며 **항상 clean context에서 score**한다. Entropy branch에서는 **chosen $T_k^+$를 teacher-force**하고 V/T-off prediction을 계산한다. 두 종류의 branch를 혼동하지 마라.

## 3. 목표 계산 구조: clean + V 그룹 + T 그룹

먼저 다음 세 실행 단계의 shared-prefix 구현을 목표로 하라.

| 실행 단계 | 계산할 내용 | 반환값 |
|---|---|---|
| Clean backbone | 기존 clean interleaved trajectory | clean CE, chosen log-prob, clean entropy, differentiable prefix state |
| V-sidecar 그룹 | 모든 step의 V-off chosen과 visual-origin rejected | V-off entropy, clean-context rejected-V log-prob |
| T-sidecar 그룹 | 모든 step의 T-off chosen과 textual-origin rejected | T-off entropy, clean-context rejected-T log-prob |

한 그룹 안에서도 entropy branch에는 ablation mask가 적용되고 DPO rejected branch에는 적용되지 않는다. 따라서 mask는 **그룹 단위가 아니라 branch 단위**로 지정하라.

```text
step k의 clean prefix
├── chosen T_k+                  : clean backbone의 결과 재사용
├── chosen T_k+ / V-off          : entropy
├── chosen T_k+ / T-off          : entropy
├── rejected T_k,V- / clean      : DPO
└── rejected T_k,T- / clean      : DPO
```

모든 branch의 결과는 clean main path로 되돌아가지 않는다. 각 branch는 자기 step까지만 계산하고 이후 trajectory를 rollout하지 않는다.

**‘3-pass’는 위 실행 단계의 목표이지, 실제 `model.forward()`가 반드시 세 번 호출된다는 보장이 아니다.** ILVR의 self-generated latent가 내부 순차 호출을 요구하면 그 의존성을 보존하라. 반복 prefix 계산과 기존 latent 전개에 필수적인 호출을 구분하여 계측하라. Wrapper만 합쳐 호출 수가 줄었다고 주장하지 마라.

세 그룹의 loss를 합친 뒤 학습 microbatch당 하나의 backward를 기본으로 사용한다. 기존 gradient accumulation과 optimizer-step cadence는 유지한다. 그룹 사이에 optimizer update를 하지 마라.

3-pass 구현의 correctness가 검증된 다음에만 V/T 그룹을 합친 `clean + all-sidecars`의 2-pass 모드를 검토한다. Clean latent 생성까지 억지로 단일 dense Transformer pass로 바꾸는 것은 이번 작업의 요구사항이 아니다.

## 4. Shared-prefix와 attention의 필수 조건

**Prefix 재사용:** clean path의 layer별 K/V 또는 동등한 activation을 한 번 계산하여 공유하라. 각 branch에서 prefix를 다시 projection/MLP 처리하거나, 전체 prefix를 branch마다 복제한 일반 batch를 최종 구현으로 사용하지 마라. Backbone은 수정하지 않고 읽기 전용으로 공유하며, branch-local K/V 저장 공간은 분리하라.

**Branch 격리:** sidecar는 분기 지점까지의 clean ancestor와 자기 branch 내부 이전 위치만 볼 수 있다. 다른 sidecar, 같은 step의 clean target hidden states, clean future에는 접근할 수 없다. 모든 V 또는 T를 하나의 global-corrupted trajectory에서 동시에 제거하는 구현은 금지한다.

**첫 target token:** `target_text_span`과 해당 token을 예측하는 query/logit 위치는 같다고 가정하지 마라. Causal label shift를 추적하고, 첫 target token을 예측하는 boundary query도 branch 조건에 맞게 재계산하라. Clean prefill의 마지막 logits를 V/T-off 첫 token에 그대로 사용하지 마라. 필요한 최소 boundary replay와 latent/control-token 범위를 설계에 명시하라.

**Position과 recurrence:** 각 branch의 logical position, Qwen multimodal RoPE 및 기존 latent feedback 경로를 독립 실행과 동일하게 유지하라. Memory상의 packed index를 그대로 position ID로 쓰지 마라. Branch의 변경된 state가 clean recurrence에 유입되지 않게 하라.

**Gradient:** clean-prefix reuse를 `detach()` 또는 `no_grad()`로 구현하지 마라. Policy clean state로 향하는 기존 gradient를 보존하라. Activation checkpointing의 cache mutation 및 재계산도 검사하라. Dense attention은 correctness fallback으로 허용하되, 성능 모드에서는 가능한 block-sparse 경로를 검증하고 실제 절감량을 측정하라.

## 5. Reference와 비용 계산

Reference policy 및 conditioning semantics는 이번 최적화에서 바꾸지 마라. Reference log-prob cache는 **reference에 제공되는 전체 입력이 고정된 경우에만** 재사용하라.

Frozen checkpoint라도 현재 policy가 생성하는 latent를 reference 입력으로 공유하면 그 입력은 학습 중 달라질 수 있다. 이때 과거 cache를 그대로 사용하지 마라. 현재 구현의 cache key와 checkpoint/tokenizer/template/position/latent-input 조건을 확인하고, 유효하지 않으면 online no-grad reference 경로를 유지하라.

Reference forward를 자동으로 0회라고 계산하지 마라. 다음을 분리해 보고하라.

```text
logical execution groups
실제 policy model/decoder 호출 수
clean latent 전개에 필요한 내부 호출 수
reference 호출 수 및 cache hit/miss
checkpoint 재계산
branch microbatch 수
```

전체 target text 길이가 $L_T=\sum_k|T_k^+|$라면, entropy sidecar의 target token은 $2L_T$이다. DPO sidecar에는 별도로 $\sum_{k,m}|T_{k,m}^-|$가 필요하고 boundary replay 비용도 추가된다. 이는 비용 점검 기준이며 정확한 FLOPs 공식은 아니다. Attention, padding, LM head와 full-vocabulary entropy 비용도 포함하라.

Memory가 부족하면 모든 branch를 빠짐없이 처리하는 deterministic chunking을 허용하되, loss의 전체 분모를 유지하고 호출 증가를 보고하라. Random sampling이나 branch 누락으로 바꾸지 마라.

## 6. 구현 순서와 검증

실제 repository 구조를 조사한 뒤 아래 순서로 진행하라. 기존 모듈과 테스트를 재사용하고 관련 없는 리팩터링은 하지 마라.

1. **Baseline audit:** 현재 Stage 2B의 호출 graph, latent feedback, attention backend, loss, reference 입력을 확인하고 작은 batch를 profile한다. 기존 naive 경로를 기준 구현으로 남긴다.
2. **독립 local reference 검증:** 각 step에서 이전 prefix는 clean하게 유지되고 현재 step에만 intervention이 적용되는지 확인한다. 이 경로가 잘못되어 있으면 먼저 고치고 변경 이유를 기록한다.
3. **Shared-prefix 3-pass:** differentiable clean state, branch metadata, 격리된 grouped replay를 구현한다. 반환 score/entropy는 기존 loss 함수에 연결한다.
4. **동등성 검증:** 아래 테스트가 통과한 뒤 실제 Stage 2B에 통합한다.
5. **성능 검증:** 이후에만 2-pass 병합 및 sparse backend 최적화를 수행한다.

필수 테스트:

- **No-op:** ablation을 끈 sidecar의 logits가 동일 context의 clean 결과와 일치한다.
- **Local isolation:** 앞선 step의 sidecar를 바꿔도 다른 step의 clean-prefix 기반 sidecar 결과는 변하지 않는다.
- **First-token correctness:** 첫 target token의 log-prob 및 entropy에 branch 조건이 실제 반영된다. 첫 token이 다른 chosen/rejected pair도 검사한다.
- **Mask isolation:** 차단 대상 edge만 사라지고, future·다른 branch로의 leakage가 없다. DPO rejected에는 V/T mask가 적용되지 않는다.
- **Loss/gradient equivalence:** dropout을 끈 fp32 toy case에서 naive와 shared-prefix의 logits, 각 loss 항, 전체 loss 및 trainable parameter gradient를 비교한다. 초기 기준은 `atol=1e-5, rtol=1e-4`로 두고 backend별 차이는 근거와 함께 별도 기록한다.
- **Trainability:** reference는 frozen이고 policy의 필요한 latent/prefix gradient는 유지된다. Sidecar에 clean CE를 중복 적용하지 않는다.
- **Regression:** 기능 off일 때 기존 Stage 2 동작이 보존된다. Padding, 가변 step 길이, invalid negative, 빈 target, checkpointing 및 현재 사용 중인 distributed 설정을 검사한다.

Benchmark는 같은 데이터·precision·loss·유효 token 수를 사용하는 naive/3-pass/가능하면 2-pass를 비교한다. $K$를 바꿀 때 총 길이 증가와 branch 수 증가의 효과를 구분하라. Throughput, forward+backward 시간, peak allocated GPU memory, 실제 호출 수 및 prefix 중복 처리량을 기록하라. Compile warm-up은 별도 처리한다.

GPU 또는 필요한 backend가 없어 실행하지 못한 테스트는 명시하고 성능 향상이나 완료를 추정해서 선언하지 마라.

## 7. 범위와 최종 산출물

참고 코드는 `frankxwang/dpo-prefix-sharing`의 prefix visibility 및 sparse-attention 패턴과 `huggingface/trl`의 DPO scoring으로 한정한다. 현재 ILVR fork의 model/trainer를 외부 trainer로 교체하거나 dependency를 일괄 업그레이드하지 마라. 외부 코드 이식은 현재 모델과 버전의 호환성을 확인한 뒤 필요한 부분만 수행하라.

새로운 loss, context-calibrated DPO, reference-free objective, source-level helper-image perturbation, online negative generation은 추가하지 마라.

최종 보고에는 수정 파일, 재현 명령, 테스트 결과, naive 대비 score/gradient 오차, 실제 계산 비용, 남은 제약을 포함하라. 세 기존 명세 중 forward-count와 cache 관련 설명은 검증 결과에 맞게 갱신하라.

**완료 기준은 ‘forward라는 함수 호출을 세 번으로 묶음’이 아니라, 모든 valid step의 기존 objective와 local semantics를 보존하면서 반복 clean-prefix 계산을 실제로 제거했음을 테스트와 계측으로 입증하는 것이다.**
