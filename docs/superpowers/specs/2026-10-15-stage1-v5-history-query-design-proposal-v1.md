# Stage1-v5 optional history query — design proposal v1

작성일: 2026-10-15. 상태: **자체 검토 후 동결한 제안; 최종 승인 전**.
Worktree: `/root/.codex/worktrees/1061/ILVR`, branch `stage1_v5`.
제품 코드 기준: `148b0e8ead14456d02cc595f0cc32a0752ee22e8`.
승인 기록: `../2026-10-15-stage1-v5-history-query-workflow.md`.

## 1. 목적과 승인 경계

Stage1-v5에서 helper image의 supervision target을 찾는 query에 선택 가능한
방식 하나를 추가한다. 기본 실행은 기존 selector를 유지하며, 새 방식은
reasoning text의 최근성과 과거에 선택한 visual latent의 참조도를 반영한다.
성과는 선택 경로의 계약 준수로 먼저 판단하며, accuracy 개선은 별도 실험이다.

사용자는 옵션 이름 `--query_mode`, legacy/adaptive 구분, 기존 기록의
gamma0.9, 현재 builder 안의 분기와 별도 math/attention/prefix helper 구조,
과거 target이 주입된 teacher prefix, attention mass 보존, 현재 v5의
DoRA/frozen teacher/loss 유지와 검증 범위를 확인했다. 마지막 `ㅇㅋ`는
해당 설계 구간으로 현재-v5용 명세를 작성하는 승인이다.

이 문서는 아직 최종 승인된 명세나 구현 계획이 아니다. 명세의 최종 승인은
계획 작성으로만 이어지며, 구현 방법 선택과 계획 검토는 그다음 단계다.
이 작업에는 새 학습, GPU premise probe 또는 평가 실행 승인이 없다.

## 2. 근거와 현재 코드의 차이

참고 문서는 canonical `documents/ILVR_History_Query_Spec.md`와
`documents/ILVR_History_Query_Plan.md`다. 각각 SHA256:

- `e63bd2cb7deb580c7a7f751c4a7ac814fae7a2b7fd95fee64cd5259e9b151ad8`
- `f9769dea4cead2b187b5d226c017d2cb28db45066f0c7bd0bada809676c03f97`

그 문서들은 구버전 커밋의 full fine-tuning, ZeRO3, 갱신형 EMA와 flat src
구조를 전제로 한다. 이 작업은 현재 v5의 DoRA, ZeRO1, frozen teacher
tau1, 실제 `src/stage1/` 경로와 기존 저장/재개 계약에 맞춘 별도 적용이다.
문서의 gamma0.8, CE1, 새 EMA runtime, 단일 노드 smoke와 새 launcher는
이 설계에 적용하지 않는다. canonical AGENTS와 training-defaults가 우선한다.

현재 확인한 호출 경로:

`CustomTrainerStage1V5.compute_loss`
→ inherited `_prepare_stage1_loss_inputs`
→ `Stage1TeacherTargetBuilder.build`
→ `_build_sample` → `_reasoning_query` → `_select`.

현재 teacher language pass는 sample당 한 번이며 선택한 과거 target을
그 pass에 주입하지 않는다. `textual_states=previous`는 현재 text 평균과
직전 selected raw feature 평균을, `all`은 누적 text token 평균과 selected
raw feature 평균을 사용한다. 이 진술은 소스 관측이며 신규 방식의 runtime
검증 결과가 아니다.

## 3. 범위와 구조 선택

기존 target builder가 legacy/adaptive dispatch를 소유한다. 선택된 구조는
새 별도 builder class를 병렬로 연결하는 대안보다 현재 입력/출력 경계를
재사용하기 쉽다. 다만 prefix 조립과 attention 계산은 builder에 몰아넣지 않는다.

예상 책임 경계:

- `src/stage1/query.py`: frozen config, 검증, text discount와 query 수식.
- `src/stage1/history_attention.py`: effective Q/K 관측과 reduced probability.
- `src/stage1/history_teacher.py`: prefix embedding 조립과 순차 target ledger.
- `src/stage1/targets.py`: 기존 sample ownership/candidate/selector와 mode dispatch.
- 기존 `utils_deepseed`, Stage1 composition/trainer, v5 resume/contract,
  Stage1 launcher와 multinode 전달 경로: 설정 연결 및 저장/복원.

파일별 구현 순서와 소유권은 이후 계획에서 정한다. 기존 metadata와
`Stage1TeacherTargets`를 재사용하며 다른 worktree를 runtime import하지 않는다.

Adaptive v1은 현재 Stage1-v5/Qwen2.5-VL/DoRA/프로세스당 batch1만 지원한다.
기존 v5의 decoder SDPA, vision FlashAttention2, BF16, checkpointing과
ZeRO1 요구사항을 보존한다. Legacy는 기존 지원 범위와 동작을 유지한다.
Stage2, standard Stage1의 adaptive, Qwen3, moving EMA, packed/sliding-window
attention, KV-cache 재사용, learnable gate와 generation 변경은 범위 밖이다.
참고 문서의 추가 text/latent ablation 모드를 이번에 모두 구현하지 않는다.

## 4. 공개 설정

| CLI | 기본값 | 의미 |
|---|---|---|
| `--query_mode` | `legacy` | `legacy`, `adaptive` |
| `--history_text_gamma` | `0.9` | adaptive의 유한한 discount, 0…1 |
| `--history_attention_layers` | `last` | `last`, `last4`, `all` |

`--history_query_mode`는 출시된 옵션이 아니므로 alias를 새로 만들지 않는다.
기존 `--textual_states previous|all`은 legacy에서만 적용한다. Adaptive에서는
discounted-segment text와 attention-mass latent를 사용하며 이 선택을 기록한다.
Adaptive와 함께 `--textual_states`를 명시하면 충돌로 거부한다. Legacy에서
`--history_*` 설정을 명시하면 비활성 옵션을 숨기지 않고 오류를 낸다.
명시 여부는 기존 parser의 explicit-option 추적에 연결한다.

Query/key/head chunk는 각각 64/2048/4로 시작하는 내부 config다. 새 CLI
knob를 늘리지 않으며 resolved 값은 실행 metadata에 남긴다. `last4`에서
layer가 네 개 미만이면 오류다. Unknown mode, nonfinite gamma와 지원하지
않는 profile은 model loading/training 전에 가능한 범위에서 거부한다.

## 5. 위치와 순차 teacher context

현재 collator가 만든 token 좌표와 sample/helper ownership을 사용한다.
단일 assistant trajectory와 assistant 이전의 original image를 지원한다.
Assistant header는 longest-token-pattern match로 제외하고, reasoning span은
header 끝 또는 직전 latent END 다음부터 현재 START 직전까지다. Padding,
image/latent marker와 tokenizer special ID는 text 평균에서 제외한다.
Source 문자열을 retokenize하거나 token 위치를 재배열하지 않는다.

구간 i의 START 위치를 s_i라 하면 teacher 입력은 `input_ids[:, :s_i]`에
대응하는 prefix다. 현재 START와 현재/미래 latent, 미래 reasoning은 없다.
과거 j<i에서 실제 선택한 Kstar_j개의 raw target만 해당 구간 첫 Kstar_j개
pad에 주입한다. 선택하지 않은 pad는 원래 embedding/mask를 유지한다.

Sample별 처리:

1. 구조, mask, helper ownership과 지원 profile을 검증한다.
2. Original image feature와 helper candidate를 frozen teacher로 만든다.
   기존 multiple-helper-to-block mapping과 candidate pooling을 유지한다.
3. Assistant header까지의 prompt prefix를 teacher에 통과시켜 global intent
   u를 한 번 계산한다. 기존 `_user_query`의 prompt/image pooling과 fallback을
   유지한다. 현재 custom wrapper가 반환하는 final-normalized hidden tensor
   계약을 따른다. 이 근사 global-intent helper를 새 latent attention collector로
   복사하지 않는다.
4. i 순서로 prefix를 만들고 이전 target ledger만 주입한다. Backbone에서
   current text/past active latent의 final-normalized hidden state를 얻는다.
5. 아래 수식으로 query를 만들고 기존 `_select`로 현재 helper target을 고른다.
   선택 순서와 target을 ledger에 추가한다.
6. 기존 packed target/mask/selected-count/diagnostic 계약으로 반환한다.

Helper image는 candidate extractor에서만 사용하며 teacher language context에
넣지 않는다. 미래 helper를 미리 encode하더라도 현재 query/target이 달라져서는
안 된다. Original image feature와 ledger는 해당 build 안에서만 재사용한다.

## 6. Query 수식

각 구간의 유효 text hidden 평균을 ell_i라 한다. FP32로 누적한다.

```text
S_i = gamma * S_(i-1) + I[nonempty T_i] * ell_i
D_i = gamma * D_(i-1) + I[nonempty T_i]
d_i = S_i / D_i                 (D_i > 0일 때만 존재)
```

빈 구간도 나이를 증가시킨다. gamma0이면 현재 구간만 사용하고, 현재가
비어 있으면 과거 text로 fallback하지 않는다. gamma1은 token 평균이 아닌
구간 평균이다. 상태는 sample/teacher snapshot/build 경계를 넘지 않는다.
Frozen causal teacher이므로 이미 관측한 과거 text 요약을 사용할 수 있으나
그 전제는 prefix 위치/mask/adapter가 동일하다는 것이다.

P_i는 과거에 실제 target을 주입한 active pad 위치다. 선택 layer/head와
현재 text 위치에 대해 native causal attention probability를 평균하여
각 p의 a_p를 얻는다. Softmax 분모는 p들만이 아니라 모든 유효 causal key다.
Inactive pad도 실제 context에 유효하면 분모에는 포함된다.

```text
g_p = 현재 injected prefix에서의 과거 active pad p의 final hidden state
m_i = sum_p a_p
b_i = sum_p a_p * g_p
v_i = 1 if d_i exists else 0
q_i = (u + v_i*d_i + b_i) / (1 + v_i + m_i)
```

작은 m_i를 1로 정규화하거나 uniform weight로 교체하지 않는다. Ti/P_i가
비어 있으면 m_i=b_i=0이다. Attention은 target-forced teacher context의
직접 참조 proxy이며 student의 인과적 중요도나 confidence라는 주장은 하지 않는다.
Query가 nonfinite이거나 norm<=1e-8이면 오류다.

## 7. Effective DoRA prefix와 attention 관측

`resolve_qwen2_5_vl_model`로 기존 teacher의 effective model에 접근한다.
Adapter를 비활성화/merge하거나 새 plain backbone을 만들어 대체하지 않는다.
Original image와 raw selected targets의 embedding 주입은 현재 wrapper의
규칙과 비교 검증한다. Prefix MRoPE는 original image grid와 같은 token
좌표에서 계산하며 1D arange로 대체하지 않는다.

Teacher는 eval/no-grad이고 gradient가 없다. Bare language backbone에
`inputs_embeds`, prefix mask, MRoPE position IDs를 전달하며 `input_ids=None`,
`use_cache=False`, `output_hidden_states=False`, `output_attentions=False`,
`return_dict=True`를 사용한다. LM head와 all-layer hidden 반환은 필요 없다.
Original batch dict/tensor는 제자리 변경하지 않는다.

선택 decoder layer의 pre-hook으로 실제 position embeddings를 받고 effective
q_proj/k_proj output hook으로 projection을 관측한다. 실제 MRoPE/GQA mapping,
head scaling, causal/padding mask를 적용한다. FP32 streaming log-sum-exp로
전체 key 분모를 구하고 active latent 확률만 유지한다. Logit을 먼저 평균하지
않으며 layer/head별 확률을 구한 뒤 평균한다.

Full L×H×S×S tensor를 만들지 않는다. Observer는 prefix pass에만 활성화하며
global-intent helper, student F/reference/OFF pass에는 남아 있으면 안 된다.
Context manager/finally로 성공/예외 모두에서 hook/cache를 제거한다. Capture
누락/중복과 mask/profile 불일치는 오류다. Attention backend를 변경하지 않는다.

## 8. Candidate, student와 v5 보존 계약

Candidate pooling, cosine ranking, top-p/top-k, stable selection 순서와
variable Kstar를 유지한다. 선택 target을 spatial 순서로 다시 정렬하지 않는다.

현재 반환 계약은 `latent_values [N,H]`, `latent_mask [B,T]`와 기존 metadata다.
N은 mask true 수 및 ledger의 active position 수와 같아야 한다. 선택 target,
주입 위치와 alignment target의 순서는 같으며 alignment는 target t 대
predictor t-1을 유지한다. 참고 문서의 구버전 `[1,N,H]` 반환을 이식하지 않는다.

한 번 만든 target snapshot을 현재 v5의 F/reference/V-off/Text-off에 그대로
공유한다. v5 loss, auxiliary target selection/reduction, original-block 분모,
student dropout/checkpointing, optimizer와 immutable reference를 바꾸지 않는다.
새 teacher 관측은 student RNG stream에 영향을 주어서는 안 된다.

Legacy의 decoder-call 계약은 그대로 둔다. Adaptive에서 M개 latent 구간이면
helper 내부는 prompt 1회와 prefix M회다. Eligible auxiliary target이 있으면
기존 F/reference/packed-OFF 3회를 더해 M+4회이며, checkpoint 재계산/vision
호출은 별도다. 빈 target 등의 분기는 실제 호출로 별도 측정한다. Adaptive를
기존 four-call profile이나 같은 처리량이라고 표시하지 않는다.

## 9. 저장과 재개

현재 `stage1_v5.json`의 execution 안에 versioned `query_config`를 저장한다.
Loss formula ID와 query formula identity를 구분한다. Query identity는
discounted-segment recurrence, injected-prefix hidden, full-causal denominator,
mass-preserving fusion과 prompt-intent 규칙을 식별한다. Adaptive config에는
gamma, layer selector/resolved indices, chunk와 지원 mask/profile을 포함한다.
새 query module들의 content hash도 기존 source evidence에 포함한다.

Resume resolver가 query explicit inputs를 먼저 분리/검증하고 기존 v5
objective resolver에는 그 objective의 필드만 전달한다. 생략한 query 설정은
저장값을 상속하며 mode/gamma/layer/formula/reference 불일치는 학습 전 거부한다.

Query config가 없는 역사적 checkpoint의 query 의미는 legacy로만 해석한다.
그 checkpoint를 adaptive로 재개하거나 기존 파일을 재작성하지 않는다.
기존 reference/source identity와 실행 검증도 유지한다. Source 호환성을
확인할 수 없는 과거 resume은 조용히 승인하지 않고 구체적 이유로 거부한다.
새 objective에는 fresh run이 필요하다. 별도 EMA 갱신/저장 체계는 만들지 않는다.

## 10. 오류, 자원과 실행 범위

Malformed/nested marker, helper ownership 불일치, missing candidate,
image-feature/token 불일치, target/mask 불일치, nonfinite query/attention은
즉시 오류다. Adaptive 오류를 uniform pooling이나 CE-only로 숨기지 않는다.
원래 latent 구간이 없는 정상 sample은 빈 target 계약으로 처리할 수 있다.

Teacher prefix 재실행과 full-context softmax 분모는 추가 연산이다. 비용은
구간 수/길이/layer 수에 의존하며 정확성이나 메모리 안전을 source만으로
보장하지 않는다. 기본 last로 시작하고 실제 지원 길이/최대 구간 구조에서
teacher latency/probe latency/peak VRAM/call count를 측정해야 한다.

실행을 후속 승인받으면 기존 multinode coordinator만 사용한다. Training smoke도
2 nodes×2 GPUs/four ranks이며 global batch16, microbatch1/GAS4, LR5e-5,
seed42/warmup10/BF16과 current-v5 전용 프로필을 양 노드에서 확인한다.
Smoke/훈련을 지금 실행하지 않는다. Checkpoint는 worktree의 checkpoints,
결과는 results, 실행 로그는 logs에 두며 기존 symlink를 보존한다.

## 11. 검증과 acceptance

구현 후 한 번의 집중 검증 묶음과 필요한 최소 integration witness를 사용한다.
각 테스트는 결과 수치/호출/실제 입력에 근거하며 이름이나 interface 선언만으로
통과했다고 판단하지 않는다.

1. Math/layout: gamma0/.9/1, 빈 구간, 작은 attention mass, header/special 제외,
   variable Kstar와 target-position ledger를 검증한다.
2. Attention: tiny 실제 Qwen/DoRA에서 native eager 확률과 reduced observer를
   비교한다. MRoPE/GQA/causal mask/선택 layer/여러 chunk를 포함한다. FP32는
   atol1e-6/rtol1e-5, BF16는 atol5e-3/rtol2e-2를 초기 허용치로 삼고 오차를 기록한다.
3. Prefix: 실제 inputs_embeds를 capture하여 과거 target이 올바른 위치에
   주입됐는지, helper가 LM context에 없는지 확인한다. Native wrapper의
   embedding/position/hidden과 비교하며 observer 유무와 예외 후 무누수를 검증한다.
4. Causality: 미래 text/latent/helper를 바꿔도 현재 query/target은 동일하다.
   현재 helper 변경은 query가 아니라 candidate/선택에만 영향을 줄 수 있다.
5. Integration: legacy target/mask/호출/RNG와 고정 reference의 기준 결과를
   비교한다. Adaptive는 packed target/mask와 v5 공통 snapshot, teacher
   no-grad/student grad, 실제 decoder 호출 수, save/resume과 충돌 거부를 검증한다.
6. CLI/coordinator: default legacy, 지원 범위/명시 충돌, effective 설정 저장과
   양 노드 전달을 dry-run으로 검증한다. 별도 승인을 받은 실제 four-rank
   witness가 있어야만 multi-node/GPU/메모리 지원을 주장한다.

CPU/reference, GPU/BF16, four-rank, 긴 입력 profiling과 benchmark 결과를
서로 구분한다. 미실행은 미검증이다. Unit pass는 accuracy 개선의 증거가 아니다.
완료 시 기존 프로젝트 full suite와 독립 spec/quality review, fresh verification,
해당되는 branch finish gate를 적용한다. 정확한 명령과 실행 방법은 승인된
계획에서 정하며 선택된 구현 방법을 임의로 바꾸지 않는다.

## 12. 자체 검토와 남은 불확실성

자체 검토: placeholder 없음; query/loss identity, 현재 반환 shape, teacher
lifecycle과 source 문서 차이를 명시했다. 기본 legacy 보존과 adaptive의
context 변화, 호출 비용, old-resume 취급을 구분했다. 신규 public surface는
query mode와 필요한 gamma/layer 설정으로 제한했다.

아직 신규 observer/prefix adapter가 없어 실제 effective DoRA/MRoPE native
확률·hidden 일치, 실사용 길이의 비용·VRAM과 성능 효과는 미검증이다.
이 제안을 동결해 hate/macrothink에 동일 버전의 판단 기준으로 제공한다.
Premise 실험은 제안과 별도 승인이 필요하며 이 문서 작성은 그 승인이 아니다.
