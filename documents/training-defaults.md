# 공통 학습 기본 설정 — 새 loss 계수와 역사적 54% 실행 프로필

2026-09-19 사용자 지시: 216/400(54%)을 기록한 `voff-immediate-mask-b05-fixed-20260919-021026`의 실행 프로필을 이후 모든 ILVR 세션의 기본값으로 사용한다. 이 문서는 canonical, main/ours1, 모든 worktree와 동기화 사본에서 새 Stage 1, Stage 2A/2B, CE-only, preference ON/OFF 및 ablation을 시작할 때 적용한다. 이후의 명시적 사용자 override와 저장된 resume 설정은 우선한다. 기존 run을 새 기본값으로 조용히 바꾸지 않으며, 현재 진행 중인 명시적 `until_next_latent` 실행도 자신의 설정을 유지한다.

2026-10-01 승인된 새 기본값: CE weight 0.1, 일반 preference beta 10.0 / margin 0.5 (beta 바깥 차감), 모든 DIL-BCE variant beta 1.0 / margin 0.0. 아래 54% 실행 결과는 기존 CE1 / beta0.5 / margin0 결과이며 새 loss 계수로 54%를 재현했다는 의미가 아니다. 현재 실행과 저장된 resume 계수 및 공식은 그대로 유지한다.

## 공통 기준

| 항목 | 기본값 |
|---|---|
| Topology | **2 nodes × 2 GPUs = 4 ranks**; 기존 multi-node coordinator 사용 |
| Per-device microbatch | **1** |
| Gradient accumulation | **4** |
| Effective global batch | **16 = 4 × 1 × 4** |
| Learning rate | **5e-5** |
| Seed | **42** |
| Warmup | **10 optimizer updates** |
| CE weight | **0.1** (`CE_WEIGHT` / `--ce_weight`; Stage 1 coordinator `--ce-weight`) |
| Preference beta / margin | 일반 preference **10.0 / 0.5**; `-logsigmoid(beta * delta - margin)`으로 beta 바깥 차감. 모든 DIL-BCE (`dil_bce_rf`, `dil_bce_rf_topk`, `dil_bce_rf_topk_reverse`) **1.0 / 0.0**. beta alias는 `STEP_PREFERENCE_BETA` / `--step_preference_beta` 및 legacy `STEP_DPO_BETA` / `--step_dpo_beta` |
| Preference weight | **1.0** (`STEP_PREFERENCE_WEIGHT` / `--step_preference_weight` / `lambda_preference`); 모든 새 preference 학습 variant에 적용 |
| Scheduler / optimizer | 기존 linear schedule / AdamW, weight decay 0.01, max grad norm 1.0 |
| 기본 데이터 노출량 | **1 epoch 상당량**; 필터 후 실제 sample 수와 stage protocol로 update budget 결정 |
| 기본 precision | BF16 |
| DoRA를 사용하는 경우 | r32, alpha64, dropout0.05, `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, latent size 8 |

공통 topology·batch·학습률·schedule을 적용하면서 stage별 목적함수, teacher, EMA 및 ZeRO 요구사항은 유지한다. 모든 새 일반 preference variant와 ablation은 **beta=10.0, margin=0.5**, 모든 DIL-BCE variant는 **beta=1.0, margin=0.0**, 공통 **CE weight=0.1, preference weight=1.0**을 사용한다. DIL은 nonzero margin을 지원하지 않는다. 일반 legacy DPO의 reference score 차이는 delta 안에서 beta와 함께 계산하며 tunable margin과 구별한다. CE 계수는 raw CE에 한 번만 적용하고 preference / mass KL / teacher alignment는 별도 계수를 유지한다. CE-only와 preference OFF는 preference 목적함수를 계속 끈다. 오래된 환경 파일, script fallback, README 예제, 과거 실행값 및 메모리 여유는 기본값 예외의 근거가 아니다.

## Stage 1 DataLoader worker 기본값

2026-10-27 사용자 지시: 이후 새 Stage 1 학습은 **rank당 DataLoader worker 4개**를 사용한다. 4 GPU ranks에서는 CPU 데이터 로딩 worker 총 16개이며, GPU rank 수나 gradient accumulation 값과는 별개다.

기존 환경·실행 설정을 읽은 뒤 coordinator CLI에 `--dataloader-num-workers 4`를 명시하고, 양 노드의 worker CLI `--dataloader_num_workers 4`와 실제 Trainer/DataLoader의 `num_workers=4`를 확인한다. 과거 Stage 1 실행의 workers0이나 parser fallback 0을 다시 계승하지 않는다. 이는 Stage 1 실행 기본값에 대한 사용자 override이며 Stage 2의 별도 실행 프로필을 변경하지 않는다.

과거 실행·checkpoint 기록은 보존한다. 실행 중인 DataLoader에는 설정 파일 수정만으로 적용되지 않으므로, 현재 실행의 중단·재시작 여부는 별도로 확인하고 적용 여부를 실제 실행값과 구분해 보고한다.

## 신규 Stage1-v3 기본 프로필

2026-10-13 사용자 승인으로 `stage1-v3`의 **새 실행**에는 아래 값을 적용한다.
이 변경은 Stage 1 standard나 Stage 2의 목적함수 계수를 변경하지 않는다.

| 항목 | 기본값 |
|---|---|
| Gain weighting | `token_entropy`: 기존 전체 입력 학생 F의 T=1 예측에서 detached raw entropy `H=-sum(p log p)` 사용; `log(V)` 정규화와 상한 1 제한 없음 |
| Alignment weight | `sim_weight=0.1` |
| Gain update / coefficient | `adamw_projected`, `gain_optimizer_weight=10`; legacy post-Adam 변위 계수 `gain_alpha`와 구별 |
| Text KL | `text_kl_beta=0.01`, `text_kl_warmup_ratio=0`: 첫 step부터 weight 0.01 |

Text selector 기본값은 기존 `kl`이며, 별도 LR warmup 10 updates와 공통
CE0.1/global batch16/LR5e-5 프로필은 유지한다. RV 보호 gradient는 entropy로
가중하지 않는다. `--stage1-v3-gain-entropy-mode normalized`는 기존 `H/log(V)`
방식, `raw`는 새 기본값이며 Python CLI에서는 밑줄 표기를 사용한다.

**Resume는 저장된 설정이 우선한다.** 기존 checkpoint의 entropy 방식,
alignment, Gain 방식과 계수(0 포함), Text KL 계수와 warmup을 복원한다.
`gain_entropy_mode`가 없는 과거 checkpoint는 normalized로 해석하고 원본
파일을 재작성하지 않는다. 저장된 목적함수와 충돌하는 명시적 override는
거부하며, 새 기본값 사용에는 fresh run이 필요하다. 구현은
`/root/.codex/worktrees/541e/ILVR`의 `stage1-v3` 경로에 적용되므로 다른
checkout의 지원 여부와 양 노드의 resolved 설정은 실행 전 별도로 확인한다.
이 기본값 변경 자체는 학습 실행을 승인하지 않는다.

## Stage 2 preference 기본 프로필

| 항목 | 기본값 |
|---|---|
| Objective / modality | `context_preference`, V-only (`STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true`) |
| Target / prefix / branch | `immediate_text`, `clean_prefix`, `independent` |
| V-off mask | 현재 step latent 본문 key 차단; START/END는 보이고 supervision 유지 (`SUPERVISE_LATENT_BOUNDARIES=true`, suppress=false); original image 접근 허용 (`block_original_image=false`) |
| Loss | CE weight 0.1, 일반 preference beta10 / margin0.5 (beta 바깥), 모든 DIL-BCE beta1 / margin0, preference weight 1.0, entropy weight 0, 비활성 entropy margin 0.1 |
| Prefix / sampler | shared prefix ON, `reference_random`; source global order/update group과 zero-loss padded tail 유지 |
| Rendering / boundary | `concatenated-v1`; 인접 text/text 경계를 가로지르는 whole token은 원본 문자열·token·CE를 유지한 채 immediate preference span에서만 제외 |
| Attention | decoder `sdpa`, vision `flash_attention_2`, sidecar `flex`, sidecar query budget 1024 |
| Checkpointing | native gradient checkpointing OFF, latent activation checkpointing ON, 기존 sidecar checkpoint 유지 |
| Runtime | ZeRO-2, max train sequence 8192, data workers 0, BF16 |
| Phase / initialization | `stage2a_ratio=0.5`, direct-from-Stage-1 ON; original Qwen2.5-VL-3B base + reproduced Stage 1 checkpoint-431 |

이 알고리즘 프로필은 `/root/workspace/ILVR/.worktrees/voff-reproduction-20260917`의 commit `1a63667bc9db3f54c5cfc350e874797e790e952a` 또는 동일한 수정이 독립 검토된 descendant에 묶인다. main/ours1이나 다른 checkout에 같은 hyperparameter만 설정해도 알고리즘이 동일해지는 것은 아니다. 특히 `immediate_text` 경계 처리, `reference_random` tail/update 처리, context-group별 독립 embedding lookup, clean CE/logit-before-sidecar 순서가 있어야 한다. 양 노드의 code와 environment가 호환되는지 preflight에서 확인한다.

새 계수의 실행은 `/root/workspace/ILVR/.worktrees/dil-loss-defaults-20261001`의 reviewed revision 또는 호환성이 독립 검토된 descendant를 사용한다. CE1-only 제한이 남은 이전 worktree에 계수만 설정하지 않는다. 위 기존 54% code/result 바인딩은 역사적 사실이며 새 계수의 검증 결과가 아니다. 역사적 일반 preference의 nonzero beta-scaled margin은 resume에서 바깥 차감으로 재해석하지 않는다. 정확한 saved 공식이 지원되지 않으면 학습 전 명확히 거부한다.

## Update budget: retained data와 sampler 동작으로 계산한다

- 현재 Stage 2 retained sample 3,379개와 global batch 16의 전체 노출 기본값은 **ceil(3379 / 16) = 212 updates**다. `reference_random`은 3,379개를 3,380 presentations로 만들기 위해 zero-loss slot 하나만 materialize한다. 마지막 optimizer update는 실제 sample 3개와 그 zero-loss slot 하나의 partial accumulation이며, gradient 분모는 configured global batch 16으로 유지한다. 나머지 12개는 padding forward로 만들지 않는다.
- 이전 Stage 1 재현의 **431 updates**와 이전 global-batch-8 Stage 2 계산의 **423 updates**는 역사적 값이다. 새 실행의 보편적인 상수가 아니다.
- retained 수가 달라지면 실제 dataloader 길이, `reference_random` padding/drop-last, gradient accumulation 마지막 partial 처리 및 Trainer update 계산을 다시 확인한다. `epochs=1`만으로 budget을 추정하지 말고 필요한 경우 `--max_steps`를 명시한다.
- Stage 2A→2B 통합 실행은 **전체 budget과 A/B별 budget을 각각 기록**한다. 기존 `stage2a_ratio`/phase boundary를 유지해 재계산한다. 전체 1 epoch를 분할하는 protocol이면 각 phase에 1 epoch를 더해 총량을 두 배로 만들지 않는다. 각각 독립 학습하는 실험은 해당 phase의 retained 수와 요청된 노출량으로 계산한다.
- Preference ON/OFF, CE-only, boundary ablation 대조에서는 같은 초기 checkpoint·retained IDs·global batch·update 수·LR schedule·seed를 맞춘다. CE-only와 preference OFF, 명시적 weight-0 ablation, sidecar 자체 OFF를 구분해 기록한다.
- Resume는 저장된 optimizer/scheduler/global step 및 CE weight, preference beta / margin / weight와 loss 공식 의미를 포함한 학습 설정을 유지한다. 새 기본값으로 바꾸려면 기존 run에 덮어쓰거나 resumed state를 조용히 바꾸지 않고 새 run으로 시작한다. 기존 결과·checkpoint·run.env는 보존한다.

## 초기 checkpoint와 stage별 설정

새 Stage 2A 또는 Stage 1에서 직접 시작하는 Stage 2B의 DoRA baseline은 `/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`의 original base와 다음 **재현된 Stage 1 checkpoint**를 사용한다.

```text
/root/workspace/ILVR/checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431
```

54% Stage 2 checkpoint를 새 학습 initializer로 사용하지 않는다. 일반 Stage 2B가 Stage 2A를 이어받는 protocol이면 해당 A checkpoint를 사용한다. 과거 local checkpoint-215나 기존 Stage 2A artifact를 새 Stage 1 재현 baseline으로 표시하지 않는다. Stage 1 PEFT checkpoint에는 original base model과 맞는 로딩 경로를 사용하며, inference용 prepared artifact를 training resume checkpoint로 대신 넣지 않는다.

Stage 1은 자신의 frozen teacher/EMA (`ema_tau=1.0`), 검증된 non-reentrant checkpoint + sample-stable dropout context 및 reproduced DoRA 경로의 ZeRO-1 요구사항을 유지한다. Stage 2에 Stage 1 EMA를 복사하지 않는다. Stage 2A/B의 명시적 phase 목적함수와 할당은 유지하며, A 뒤의 B는 A checkpoint를 읽는다. Stage 2 54% 기본 경로는 위 표의 ZeRO-2와 attention/checkpoint 조합을 사용한다.

## 검증된 Stage 2 InfiniBand 실행 경로

2026-10-22의 `rdma-stage2-smoke-20261022-062054`는 위 Stage 2 기본 프로필로 2 nodes / 4 ranks / global batch 16에서 2 optimizer updates, 실제 양 노드 `NET/IB`, finite loss/gradient, 원격의 완전한 checkpoint-2, 양 노드 exit 0을 확인했다. `max_steps=212`와 기존 scheduler를 유지한 planned stop이며, 2 updates를 새 학습 budget으로 사용하지 않는다. 정확도·수렴이나 GPUDirect RDMA 활성화의 증거는 아니다.

Stage 2 IB 실행은 [검증된 IB 운영 지침](../docs/multinode-training.md#verified-stage-2-infiniband-execution)에 따라 양 노드의 기존 `llm_kd_rdma`와 `/root/workspace/ILVR/.worktrees/rdma-stage2-20261022`의 `a39fe86774c965022c058f2cec4cc8dd311bab7a` 또는 검토된 호환 descendant를 재사용한다. 이는 corrected-loss baseline의 descendant이며, canonical launcher의 `llm_kd` 경로에 환경변수만 추가하는 것으로 대체하지 않는다. 원본 컨테이너·역사적 artifact·resume 설정은 보존한다. Stage 1·평가·다른 프로젝트의 IB 검증으로 일반화하지 않는다.

## Launcher와 세션 적용

- `run_multinode_stage1_training.sh` / `run_multinode_stage2_training.sh`가 실행 경로다. 노드별 worker script를 직접 실행하는 single-node fallback은 기존 AGENTS 정책으로 금지된다.
- launcher와 source fallback이 이 정책에 맞게 수정되었다고 가정하지 않는다. 다른 checkout이나 과거 스크립트의 내장값은 다를 수 있으므로 환경/run 설정을 **source한 뒤** 이 문서의 전체 applicable profile을 effective environment/CLI에서 명시해 stale 값을 덮어쓴다. 이후의 명시적 사용자 override와 resume만 예외다.
- 새 Stage 2 preference 실행은 적어도 `world=4`, microbatch 1, accumulation 4, global batch 16, LR5e-5, seed42, warmup10, 일반 beta10 / margin0.5 (beta 바깥) 또는 DIL beta1 / margin0, preference weight1, CE0.1, entropy0, `context_preference`, V-only, `immediate_text`, `clean_prefix`, shared-prefix ON, `reference_random`, `concatenated-v1`, DoRA/latent/attention/checkpoint/ZeRO/max-sequence/data-worker 설정 및 초기 checkpoint를 모두 확인한다.
- launch 전 기존 coordinator dry-run을 실행하고 양 노드의 resolved CLI/env, checkout commit, 수정 파일 identity 및 environment 호환성을 기록한다. GPU 개수나 headline hyperparameter만으로 동일성을 추정하지 않는다.
- 오래된 README/example/matrix sweep의 batch·epoch·LR·preference beta/weight는 역사적 실행 또는 명시적인 sweep 목적의 값이다. 새 baseline 시작 명령으로 그대로 복사하지 않는다. 학습 sweep도 사용자가 요청한 범위가 아니면 baseline 예외로 취급하지 않는다.
- 현재 3,379개 Stage 2 profile에서는 `STAGE2_TOTAL_STEPS=212`를 사용한다. 데이터나 phase가 바뀌면 이 값만 복사하지 말고 위 budget 규칙으로 다시 계산한다.
- 데이터·checkpoint·결과·로그 저장 정책과 동일 commit/호환 환경 preflight는 기존 AGENTS 및 multi-node 가이드를 따른다.

## 평가 기본값

Stage 2 기본 평가는 original TEST 400개, greedy decoding (`temperature=0`, `top_p=1`), max new tokens 4096, max sequence length 128000, repetition penalty 1, vision ablation 없음, `external_full_latent_block`를 사용한다. PEFT checkpoint 평가 준비에는 final PEFT artifact와 **ORIGINAL base model**을 함께 지정한다. 평가 설정을 학습 기본값과 별개로 양 노드 resolved evidence에 남긴다.

54% 프로필의 근거는 [학습 실행 기록](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/immediate-mask-fixed-launch.json), [평가 실행 기록](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/immediate-mask-evaluation-launch.json), [216/400 평가 요약](../.worktrees/voff-reproduction-20260917/results/eval-voff-immediate-test400-20260919-083208/summary.json)이다. 2026-09-19 adoption의 실행 근거이며 새 2026-10-01 loss 계수의 54% 재현 증거는 아니다. 또한 54%가 모든 목적함수의 최적값이거나 다른 checkout의 알고리즘 동등성을 증명했다는 뜻은 아니다.

과거 Stage 1 재현 결과는 203/400 (50.75%), train time 약 46분 50초였다. [실행 설정](../results/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/run.json), [평가 결과](../results/eval-stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/summary.json).
