# 공통 학습 기본 설정 — Stage 1 재현 기준

2026-09-14 사용자 지시: 성공한 Stage 1 재현 설정의 공통 학습 기준을 이후 Stage 1, Stage 2, Stage 2A, Stage 2B, CE-only, preference ON/OFF 및 ablation에 적용한다. 이 문서는 main/ours1 및 다른 worktree·동기화 사본에서 새 실험을 시작하는 모든 세션의 기본 정책이다. 기존 실행의 재개에는 저장된 설정을 우선하고, 기본값을 바꾸어 조용히 이어서 학습하지 않는다.

## 공통 기준

| 항목 | 기본값 |
|---|---|
| Topology | **2 nodes × 2 GPUs = 4 ranks**; 기존 multi-node coordinator 사용 |
| Per-device microbatch | **1** |
| Gradient accumulation | **2** |
| Effective global batch | **8 = 4 × 1 × 2** |
| Learning rate | **5e-5** |
| Seed | **42** |
| Warmup | **10 optimizer updates** |
| Scheduler / optimizer | 기존 linear schedule / AdamW, weight decay 0.01, max grad norm 1.0 |
| 기본 데이터 노출량 | **1 epoch 상당량**; 필터 후 실제 sample 수와 stage protocol로 update budget 결정 |
| 기본 precision | BF16 |
| DoRA를 사용하는 경우 | r32, alpha64, dropout0.05, 기존 decoder target modules와 latent size8 유지 |

튜닝 방법 자체(full/LoRA/DoRA), Stage 2A/B 목적함수, preference beta/weight, teacher 정의, attention backend와 checkpoint 구현을 동일한 것으로 바꾸라는 정책은 아니다. 공통 batch·학습률·schedule 기준은 공유하고 stage별 검증된 알고리즘을 유지한다. 새 실험은 실제 resolved 설정을 남긴다. 사용자가 명시적으로 다른 값을 지정한 실험만 그 범위에서 예외이며, 메모리 여유나 과거 shell 예제만으로 global batch를 바꾸지 않는다.

## Update budget: 431을 모든 stage에 복사하지 않는다

- 기준 Stage 1 재현: 원본 3,450개, global batch 8, 기존 historical remainder 정책에 맞춘 **431 updates**. 이 실행의 431은 모든 데이터셋에서 보편적인 상수가 아니다.
- Stage 2의 retained sample이 3,379개이고 전체를 한 번 처리하는 protocol이라면 목표는 **ceil(3379 / 8) = 423 updates**다. 기존 global batch 16의 212를 재사용하지 않는다.
- 실제 Trainer의 dataloader 길이, sampler padding/drop-last, gradient accumulation 마지막 partial 처리, epoch-step 계산을 확인한다. 이 저장소의 일부 Trainer 경로는 floor 계산을 사용하므로 `epochs=1`만으로 423이 된다고 가정하지 않는다. 필요한 경우 기존 `--max_steps`를 명시하여 protocol에 맞는 budget을 확정한다.
- Stage 2A→2B 통합 실행은 **전체 budget과 A/B별 budget을 각각 기록**한다. 기존 `stage2a_ratio`/phase boundary를 유지해 재계산한다. 전체 1 epoch를 분할하는 protocol이면 각 phase에 1 epoch를 더해 총량을 두 배로 만들지 않는다. 각각 독립 학습하는 실험은 해당 phase의 retained 수와 요청된 노출량으로 계산한다.
- Preference ON/OFF, CE-only, boundary ablation 대조에서는 같은 초기 checkpoint·retained IDs·global batch·update 수·LR schedule·seed를 맞춘다. loss weight 0과 sidecar 계산 자체 OFF를 구분해서 기록한다.
- Resume는 저장된 optimizer/scheduler/global step 및 학습 설정을 유지한다. 새 기본값으로 바꾸려면 기존 run에 덮어쓰지 않고 새 run으로 시작한다.

## 초기 checkpoint와 stage별 설정

새 Stage 2A 또는 Stage 1에서 직접 시작하는 Stage 2B의 DoRA baseline은, 별도 초기화 목적을 명시하지 않은 한, 다음 **재현된 Stage 1 checkpoint**를 사용한다.

```text
checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431
```

일반 Stage 2B가 Stage 2A를 이어받는 protocol이면 해당 A checkpoint를 사용한다. 과거 local checkpoint-215나 기존 Stage 2A artifact를 새 Stage 1 재현 baseline으로 표시하지 않는다. Stage 1 PEFT checkpoint에는 원본 base model과 맞는 로딩 경로를 사용하며, inference용 prepared artifact를 무조건 training resume checkpoint로 대신 넣지 않는다.

Stage 1에는 frozen teacher (`ema_tau=1.0`), 현재 검증된 non-reentrant checkpoint + sample-stable dropout context를 적용한다. Stage 2에는 Stage 1 EMA teacher를 새로 만들거나 dropout/attention 코드를 복사하지 않는다. Stage 2의 native/selective checkpoint 및 clean/vision/sidecar attention backend는 실행할 worktree에서 검증된 조합을 명시한다. ZeRO는 Stage 1 DoRA 재현에서 1이었지만 Stage 2의 검증된 ZeRO-2나 다른 명시된 tuning protocol까지 일괄 ZeRO-1로 바꾸지 않는다. 이런 차이를 동일한 전체 runtime 구현의 재현이라고 주장하지 않는다.

## Launcher와 세션 적용

- `run_multinode_stage1_training.sh` / `run_multinode_stage2_training.sh`가 실행 경로다. 노드별 worker script를 직접 실행하는 single-node fallback은 기존 AGENTS 정책으로 금지된다.
- main과 ours1의 worker launcher 기본값은 Stage 1 microbatch1/accum2/epochs1/LR5e-5/EMA1, Stage 2 microbatch1/accum2/epochs1/LR5e-5로 갱신했다. Stage 1 coordinator의 evidence 기본값도 이에 맞춘다. 명시적 CLI 인자는 계속 우선한다.
- 다른 worktree나 과거 실험 스크립트의 내장 기본값은 다를 수 있다. **이 문서를 기준으로 인자를 명시**하고 그 checkout에서 dry-run 결과를 확인한다. 과거 재현용 snapshot을 일괄 수정하지 않는다.
- 오래된 README/example/matrix sweep의 batch·epoch·LR는 역사적 실행 또는 명시적인 sweep 목적의 값이다. 새 baseline 시작 명령으로 그대로 복사하지 않는다. 학습 sweep도 사용자가 요청한 범위가 아니면 baseline 예외로 취급하지 않는다.
- Launch 전: `world=4, micro=1, accumulation=2, global=8, lr=5e-5, seed=42, warmup=10`과 실제 update budget을 확인한다. 다른 값이 발견되면 기존의 명시적 사용자 요청 또는 resume 목적이 있는지 확인하고, 없으면 launch 전에 고친다.
- 처음 한 번의 설정 확인은 각 노드의 실제 resolved args/evidence로 검증한다. GPU 개수만 보고 global batch나 update 수를 추정하지 않는다.
- 데이터·checkpoint·결과·로그 저장 정책과 동일 commit/호환 환경 preflight는 기존 AGENTS 및 multi-node 가이드를 따른다.

Stage 1 재현 결과는 203/400 (50.75%), train time 약 46분 50초였다. 이 결과가 모든 Stage 2 목적함수의 최적 hyperparameter임을 증명한 것은 아니며, 사용자 지정 공통 baseline으로 채택한다. [실행 설정](../results/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/run.json), [평가 결과](../results/eval-stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/summary.json).

