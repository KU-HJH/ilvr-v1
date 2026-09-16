# ours1 학습 및 평가 hyperparameter 설정

이 문서는 2026-09-06부터 실행한 `ours1` DoRA Stage 2A/2B 실험의 실제
launch 설정을 기록한다. Stage 2B의 CE, CE+Preference, CE+Entropy는 동일한
Stage 2A checkpoint에서 서로 독립적으로 시작했으며, optimizer와 scheduler도
variant마다 새로 초기화했다.

## 실행 상태

| 구분 | Run ID | 상태 | 최종 step |
| --- | --- | --- | ---: |
| Stage 2A | `ours1-stage2a-dora-z2-20260906t040535z-r3` | 완료 | 423 |
| CE | `ours1-ctx-ce-dora-20260906-150356` | 완료 | 423 |
| CE+Preference | `ours1-ctx-pref-dora-20260906-150356` | 완료 | 423 |
| CE+Entropy | `ours1-ctx-entropy-dora-20260906-150356` | 완료 | 423 |
| Full | `ours1-ctx-full-dora-20260906-150356` | 사용자 요청으로 중단 | 14/423 |

Full은 `save_steps=423`이어서 중간 checkpoint가 없다. 나중에 실행할 때는
Stage 2A `checkpoint-423`에서 새로운 run으로 다시 시작해야 한다.

## Learning rate 5e-5 재학습 campaign

2026-09-07에 기존 Stage 2A `checkpoint-423`을 공통 초기값으로 유지하고
Stage 2B 세 variant를 learning rate `5e-5`로 다시 학습하는 순차 campaign을
detached/background로 시작했다. 각 variant는 Stage 2A adapter weight만 읽고
optimizer와 scheduler를 새로 초기화한다.

| 순서 | Variant | Run ID | 시작 시 상태 |
| ---: | --- | --- | --- |
| 1 | CE | `ours1-ctx-ce-dora-lr5e5-20260907-050655` | 실행 중 |
| 2 | CE+Preference | `ours1-ctx-pref-dora-lr5e5-20260907-050655` | 대기 |
| 3 | CE+Entropy | `ours1-ctx-entropy-dora-lr5e5-20260907-050655` | 대기 |

Campaign log는
`logs/ours1-context-dora-lr5e5-20260907-050655/campaign.log`에 기록한다.
이 campaign은 아래 공통 설정에서 learning rate만 `1e-5`에서 `5e-5`로
변경하며, variant별 loss coefficient를 포함한 나머지 학습 조건은 유지한다.

## 공통 모델 및 DoRA 설정

| 항목 | 값 |
| --- | --- |
| Original base model | `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct` |
| Stage 1 adapter | `checkpoints/stage1-dora-z1-ema-frozen-b1-ga2-e1-lr2p5e5-egb8-20260902-054900/checkpoint-431` |
| Stage 2B 초기 checkpoint | `checkpoints/ours1-stage2a-dora-z2-20260906t040535z-r3/checkpoint-423` |
| Tuning method | DoRA |
| Rank | 32 |
| Alpha | 64 |
| Dropout | 0.05 |
| Target module suffixes | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Latent size | 8 |
| Latent boundary CE supervision | 활성화 |

Stage 2B variant는 이전 variant의 결과를 이어서 학습하지 않는다. 세 완료
variant와 Full은 모두 위의 동일한 Stage 2A checkpoint를 입력으로 사용한다.

## 공통 최적화 및 분산 학습 설정

| 항목 | 값 |
| --- | --- |
| Learning rate | `1e-5` |
| Optimizer | `adamw_torch` |
| Weight decay | `0.01` |
| LR scheduler | Linear |
| Warmup | 10 optimizer steps |
| Gradient clipping | `1.0` |
| Precision | BF16 |
| DeepSpeed | ZeRO-2 |
| Nodes / GPUs | 2 nodes × 2 GPUs |
| World size | 4 |
| Rank-local microbatch | 1 |
| Gradient accumulation | 2 |
| Effective global batch | 8 |
| Gradient checkpointing | 활성화 |
| Seed | 42 |
| Dataloader workers | 0 |
| `OMP_NUM_THREADS` | 1 |
| Stage 2B optimizer steps | variant당 423 |
| Save interval | 423 steps |
| Stage 2 전체 budget | 846 steps: Stage 2A 423 + Stage 2B 423 |
| Stage 2A ratio | 0.5 |

Stage 2A도 동일한 DoRA 크기, learning rate, batch 구성, BF16, ZeRO-2,
gradient checkpointing, seed, 최대 sequence length를 사용했다. Stage 2A는
`mode=stage2a`로 CE만 학습했고, Stage 2B는 `mode=stage2b`로 실행했다.

## 공통 데이터 설정

| 항목 | 값 |
| --- | --- |
| Training source | `data/TRAIN.jsonl` |
| Image root | `data/` |
| Raw rows | 3,450 |
| Retained rows | 3,379 |
| Filtered rows | 71 |
| Sampler-padded rows per epoch | 3,384 |
| Microbatches per rank | 846 |
| Optimizer steps per epoch | 423 |
| Maximum clean sequence length | 8,192 tokens |
| Assistant rendering | `adjacent-text-space-lf-v1` |

모든 Stage 2B variant는 동일한 source, filtering, rendering, seed를 사용한다.

## 공통 Stage 2B context 설정

| 항목 | 값 |
| --- | --- |
| Step preference enabled | `true` |
| Mode | `stage2b` |
| Objective | `context_preference` |
| CE coefficient | 1 |
| Preference beta | 0.1 |
| Preference margin | 0 |
| Entropy margin | 0.1 |
| Language attention | SDPA |
| Vision attention | FlashAttention 2 |
| Sidecar attention | FlexAttention |
| Sidecar query budget | 1,024 |
| Reference model forward | 0 |
| Negative generation/rollout | 0 |
| Maximum scoring decoder forwards | microbatch당 3 |

CE variant도 같은 context route와 계측을 사용하지만 auxiliary coefficient가
0이므로 최종 loss에는 CE만 포함된다. 기존 latent decoder generation은 scoring
decoder 3회 제한과 별도로 계측된다.

## Variant별 loss coefficient

최종 loss는 다음 형태다.

\[
\mathcal L = \lambda_{CE}\mathcal L_{CE}
+ \lambda_P\mathcal L_{ContextPref}
+ \lambda_H\mathcal L_{Entropy}
\]

| Variant | `lambda_CE` | `lambda_P` | `lambda_H` |
| --- | ---: | ---: | ---: |
| CE | 1 | 0 | 0 |
| CE+Preference | 1 | 1 | 0 |
| CE+Entropy | 1 | 0 | 0.1 |
| Full | 1 | 1 | 0.1 |

Full 행은 계획·launch 설정을 나타낸다. 해당 run은 14/423 step에서 중단되어
완료된 학습 artifact가 아니다.

## 공통 평가 설정

Stage 2A, CE+Preference, CE+Entropy 비교에는 다음 조건을 사용한다.

| 항목 | 값 |
| --- | --- |
| Evaluation source | `data/TEST.jsonl` |
| Rows | 400 |
| Task | `comt` |
| Shards | 4, GPU당 100 rows |
| Maximum sequence length | 32,768 |
| Maximum new tokens | 4,096 |
| Repetition penalty | 1.05 |
| Temperature | 0 |
| Top-p | 1.0 |
| Primary metric | Exact-match `correct/400` |
| Secondary metric | Valid-gold accuracy |

TEST의 SHA-256은
`6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`이다.
현재 독립적으로 복원된 train/test non-overlap subset은 없으므로 이 평가로
objective 자체의 일반적 우열이나 non-overlap 일반화 성능을 주장하지 않는다.

## 설정 근거

- Stage 2A launch: `logs/ours1-stage2-20260906t040535z/launch-stage2a-r3.sh`
- Stage 2B launches: `logs/ours1-context-dora-20260906-150356/launch-{ce,pref,entropy,full}.sh`
- Campaign notes: `results/ours1-context-dora-20260906-150356/run-notes.md`
- Stage 2A completion witness: `results/ours1-stage2a-dora-z2-20260906t040535z-r3/initial-train-witness.json`
- Length manifests: `results/ours1-ctx-*-dora-20260906-150356/stage2-length-manifest.json`
