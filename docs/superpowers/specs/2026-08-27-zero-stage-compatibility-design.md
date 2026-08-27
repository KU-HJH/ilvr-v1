# Stage 1/2 ZeRO-1/2/3 호환성 및 배치 최적화 설계

## 배경

현재 검증된 기준 경로는 두 대의 NVIDIA B200에서 실행되는 ZeRO-3이다.
Stage 1은 optimizer-step, gradient 진단, EMA 갱신 및 benchmark evidence가
ZeRO-3 구현에 명시적으로 결합되어 있다. Stage 2는 일반 Hugging Face
Trainer 경로를 사용하므로 ZeRO-1은 작은 분산 save/resume probe에서 이미
동작했으며, sequence-parallel mesh 항목을 제거한 ZeRO-2 설정도 같은 probe를
통과했다. 그러나 기존 `configs/config_stage2.json`은 4-way data parallel과
2-way sequence parallel을 함께 요구하므로 두 GPU용 ZeRO-2 설정으로 사용할
수 없다.

Stage 2의 rank-local 샘플은 latent block 수가 서로 다를 수 있다. ZeRO-3는
각 decoder forward에서 parameter gather collective를 수행하므로 rank별 forward
횟수가 달라지면 분산 학습이 멈출 수 있다. ZeRO-1/2 지원과 별개로 ZeRO-3의
기존 기능을 보존하려면 이 불균형도 제거해야 한다.

## 목표

1. Stage 1 Full/DoRA가 ZeRO-1, ZeRO-2, ZeRO-3에서 학습, 저장 및 resume을
   지원한다.
2. Stage 2 Full/LoRA/DoRA가 ZeRO-1, ZeRO-2, ZeRO-3에서 학습, 저장 및
   resume을 지원한다.
3. ZeRO-3를 기본값으로 유지하고 ZeRO-1/2는 명시적으로 선택한다.
4. 공식 검증 환경은 2×B200으로 한정하되 다른 world size를 runtime에서
   인위적으로 차단하지 않는다.
5. Stage 1에서 effective global batch 16을 고정하고 ZeRO stage별 최적
   micro-batch/gradient-accumulation 조합을 찾는다.
6. capacity, 처리량, VRAM, checkpoint 및 resume 결과를 ZeRO stage에 결합된
   evidence로 기록한다.
7. 기존의 유효한 ZeRO-3 evidence를 재사용하고 장시간 smoke와 validator를
   중복 실행하지 않는다.

## 비목표

- DeepSpeed 자체를 수정하지 않는다.
- 자동 OOM 복구나 실행 중 ZeRO stage 변경을 구현하지 않는다.
- Stage 2의 `per_device_train_batch_size=1` 제약을 완화하지 않는다.
- 1/4/8 GPU 환경을 공식 성능 기준으로 인증하지 않는다.
- 기존 8-rank sequence-parallel 설정을 두 GPU용 ZeRO-2 설정으로 재해석하지
  않는다.
- ZeRO stage가 다른 checkpoint 사이의 optimizer state 변환을 지원하지
  않는다.

## 사용자 인터페이스와 설정

다음 파일을 canonical DeepSpeed 설정으로 제공한다.

- `configs/deepspeed_zero1.json`
- `configs/deepspeed_zero2.json`
- `configs/deepspeed_zero3.json`

기존 `config_stage*.json`은 호환성을 위해 삭제하지 않는다. 새 launcher,
문서 및 benchmark는 canonical 파일만 사용한다. 두 GPU용 ZeRO-2 설정에는
`sequence_parallel_size`나 고정 `data_parallel_size`를 포함하지 않는다.

launcher는 설정 파일을 작게 읽어 다음을 GPU 초기화 전에 검증한다.

- JSON object이며 `zero_optimization.stage`가 정확히 1, 2, 3 중 하나인가
- 명시적인 data/sequence-parallel mesh가 현재 world size와 충돌하지 않는가
- bf16, batch auto 값 및 offload 정책이 지원 범위 안에 있는가

해석된 `zero_stage`, canonical config path, config digest와 world size를 run
manifest에 기록한다. ZeRO config를 생략한 Stage 1 공식 launcher는 ZeRO-3를
사용한다. Stage 2의 비-DeepSpeed 단일 GPU 경로는 계속 허용한다.

## 공통 runtime identity

작은 immutable runtime profile을 도입해 Stage 1/2 launcher, trainer,
checkpoint validator와 benchmark가 동일한 DeepSpeed identity를 소비하게 한다.
profile은 최소한 다음 필드를 가진다.

- `enabled`
- `zero_stage`
- `config_path`
- `config_digest`
- `world_size`

DeepSpeed가 비활성화된 경우 `zero_stage`와 config identity는 비어 있어야 한다.
ZeRO stage나 config digest가 다른 실행의 capacity, throughput 또는 resume
evidence는 서로 재사용할 수 없다.

## Stage 1 optimizer-step과 EMA

현재 `_DeepSpeedStepAdapter`가 담당하는 실제 gradient-accumulation boundary와
`was_step_applied()` 판정을 유지하되 gradient norm 수집을 ZeRO stage별
전략으로 분리한다.

- ZeRO-3 전략은 현재 검증된 pre-step norm witness 경로를 그대로 유지한다.
- ZeRO-1/2 전략은 pinned DeepSpeed 0.18.0 optimizer가 제공하는 partition-aware
  norm 계산과 loss scale을 사용한다.
- 모든 전략은 finite scalar norm, 실제 update 적용 여부 및 accumulation
  boundary를 동일한 결과형으로 반환한다.
- overflow나 skip으로 update가 적용되지 않은 step에서는 EMA를 갱신하지
  않는다.
- 성공한 optimizer step마다 EMA를 정확히 한 번만 갱신한다.

ZeRO-1/2에서는 student parameter가 rank마다 복제되어 있으므로 각 rank가
로컬 EMA teacher를 직접 갱신한다. ZeRO-3에서만 partitioned student/teacher
parameter를 gather한다. DoRA external-parameter 등록도 실제 ZeRO-3에서만
활성화한다.

## Stage 2 rank 균형 sampler

멀티 GPU Stage 2는 모든 ZeRO stage에 동일한 latent-block-balanced distributed
sampler를 사용한다.

1. retained sample을 `latent_block_count` 1~6 bucket으로 나눈다.
2. seed와 epoch으로 각 bucket을 결정적으로 shuffle한다.
3. 한 global micro-step에 배정되는 모든 rank가 같은 block-count bucket에서
   하나씩 받도록 묶는다.
4. bucket 크기가 world size로 나누어지지 않으면 표준 DistributedSampler와
   같이 소수 샘플을 결정적으로 보충한다.
5. 보충된 샘플 수와 bucket별 분포를 manifest/evidence에 기록한다.

이 방식은 모든 rank의 decoder forward 횟수를 일치시켜 ZeRO-3 parameter
gather collective 순서를 보존한다. ZeRO-1/2에도 같은 sampler를 적용해 ZeRO
stage 비교에서 sample stream 차이를 제거한다. world size 1에서는 기존 단일
GPU 순서를 유지한다.

Resume은 seed, epoch, world size 및 이미 소비한 batch 수로 sampler 순서를
결정적으로 재구성한다. 다른 world size로의 resume은 허용하지 않는다.

## 저장과 resume 계약

모든 numbered checkpoint는 다음 identity와 상태를 가져야 한다.

- ZeRO stage, DeepSpeed config digest, world size
- Trainer global step과 RNG state
- optimizer와 scheduler state
- Stage 1 EMA state
- tuning method와 adapter topology
- Stage 2 sampler identity와 latent bucket 통계

Resume은 생성 당시와 같은 ZeRO stage, config digest 및 world size만 허용한다.
다른 일반 학습 실행에서는 world size를 제한하지 않지만, 기존 checkpoint의
분산 ownership을 암묵적으로 변환하지 않는다.

Stage 1/2 PEFT sparse rows는 parameter가 실제로 partitioned된 ZeRO-3에서만
collective gather한다. ZeRO-1/2에서는 rank 0이 replicated state를 직접
저장하고 다른 rank는 같은 경로에 쓰지 않는다. Full과 PEFT checkpoint의
기존 artifact 분류 및 Stage 1→Stage 2 handoff 규칙은 유지한다.

## Benchmark와 최적 batch 탐색

공식 비교는 world size 2와 effective global batch 16을 고정한다. Stage 1은
사용자 요청에 따라 큰 micro-batch부터 다음 순서로 검사한다.

| 순서 | Micro-batch/GPU | Gradient accumulation | Effective global batch |
|---:|---:|---:|---:|
| 1 | 8 | 1 | 16 |
| 2 | 4 | 2 | 16 |
| 3 | 2 | 4 | 16 |
| 4 | 1 | 8 | 16 |

각 후보에 별도의 capacity pass와 throughput pass를 중첩하지 않는다. 동일한
authenticated sample stream을 사용하는 하나의 3 optimizer-step
producer가 OOM 여부, 처리량과 peak VRAM을 함께 기록한다. 성공 후보 가운데
samples/sec가 가장 높은 조합을 선택한다. 선택된 조합만 checkpoint 저장 후
1-step resume을 확인한다.

Stage 2는 구조상 micro-batch/GPU가 1이어야 하므로 공식 2-GPU 비교에서는
gradient accumulation 8을 사용한다. 다른 world size의 runtime은 양의
accumulation 값을 받을 수 있고 실제 effective global batch를 manifest에
기록하지만 공식 최적값으로 인증하지 않는다. Stage 2에서는 batch sweep을
수행하지 않는다.

각 producer evidence는 다음을 포함한다.

- stage와 tuning method
- ZeRO stage/config digest/world size
- micro-batch, accumulation, effective global batch
- 고정 sample-stream identity
- optimizer steps와 samples seen
- samples/sec와 seconds/optimizer-step
- rank별 peak allocated/reserved bytes
- finite loss 및 gradient diagnostics
- success, capacity OOM 또는 integration failure 분류
- checkpoint/resume witness가 있으면 그 identity

기존 ZeRO-3 evidence는 조건과 identity가 정확히 일치할 때 재사용한다. 최종
report와 success publication은 producer witness를 소비하며 모델/checkpoint
tree를 다시 hash하거나 full validator를 재실행하지 않는다.

## 실패 분류

- `capacity_oom`: CUDA allocator가 보고한 실제 capacity OOM
- `integration_failure`: NCCL 오류, collective 순서 불일치, timeout, 잘못된
  checkpoint, nonfinite diagnostic 또는 contract 위반

OOM이 아닌 실패를 batch 축소의 근거로 사용하지 않는다. 잘못된 config,
ZeRO identity 또는 resume identity는 분산 process를 시작하기 전에 가능한 한
일찍 실패시킨다.

## 최소 검증 전략

과도한 smoke와 validation을 피하기 위해 다음만 수행한다.

1. ZeRO profile/config parsing, optimizer strategy, sampler와 evidence schema의
   focused unit tests
2. Stage 1 ZeRO-1/2의 작은 2-rank Full/DoRA optimizer-step 및 save/resume
   lifecycle
3. 이번 feasibility spike에서 이미 통과한 Stage 2 ZeRO-1/2 DoRA save/resume
   witness 재사용
4. block-balanced sampler 변경 후 ZeRO-1/2/3 각각 rank-skew 1-step 확인
5. Stage 1의 후보별 단일 3-step capacity/throughput producer
6. 선택된 Stage 1 조합만 checkpoint→resume 확인
7. 변경된 공통 경로에 대한 가벼운 ZeRO-3 regression 1회

한 producer가 필요한 output과 witness를 생성하면 이를 다시 검증하기 위한
장시간 pass를 실행하지 않는다. 이미 인증된 dependency digest와 기존 ZeRO-3
evidence를 그대로 전달한다.

## 문서 및 결과물

- canonical ZeRO-1/2/3 config와 선택 방법
- Stage 1/2 fresh training 및 same-identity resume 예시
- ZeRO stage별 지원 tuning method 표
- Stage 1 ZeRO stage별 최적 batch/accumulation, VRAM 및 throughput 표
- Stage 2 공식 고정 조합과 ZeRO stage별 호환성 및 peak VRAM witness
- OOM과 integration failure가 분리된 machine-readable evidence

## 완료 기준

1. Stage 1 Full/DoRA가 ZeRO-1/2/3 optimizer lifecycle을 통과한다.
2. Stage 2 Full/LoRA/DoRA가 세 ZeRO 설정을 선택할 수 있다.
3. rank별 latent block 수가 다른 Stage 2 분산 step이 hang 없이 완료된다.
4. Full 및 PEFT numbered checkpoint가 optimizer, scheduler, global step과 필요한
   sidecar를 보존하고 같은 identity로 resume된다.
5. Stage 1의 ZeRO stage별 최적 batch/accumulation 결과가 evidence와 report에
   기록된다.
6. Stage 2의 공식 `micro-batch 1 / accumulation 8`과 ZeRO별 관측 결과가
   기록된다.
7. 기존 ZeRO-3 결과를 불필요하게 다시 계산하지 않고 모든 필수 검증이 한 번씩
   완료된다.
