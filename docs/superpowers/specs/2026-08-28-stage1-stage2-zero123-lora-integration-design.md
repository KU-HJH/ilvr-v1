# Stage 1/2 ZeRO-1/2/3 및 LoRA 통합 설계

## 배경

현재 `main`은 ILVR Stage 1 trainer와 ZeRO-3 설정만 실제 학습 경로에
연결한다. 별도 `feat/vcsd-ilvr-stage2` 브랜치는 Stage 1/2 라우팅,
canonical ZeRO-1/2/3 설정, checkpoint identity, Stage 2 Full/LoRA/DoRA,
Stage 1 Full/DoRA 호환성 경로를 구현한다. 해당 linked worktree에는 Stage 2
PEFT target 검증과 adapter-only 저장을 바로잡는 미커밋 수정도 존재한다.

이번 변경은 이 기존 구현을 재사용하여 현재 `main`에 통합하고, 기존 설계에서
legacy-only로 남겨 둔 Stage 1 LoRA를 Full 및 DoRA와 같은 공식 ZeRO 호환
method로 승격한다.

## 목표

1. Stage 1 Full, LoRA, DoRA가 ZeRO-1, ZeRO-2, ZeRO-3에서 fresh training,
   numbered checkpoint 저장 및 same-identity resume을 지원한다.
2. Stage 2 Full, LoRA, DoRA가 ZeRO-1, ZeRO-2, ZeRO-3에서 동일한 lifecycle을
   지원한다.
3. 공식 launcher가 `configs/deepspeed_zero1.json`,
   `configs/deepspeed_zero2.json`, `configs/deepspeed_zero3.json`을 명시적으로
   선택하며 Stage 1 기본값은 ZeRO-3을 유지한다.
4. 현재 main의 사용자 소유 변경과 linked worktree의 승인된 Stage 2 미커밋
   수정은 손실 없이 보존한다.
5. 실제 2-GPU tiny DeepSpeed lifecycle로 두 ILVR stage, 세 tuning method,
   세 ZeRO stage의 18개 조합을 검증한다.

## 비목표

- 전체 크기 3B/7B 모델의 장시간 학습 또는 성능 benchmark를 실행하지 않는다.
- ZeRO stage 또는 world size가 다른 checkpoint 사이에서 optimizer state를
  변환하지 않는다.
- 자동 OOM 복구, retry orchestration 또는 DeepSpeed 자체 변경을 추가하지
  않는다.
- Stage 2의 per-device batch size 1 제약을 완화하지 않는다.
- 기존 8-rank sequence-parallel `configs/config_stage2.json`을 canonical
  2-GPU ZeRO-2 설정으로 재해석하지 않는다.

## 통합 방식

`feat/vcsd-ilvr-stage2`의 커밋된 head를 현재 main에 통합한다. 선별
cherry-pick이나 현재 main에서의 재구현은 Stage 2 model fork, checkpoint,
PEFT 및 sampler 의존성을 누락할 위험이 있으므로 사용하지 않는다.

linked worktree의 다음 승인된 미커밋 변경은 별도 patch로 보존하여 통합 후
적용한다.

- Stage 2 adapter config의 compact target selector를 실제 resolved module과
  비교한다.
- Stage 2 PEFT 저장이 resized base embedding을 포함하지 않도록
  `save_embedding_layers=False`를 적용한다.
- 위 동작의 causal regression test를 포함한다.

통합 전후에 정확한 대상 경로를 확인하고 사용자 소유 untracked 파일과 삭제된
tracked bytecode를 staging하거나 복원하지 않는다.

## 사용자 인터페이스

Stage 1 공식 launcher는 다음 인터페이스를 제공한다.

```text
./run_stage1_training.sh --method full|lora|dora \
  --deepspeed-config configs/deepspeed_zero1.json|deepspeed_zero2.json|deepspeed_zero3.json
```

기존 `--use_lora`는 호환 alias로 유지하되 내부 tuning identity는 `lora`로
정규화한다. `--method lora`와 `--use_lora`가 서로 다른 method를 동시에
지정하는 모호한 입력은 거부한다.

Stage 2는 기존 공식 인터페이스를 유지한다.

```text
TUNING_METHOD=full|lora|dora \
DEEPSPEED_CONFIG=configs/deepspeed_zero1.json|deepspeed_zero2.json|deepspeed_zero3.json \
./run_stage2_training.sh
```

DeepSpeed config는 GPU 초기화 전에 JSON, ZeRO stage, world-size mesh 및
지원되는 bf16/batch/offload 계약을 검증한다.

## Stage 1 LoRA 승격

Stage 1에는 이미 `lora` tuning type, PEFT composition, trainable parameter
boundary, EMA parameter mapping과 adapter checkpoint 경로가 있다. 새 구현은
별도의 LoRA 모델 계층을 만들지 않고 다음 제한만 제거한다.

1. 공식 launcher와 parser가 `lora`를 Full/DoRA와 같은 method로 받는다.
2. ZeRO compatibility diagnostic과 lifecycle identity가 `lora`를 허용한다.
3. LoRA가 정확히 decoder LoRA parameter와 selective latent token row만
   학습하고 visual/base parameter를 열지 않는지 검증한다.
4. ZeRO-1/2는 replicated LoRA/EMA parameter를 rank-local로 갱신하고 rank 0만
   adapter artifact를 저장한다.
5. ZeRO-3은 partitioned LoRA/EMA 및 sparse row 저장에 기존 collective gather
   경로를 사용한다.
6. 성공한 optimizer step에서만 EMA를 정확히 한 번 갱신한다.

Stage 1 Full/LoRA/DoRA는 같은 ZeRO runtime identity와 save/resume 계약을
사용한다. 기존 Full/DoRA benchmark 비교 schema의 의미를 불필요하게 바꾸지
않으며, 이번 작업에서 LoRA 성능 최적화 matrix를 실행하지 않는다.

Stage 1 capacity probe는 micro-batch/GPU 2부터 시작한다. 2가 성공하면 4, 8
순서로 증가시키고, 2가 실제 CUDA capacity OOM으로 실패한 경우에만
micro-batch 1을 fallback으로 검사한다. 별도의 fixed-EGB-16 ZeRO matrix는
`(micro, accumulation)` 후보를 `(2,4)`, `(4,2)`, `(8,1)`, `(1,8)` 순서로
검사하여 첫 실제 실행이 micro-batch 2가 되도록 한다. NCCL, checkpoint 또는
다른 integration failure는 batch 축소의 근거로 사용하지 않는다.

## Stage 2 보존 사항

Stage 2는 feature 브랜치의 명시적 `Stage2Trainer`, self-generated latent
rollout, latent-block-balanced distributed sampler와 Stage 1 checkpoint handoff를
사용한다. 모든 ZeRO stage에서 rank별 latent forward 횟수를 일치시킨다.

Full checkpoint와 PEFT adapter-only checkpoint는 구분한다. LoRA/DoRA 저장은
adapter weight, adapter config, sparse token rows, runtime identity 및 Trainer
state를 보존하며 전체 base embedding을 중복 저장하지 않는다.

## Checkpoint 및 artifact 계약

번호가 붙은 checkpoint는 다음 identity가 동일할 때만 resume한다.

- ILVR stage와 tuning method
- ZeRO stage와 DeepSpeed config digest
- world size
- optimizer, scheduler, global step 및 RNG state
- Stage 1 EMA 또는 Stage 2 sampler/PEFT sidecar

새 산출물은 다음 canonical root에 저장한다.

```text
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/zero-stage-compatibility/
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/results/zero-stage-compatibility/
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility/
```

각 독립 lifecycle은 세 root에서 동일한 collision-resistant experiment ID를
사용한다. 기존 worktree-local output은 수정하거나 검증 결과로 재분류하지
않는다.

## 오류 처리

- 잘못된 config, method, ZeRO identity 또는 resume identity는 분산 process
  초기화 전에 가능한 한 빨리 거부한다.
- CUDA allocator가 직접 보고한 OOM만 capacity 문제로 분류한다.
- NCCL, collective mismatch, nonfinite loss/gradient, checkpoint 누락 및 contract
  위반은 integration failure로 보고하고 batch 축소로 숨기지 않는다.
- 실패 시 기존 로그를 보존하고 원인에 대한 최소 수정과 하나의 causal
  regression test 후 새 run ID로 재실행한다.

## 검증 전략

1. LoRA 공식 CLI/launcher 라우팅이 ZeRO config를 그대로 전달하는 단위 테스트
2. Stage 1 Full/LoRA/DoRA의 optimizer/EMA/checkpoint ZeRO stage 분기 테스트
3. Stage 2 PEFT target 및 adapter-only 저장 regression test
4. 공통 config/runtime identity, Stage 2 sampler 및 checkpoint focused tests
5. Stage 1 batch probe가 micro-batch 2에서 시작해 4, 8로 증가하고, 2의 실제
   capacity OOM에서만 1로 fallback하는 순서/분류 테스트
6. 두 H100 GPU에서 tiny model을 사용한 실제 DeepSpeed one-step save와 one-step
   resume lifecycle 18개:
   - Stage 1: Full/LoRA/DoRA x ZeRO-1/2/3
   - Stage 2: Full/LoRA/DoRA x ZeRO-1/2/3
7. 각 lifecycle에서 finite loss/gradient, optimizer step, checkpoint identity,
   method별 artifact 및 resumed global step을 한 번 확인한다.

한 producer가 필요한 witness를 만들면 같은 조합의 장시간 smoke나 broad test
cycle을 반복하지 않는다.

## 완료 기준

1. 두 Stage의 세 tuning method가 canonical ZeRO-1/2/3 설정을 선택할 수 있다.
2. 18개 tiny distributed lifecycle이 hang 없이 save/resume을 완료한다.
3. Stage 1 LoRA가 DoRA magnitude parameter나 trainable visual/base parameter를
   포함하지 않는다.
4. Stage 2 LoRA/DoRA checkpoint가 adapter-only이고 승인된 target module과
   sparse sidecar를 보존한다.
5. cross-ZeRO/cross-world-size resume은 명확하게 거부된다.
6. 모든 새 checkpoint/result/log가 canonical root에 저장된다.
7. 사용자 소유 변경과 기존 failure log를 덮어쓰거나 삭제하지 않는다.
8. Stage 1 batch probe의 첫 후보는 micro-batch/GPU 2이다.
