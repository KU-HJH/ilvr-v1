# Stage 1/2 최종 학습 worktree 통합 설계

## 상태

- 작성일: 2026-09-03
- 설계 기준: `design commitment v1`
- 구현 대상 branch: `feat/stage1-stage2-final`
- 구현 대상 worktree: `/root/workspace/ILVR/.worktrees/stage1-stage2-final`
- baseline `main`: `85e74e97de3773375ff037690511153c8d1f40cc`
- Stage 2 source: `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0`
- approved dirty PEFT patch SHA-256:
  `0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557`

**Understood as:** 최종 Stage 1 및 Stage 2 학습 코드와 그 실행·검증에 필요한
명세 및 증거를 하나의 compact worktree로 통합하여, 이후 작업자가 다른 Git
worktree의 코드나 문서를 열지 않고 이 worktree만 기준으로 작업할 수 있게 한다.

이 문서의 “독립성”은 Git으로 관리되는 source, configuration, test, 운영 문서의
독립성을 뜻한다. 데이터셋, Python 환경, pretrained model, 학습 checkpoint 같은
대용량 runtime 자산을 Git worktree 안에 복제한다는 뜻은 아니다.

## 배경과 확인된 상태

최종 구현은 한 branch에만 모여 있지 않다.

- 현재 `main`은 최종 Stage 1 true-batch, EMA, 평가 및 공통 multi-node 변경을
  포함한다.
- `feat/vcsd-ilvr-stage2`는 Stage 2 no-cache recurrence, length manifest,
  efficiency probe, PEFT 및 Qwen Stage 2 runtime 변경을 포함한다.
- 성공한 Stage 2 학습은 `feat/vcsd-ilvr-stage2`의 tracked head만이 아니라 해당
  worktree의 미커밋 PEFT checkpoint 수정도 사용했다. 이 수정은 compact PEFT
  target selector 검증과 resized base embedding을 제외한 adapter-only 저장을
  바로잡는다.
- 현재 launcher와 multi-node helper 일부는 `/root/workspace/ILVR`을
  하드코딩한다. 그대로 복사하면 새 worktree의 실행이 기존 checkout에
  의존한다.

따라서 어느 한 branch를 통째로 복사하거나 두 branch를 무조건 병합하지 않는다.
`main`을 기준으로 최종 Stage 1을 보존하고, Stage 2 runtime과 그 직접 의존성 및
미커밋 causal regression fix를 명시적 manifest로 선별 통합한다.

## 목표

1. 새 worktree 하나에서 Stage 1과 Stage 2의 최종 학습 코드를 읽고 실행할 수
   있다.
2. Stage 1의 `main` 후속 수정과 Stage 2 성공 run에 사용된 동작을 함께 보존한다.
3. Full, LoRA, DoRA와 ZeRO-1/2/3의 기존 지원 범위를 축소하지 않는다.
4. single-node 및 multi-node launcher의 repository-local 경로가 실행 중인
   worktree에서 계산된다.
5. 최종 실행법, 관련 설계, source provenance 및 검증 범위를 이 worktree 안에서
   찾을 수 있다.
6. 모델과 데이터 같은 대용량 자산을 Git history나 worktree에 중복 저장하지
   않는다.

## 비목표

- 전체 데이터로 Stage 1 또는 Stage 2를 다시 학습하지 않는다.
- 기존 checkpoint를 복제, 변환 또는 재샤딩하지 않는다.
- 새로운 분산 학습 알고리즘, retry/recovery framework 또는 artifact manager를
  만들지 않는다.
- 과거 설계 문서의 모든 절대경로와 폐기된 명령을 일괄 수정하지 않는다.
- 기존 Stage 1/2 학습 의미, loss, sampler, EMA, PEFT 또는 resume 계약을
  의도적으로 변경하지 않는다.
- 이 통합만으로 기존 성공 run과 수치적으로 동일한 학습 궤적을 증명했다고
  주장하지 않는다.

## 결과물 구조

구현 branch는 위에 고정한 baseline `main` commit에서 만들고 다음 구조를
제공한다. 아래 `stage1-stage2-final/`은 worktree root 자체이며 그 아래에 같은
이름의 디렉터리를 다시 만들지 않는다.

```text
stage1-stage2-final/
├── README.md
├── configs/
├── run_stage1_training.sh
├── run_stage2_training.sh
├── src/
│   ├── stage1/
│   ├── stage2/
│   └── <공통 runtime 모듈>
├── transformers/
│   └── <Stage 2 Qwen fork>
├── tests/
├── docs/
│   ├── final-stage1-stage2/
│   │   ├── README.md
│   │   └── PROVENANCE.md
│   └── superpowers/
│       ├── specs/<선별된 유효 설계>
│       └── plans/<선별된 유효 구현 계획>
├── checkpoints -> <node-local canonical checkpoint storage>
├── data -> <node-local canonical dataset storage>
├── venv -> <node-local canonical environment storage>
├── logs/
└── results/
```

루트 `README.md`는 compact worktree의 canonical entry point로 갱신한다.
`docs/final-stage1-stage2/README.md`는 Stage 1에서 Stage 2로 이어지는 실행
runbook을 제공하고, `PROVENANCE.md`는 통합 source와 기존 run/evaluation
evidence를 기록한다. 같은 내용을 여러 신규 문서로 나누지 않는다.

## Source 통합 계약

### 기준

- Stage 1과 공통 최신 기반은 `main`을 source of truth로 사용한다.
- Stage 2는 `feat/vcsd-ilvr-stage2`의 최종 파일 상태를 source of truth로
  사용하되, 공유 파일은 `main`의 후속 변경을 덮어쓰지 않고 필요한 Stage 2
  hunk만 통합한다.
- branch 전체 merge 또는 broad directory copy를 사용하지 않는다.
- bytecode, cache, 기존 result/log, editor 설정 및 관련 없는 untracked 파일은
  이식하지 않는다.

### Runtime inclusion manifest

정확한 inclusion manifest는 `docs/final-stage1-stage2/PROVENANCE.md`에
파일별 source commit, source blob SHA-1, 최종 blob SHA-1 및 선택 이유를 표로
기록한다. 구현 계획에서는 이 표를 만드는 정확한 명령과 파일 목록을 고정한다.
최소 inclusion set은 다음 범주를 포함한다.

- `run_stage1_training.sh`, `run_stage2_training.sh` 및 실제 호출되는 launcher
- `src/stage1/`의 `main` 최종 학습 경로
- `src/stage2/`의 data, rollout, PEFT, checkpoint, trainer, efficiency 및 length
  manifest 경로
- `src/main.py`, `src/artifact_paths.py`, `src/utils_deepseed.py` 등 두 stage가
  직접 사용하는 공통 runtime
- `transformers/src/transformers/models/qwen2_5_vl/`의 Stage 2 model fork와
  latent rollout
- 위 경로를 직접 검증하는 Stage 1, Stage 2 및 artifact-path tests
- `configs/deepspeed_zero1.json`, `configs/deepspeed_zero2.json`,
  `configs/deepspeed_zero3.json`과 실제 launcher가 참조하는 설정

`feat/vcsd-ilvr-stage2`의 미커밋 변경 중 다음 세 파일의 causal fix는 별도
source patch로 취급하고 provenance에 content hash를 남긴다.

- `src/stage2/checkpointing.py`
- `src/stage2/trainer.py`
- `tests/stage2/test_peft_checkpointing.py`

Stage 1 관련 dirty path-only 변경, tracked bytecode 삭제 및 cache 파일은 source
patch에 포함하지 않는다.

통합 작업 중에는 고정된 source commit과 dirty patch를 읽을 수 있다. “다른
worktree를 열지 않는다”는 완료된 worktree의 운영·유지보수 계약이다. 통합이
완료되면 source worktree를 다시 열지 않아도 provenance 표와 최종 blob만으로
선택 결과를 감사할 수 있어야 한다.

### 보존 지원 조합

새 기능을 추가하지 않고 다음 기존 routing 조합을 보존한다.

| ILVR stage | Tuning method | ZeRO stage |
|---|---|---|
| Stage 1 | Full, LoRA, DoRA | 1, 2, 3 |
| Stage 2 | Full, LoRA, DoRA | 1, 2, 3 |

이는 18개 조합의 launcher/config routing과 checkpoint contract를 보존한다는
뜻이다. 이번 통합에서 18개 full training을 재실행한다는 뜻은 아니다.

### Checkpoint 계약

번호가 붙은 checkpoint는 다음 identity가 일치할 때만 resume한다.

- ILVR stage와 tuning method
- ZeRO stage와 DeepSpeed config digest
- world size
- optimizer, scheduler, global step 및 rank별 RNG state
- Stage 1의 EMA/sparse-row sidecar 또는 Stage 2의 sampler/PEFT sidecar

Stage 2 LoRA/DoRA 저장물은 adapter config와 adapter weight, sparse token-row
sidecar, runtime identity 및 Trainer state를 포함하고 전체 base model weight나
resized base embedding을 포함하지 않는다. Full tuning은 기존 full-checkpoint
계약을 유지한다. 다른 ZeRO stage 또는 world size의 optimizer state를 변환해
resume하지 않는다.

## 경로와 자산 계약

### Repository-local 경로

실행 코드는 다음 경로를 현재 worktree root 기준으로 계산해야 한다.

- local Transformers 및 `src`의 `PYTHONPATH`
- DeepSpeed configuration
- `logs/`와 `results/`
- `./checkpoints`를 통한 checkpoint destination
- `./venv/ilvr-venv/bin/python`

shell launcher는 자기 파일 위치에서 logical repository root를 계산한다. Python
runtime은 자신의 module 위치 또는 명시적으로 전달된 root를 사용한다. active
Stage 1/2 실행 경로가 다른 `.worktrees/<name>`이나 기존 checkout의 source,
config, docs를 열어서는 안 된다.

하드코딩 제거는 active launcher와 그 직접 runtime dependency로 제한한다. 과거
실험 문서의 historical path와 현재 작업에 사용되지 않는 legacy orchestration을
전면 refactor하지 않는다.

### 대용량 자산

대용량 자산은 Git에 추가하지 않는다. 별도 `shared/` 계층이나 범용 bootstrap
system도 만들지 않는다.

- preferred local dataset target: `/raid/workspace/ILVR/data`
- preferred local environment target: `/raid/workspace/ILVR/venv/ilvr-venv`
- existing checkpoint target: `/raid/workspace/ILVR/checkpoints`

필요한 target이 없으면 기존 `/root/workspace/ILVR/data`와
`/root/workspace/ILVR/venv/ilvr-venv`를 삭제하거나 이동하지 않고 복사한 뒤
크기와 핵심 파일을 검증한다. symlink layout은 정확히 다음과 같다.

```text
data -> /raid/workspace/ILVR/data
venv -> /raid/workspace/ILVR/venv
checkpoints -> /raid/workspace/ILVR/checkpoints
```

따라서 Python 경로는 중복된 `ilvr-venv` segment 없이
`./venv/ilvr-venv/bin/python`이 된다. 가상환경 검증과 test 실행은 반드시
`./venv/ilvr-venv/bin/python`을 직접 호출한다.

데이터 검증은 `TRAIN.jsonl`, `TEST.jsonl`, `images_comt/`의 존재와 source 대비
byte size를 확인한다. 환경 검증은 위 Python으로 `torch`, `deepspeed`, local
`transformers` 및 `flash_attn` import가 성공하고 Python 3.11, PyTorch 2.7.1,
CUDA 12.8, FlashAttention 2.8.3 환경을 보고하는지 확인한다. checkpoint
검증은 symlink target이 존재하며 새 model-bearing destination이 그 아래로
해석되는지 확인한다.

각 multi-node host는 같은 logical worktree layout과 동일한 Git commit을
사용하지만 symlink의 physical target은 node-local일 수 있다. 이 통합 작업은
원격 학습을 새로 시작하지 않으며, 원격 자산을 자동 배포하는 기능도 만들지
않는다.

## 문서 계약

최종 worktree의 독자는 다른 worktree를 열지 않고 다음 질문에 답할 수 있어야
한다.

1. Stage 1과 Stage 2의 역할 및 handoff artifact는 무엇인가?
2. Full/LoRA/DoRA와 ZeRO stage를 어떻게 선택하는가?
3. single-node와 multi-node 실행에 필요한 환경변수와 명령은 무엇인가?
4. checkpoint, result 및 log는 어디에 저장되는가?
5. resume identity와 PEFT adapter-only 저장 계약은 무엇인가?
6. 어떤 source commit과 dirty patch가 통합되었는가?
7. 어떤 테스트와 실제 run evidence가 있고, 무엇은 다시 검증하지 않았는가?

canonical runbook은 다음 구현 계보의 spec/plan만 상대경로로 연결한다.

- Stage 2 reimplementation 및 PEFT
- Stage 1 DoRA batched training 및 textual states
- ZeRO compatibility와 Stage 1/2 ZeRO/LoRA integration
- Stage 1 true-batch equivalence, evaluation repair 및 controlled comparison
- multi-node training
- Stage 2 no-cache efficiency

segmented-KV proposal, 구현되지 않은 multi-node sampler proposal, cache 초안,
중복 TODO 및 이전 worktree 전용 guidance는 canonical 문서에서 제외하고 이유를
provenance에 기록한다. 제외 문서는 최종 동작의 근거로 링크하지 않는다. 최종
문서는 다른 worktree의 파일을 현재 지침으로 링크하지 않는다.

## 기존 실행 증거

provenance에는 최소한 다음 lineage를 기록한다.

### Stage 1 source model

- run:
  `stage1-dora-z1-ema-frozen-b1-ga2-e1-lr2p5e5-egb8-20260902-054900`
- selected handoff: `checkpoint-431`
- role: 성공한 Stage 2 run의 Stage 1 base/adapter source

### Stage 2 training

- run:
  `stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3`
- method: DoRA
- ZeRO stage: 2
- topology: 2 nodes, 4 ranks
- learning rate: `5e-5`
- per-device batch: 1
- gradient accumulation: 2
- effective global batch: 8
- epochs: 2
- completed global step: 846
- reported training loss: 0.24126

현재 local training log에서 `learning_rate=5e-05`, ZeRO-2 config digest,
world size 4 및 Stage 1 handoff path를 직접 확인할 수 있다. 구현 단계에서
available log/checkpoint metadata의 hash와 경로를 provenance에 기록한다.

- 직접 확인된 local log:
  `logs/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/train-node1.log`
- 직접 확인된 checkpoint root:
  `checkpoints/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/`
- 직접 확인된 length manifest:
  `results/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/stage2-length-manifest.json`

### Stage 2 evaluation

- evaluation:
  `eval-stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260903-002235-r2`
- reported full score: 201/400, 50.25%
- reported non-overlap score: 149/286, 52.10%
- previously reported Stage 1 score: 45.0%

Stage 1과 Stage 2 평가는 protocol이 완전히 같다고 입증되지 않았으므로
`+5.25 percentage points`는 관측된 점수 차이로만 기록하고 엄밀한 paired
improvement로 표현하지 않는다. 평가 원본이 새 worktree에서 직접 확인되지
않으면 해당 수치는 “historical report”로 표시하고 재검증된 artifact처럼
표현하지 않는다.

설계 시점의 local audit에서는 위 evaluation ID의 원본 result 파일을 찾지
못했다. 따라서 implementation plan이 권위 있는 원본을 다시 식별하기 전까지
평가 수치와 reported training loss는 historical report 등급이며, `checkpoint-846`
디렉터리 존재는 global-step lineage만 보조한다.

## 오류 처리와 안전성

- 사용자 소유 tracked/untracked 변경을 staging, 삭제 또는 복원하지 않는다.
- dirty Stage 2 source는 먼저 patch와 hash로 고정한 뒤 이식한다.
- 첫 검증 게이트가 실패하면 문서 정리나 추가 refactor를 진행하지 않고 누락된
  runtime dependency를 식별한다.
- 기존 data, venv, checkpoint, result 및 log를 덮어쓰지 않는다.
- 자산 복사가 필요하면 source를 보존하고 destination을 새 경로로 생성한 뒤
  검증한다.
- runtime failure가 발생하면 원인을 좁혀 최소 수정과 causal regression test
  하나를 추가한다. 새 recovery abstraction은 만들지 않는다.

## 검증 전략

### 첫 검증 게이트: integration first nail

문서 정리 전에 inclusion manifest와 dirty PEFT fix를 새 worktree에 적용하고
다음을 확인한다.

1. Stage 1의 기존 focused tests가 통과한다.
2. Stage 2 data/rollout/PEFT/checkpoint/trainer focused tests가 통과한다.
3. Stage 2 Qwen forward 및 routing tests가 통과한다.
4. launcher와 artifact-path tests가 새 worktree root를 관측한다.

이 게이트가 실패하면 selective integration이 성공했다고 간주하지 않는다.
통과하더라도 manifest에 포함된 동작과 선택한 test 범위만 입증하며, successful
run의 전체 runtime 또는 수치적 동등성을 입증한다고 확대 해석하지 않는다.

### 경로 독립성

- active Stage 1/2 source, launcher 및 canonical final docs에서 다른 worktree
  경로 참조가 없어야 한다.
- launcher가 선택한 DeepSpeed config, Python executable, log/result/checkpoint
  root가 새 worktree의 logical layout에 속함을 runtime 또는 focused probe로
  확인한다.
- 역사 문서 전체에 대한 문자열 0건은 완료 기준이 아니다.

### 최종 검증

- repository virtual environment를 통한 project full test suite
- Stage 1 및 Stage 2 launcher/config validation
- symlink target과 artifact-root validation
- final documentation link 및 manifest completeness check
- `git diff --check`와 clean tracked worktree 확인

정확한 focused test node와 shell 명령은 implementation plan에서 파일 단위로
고정한다. 최종 변경은 모두 `feat/stage1-stage2-final`에 commit하며 “clean
tracked worktree”는 `git status --short --untracked-files=no`가 비어 있는
상태를 뜻한다. runtime symlink와 생성된 log/result 같은 의도된 untracked
항목은 별도로 보고한다.

장시간 full training은 실행하지 않는다. 따라서 검증 결론은 source 통합,
configuration, checkpoint contract 및 focused runtime behavior 범위로 제한한다.

## 완료 기준

1. `/root/workspace/ILVR/.worktrees/stage1-stage2-final`과
   `feat/stage1-stage2-final`이 존재한다.
2. 새 worktree의 tracked source와 docs만으로 Stage 1/2 구현과 실행 계약을
   이해할 수 있다.
3. active Stage 1/2 실행이 다른 Git worktree의 source, config 또는 docs를
   읽지 않는다.
4. Stage 1 `main` 후속 동작과 Stage 2 successful-run PEFT fix가 inclusion
   manifest와 tests로 보존된다.
5. canonical ZeRO-1/2/3 config와 Full/LoRA/DoRA routing이 유지된다.
6. 새 model-bearing output은 `./checkpoints` 아래에만, result는 `./results`,
   log는 `./logs` 아래에만 기록된다.
7. 통합 README와 provenance가 source lineage, run parameters, 평가 수치 및
   검증 한계를 명확히 구분한다.
8. 첫 검증 게이트와 최종 검증이 통과하고 tracked worktree가 clean하다.

## Critique 기록과 승인 경계

`design proposal v1`에 대한 load-bearing objection은 성공한 Stage 2가 dirty
worktree에서 실행되었기 때문에 선별 이식이 실제 runtime 동작을 누락할 수
있다는 점이다. 가장 싼 falsification은 runtime inclusion manifest와 dirty fix를
먼저 이식한 뒤 focused tests를 실행하는 것이다. 이 first nail은 구현 전에는
실행할 수 없어 아직 미검증이며, 구현의 첫 gate로 남긴다.

독립 검토에서는 compactness가 통과했지만 correctness는 dirty runtime 때문에
불확실했고, worktree independence는 하드코딩 경로 때문에 현재 상태로 실패했다.
이 설계는 정확한 manifest와 active-path 교정 후 focused tests가 통과하는지를
accept/reject 질문으로 유지한다.

`design proposal v1`에서 다음을 변경했다.

- `/raid/workspace/ILVR/shared/` 계층을 제거했다.
- 범용 bootstrap script를 제거했다.
- 과거 문서 전체의 절대경로 정리를 제거했다.
- 신규 문서를 canonical runbook과 provenance 두 개로 제한했다.
- 검증 전 branch를 `release/*`가 아니라 `feat/*`로 정했다.

사용자가 이 spec을 승인하면 별도의 implementation plan을 작성한다. spec 승인은
전체 재학습 생략과 그에 따른 수치적 동등성 미검증을 받아들이는 것이지만,
integration first nail, 경로 독립성, checkpoint 저장 및 full-suite gate를
면제하지 않는다.
