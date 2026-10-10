# Stage1-v3 — v2 CPU premise experiment outcome

## 결론과 승인 범위

사용자가 [commitment v2](2026-10-10-stage1-v3-commitment-v2.md)의 후속 CPU 실험과
root native 실행·독립 reviewer 방식에 `진행해`로 승인했다. 그 범위의 throwaway 실험을 수행했다.
작은 CPU fixture에서 **checkpoint를 켠 joint R/KL → RV → ordinary gradient 계산이 완료됐고,
독립 oracle과 수치·관측된 호출 계약이 일치했다.** 기존 v1 실패 기록은 보존한다.

새 lifecycle의 제한된 실행 가능성을 뒷받침하는 결과다. Stage1-v3 제품 구현, 실제 Trainer/
DeepSpeed 통합, native-worktree launch, 학습 성능 또는 branch 완료의 증거는 아니다.
제품 코드는 변경하지 않았고 optimizer·GPU·원격 실행·모델 저장도 수행하지 않았다.

승인된 실험의 기준 commit은 `f40ce9f457d905994184278ef7fef2c6bd5676f0`이다.
동결 proposal v2 SHA256 `424c2050661c7a00c48c956219f50e9288bfd019e6c0ceeaa0d87e0a751a0f7e`와
상속 v1의 수식·프로필·호출 예산은 변경하지 않았다.

## 관측 결과

| 비교 | 결과 |
|---|---|
| FP64/FP32 × dropout0/.05 × checkpoint OFF | 4/4 일치 |
| 같은 dtype/dropout × checkpoint ON, 기본 early-stop ON | 4/4 일치 |
| 같은 dtype/dropout × checkpoint ON, 진단 early-stop OFF | 4/4 일치 |
| live parameter와 alias 값을 다르게 둔 추가 조건 | 1/1 일치 |
| alias 재바인딩을 끈 음성 대조 | 완료 후 수치 불일치 검출 |
| lifecycle/수치/ordinary control 집중 테스트 | 26 passed, 최종3.49s |

R/RV score, Text-KL, 모든 trainable gradient와 nonzero shared-prefix gradient, projection을
비교했다. 최대 gradient/projection 절대 오차는 FP64 `1.1796119636642288e-16`, FP32
`1.1660050953585244e-7`이다. 사전 고정 기준 FP64 `atol=1e-9, rtol=1e-7`, FP32
`atol=1e-6, rtol=1e-4`를 유지했다. Text-KL 내부는 양쪽 모두 FP32이고 projection dot/norm은
기존 oracle대로 FP64다. 따라서 이 수치가 실제 BF16/FP32 collective/master 산술을 검증하지는 않는다.

native dropout은 저장소의 실제 `SampleStableStage1Dropout`을 호출한다. scratch v3 context가
호출 identity만 처리한다. packed adapter는 원본 좌표의 mask를 생성하고 실제 stochastic site를
기록한다. original inventory와 물리 layer 순서는 관측 전에 정했다. Python profiler로 실제
`autograd.grad` 진입을 관측한 순서는 `joint_r_kl(2 outputs), rv(1), ordinary(1)`이다.
joint의 두 cotangent lane은 추가 Python score traversal로 계산하지 않는다.

기본 early-stop ON은 실제 도달한 site와 완료된 mask 관측만 검사한다. 일부 native recompute는
mask를 반환하기 전에 중단된다. 관측 mask는 ON10개, OFF16개이고, 방문하지 않은 recompute site의
count0은 허용한다. ON에서 모든 가능한 mask/site set을 직접 관측했다고 주장하지 않는다.

live parameter를 alias와 다르게 두고 재바인딩을 끄면, lifecycle/mask 판정은 일치해도 RV gradient
오차가 `2.947374784918426`, projection 오차가 `1.0458376082749463`이 된다. 실행 예외가 아닌
완료된 수치 불일치로 거부된다. 따라서 이번 fixture는 호출 검사만 통과한 잘못된 derivative를
검출하는 대조를 포함한다. 정상 조건의 shared-prefix gradient norm은 약0.0298–0.0299다.

## 독립 검토와 한계

`/root/v3_v2_probe_review`가 fresh context의 `gpt-6-astra/high`로 spec 준수와 품질을 검토했다.
네 source hash, 이전 oracle과의 byte 일치, 실제 dropout/checkpoint/functional-call 경로와
기록된 증거를 확인했다. 검토자는 새 실험이나 테스트를 실행하지 않았다.
판정은 **이 한정된 CPU feasibility 주장에 Critical/Important 없음, Minor1개 보류**다.

보류 Minor: 예외 주입은 original manager에서만 수행했다. recompute 중 live parameter binding이
달라진 상태에서 예외를 주입하고 graph/unit/phase 및 binding 복원까지 관측한 테스트는 없다.
`finally` 경로는 정적으로 읽었지만 일반적인 exceptional recompute 복원은 입증되지 않았다.
무조건적인 clean review로 표시하지 않는다. 전체 검토와 root 판정은
`logs/stage1-v3-preparation/cpu-probe-v2/independent-review.md`에 보존했다.

검증 경계와 판단의 비용:

- early-stop ON의 미반환 mask·필수 site 전체 집합은 미검증이다. 범위를 넓히면 mask/누락 판정을
  과도하게 일반화할 수 있다.
- mutable buffer·tied weight·parameterization은 fixture에 없다. 실제 모듈에서는 별도 state
  재바인딩 검증이 필요하며, 이 결과로 대신하면 잘못된 derivative를 놓칠 수 있다.
- F는 square-mean scalar와 `autograd.grad`로 소유권만 검증한다. CE/alignment, Trainer `.grad`
  누적·clipping·optimizer·scheduler까지 일반화하면 잘못된 update를 놓칠 수 있다.
- Qwen/DoRA/BF16/GPU/ZeRO·다중 노드·tail/resume는 미실행이다. 원래 제품·네 rank 검증을
  유지하며 이 실험으로 대체하지 않는다.
- native/원격 경로와 전체 suite 상태는 별도다. 이를 합치면 기존 import/launch 문제를 숨기게 된다.
- 단일 sample·두 block·고정 site·단일 thread fixture다. 다른 layout, concurrency, compilation,
  일반 graph inventory까지 지원한다고 가정하면 미검증 소유권 문제가 남는다.

## 전체 CPU suite 결과

TDD가 요구하는 저장소 CPU suite를 별도로 한 번 실행했다. **exit2, 53 deselected,
6 collection errors, 14.14s**다. 실제 test 본문 실행 전에 중단됐다.

실패한 파일은 `tests/multinode/` 아래 다음6개다.

- `test_artifacts.py`
- `test_capacity.py`
- `test_evaluation_coordinator.py`
- `test_lifecycle_matrix.py`
- `test_planned_checkpoint_stop.py`
- `test_vision_gain_probe.py`

모두 unchanged `src/multinode_artifacts.py:48`의 canonical 하위 경로 가정으로 발생했다.
native 실제 경로 `/root/.codex/worktrees/541e/ILVR`에 `relative_to('/root/workspace/ILVR')`를
적용해 실패한다. 경로 수정은 제품 설계에 포함돼 있지만 이번 실험 승인 범위에는 없다.
전체 suite 통과, 제품 수정 완료 또는 branch finish를 선언하지 않는다. 반복 실행/우회하지 않았다.

## 재현과 evidence identity

아래 명령은 모두 현재 worktree에서 repository venv로 실행했다.
공통 환경은 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`
및 `PYTHONPATH=transformers/src:src:.`다.

```bash
./venv/ilvr-venv/bin/python -m pytest -q -p no:cacheprovider .superpowers/sdd/cpu-probe-v2-brief/test_probe_v2.py
./venv/ilvr-venv/bin/python .superpowers/sdd/cpu-probe-v2-brief/probe_v2.py --output results/stage1-v3-preparation/cpu-probe-v2/final-matrix.json
./venv/ilvr-venv/bin/python -m pytest -q -p no:cacheprovider tests -m 'not gpu and not distributed' --basetemp=checkpoints/workflow-validation/stage1-v3-preparation/v2-final-cpu-01/pytest
```

세 번째 명령의 `TMPDIR`도 같은 checkpoints 하위 `tmp`였다. 실행 전 canonical checkpoint 저장소와
동일한 symlink임을 검사했다. 로그는 `logs/stage1-v3-preparation/cpu-probe-v2/`의
`final-focused.log`, `final-matrix.log`, `final-project-cpu.log`다. full suite 출력 전체를 읽었고,
matrix의14개 log record는 최종 JSON과 일치함을 확인했다. 초기/최종 matrix JSON도 byte-identical이다.

scratch 파일 경로는 `.superpowers/sdd/cpu-probe-v2-brief/`다.

| 파일 | SHA256 |
|---|---|
| frozen_oracle.py | `399fff287508bd29b2f020b49926f6035db776fc668bf83da5b7cbae52364da9` |
| lifecycle.py | `dc1507b6ae57c20ddd9741c9d5533e088c9a09ca4954dbd20b50624ab379d7a2` |
| probe_v2.py | `ea332d7f4ab5cd15e38799d90ecf726c36b99c108a3f17c502bf72bd81c1c624` |
| test_probe_v2.py | `45d01eb39678c6b8270b012192c6f6f8e10ab2b215c5c606977deb50a6999966` |
| final-matrix.json | `44d3f9e6cfb0293314fefdbbeebf0d23caf0ac841b1a18462dc7de0722ffa6c9` |

TDD 기록은 stub에서25실패/legacy control1통과 → lifecycle12통과 → 통합26통과 → 독립 검토 →
최종26통과 및13조건 일치다. code/결과·검토 증거를 보존하려고 scratch를 유지한다.
native 절차는 승인된 한 실험에만 적용했고, 명시적으로 요청된 이전 oracle 재사용을 위해 원본을
읽어 새 scratch에 byte 그대로 복사했다. 두 판단의 비용은 절차/artifact 또는 oracle provenance의
모호함이며, 범위·hash·기록으로 구분했다. 제품 구현 방법을 대신 선택하지 않았다.

## 다음 승인 대상

이 증거를 반영한 commitment v2의 **최종 서면 설계 승인**이다. 수정된 수학적 설계나 완화된
허용오차는 없다. 승인되면 `superpowers:writing-plans`로 구현 계획을 작성한다. 실제 제품 코드는
그 계획 검토와 사용자 구현 방법 선택 후 진행하며, 현재 실험을 제품 구현으로 전환하지 않는다.
