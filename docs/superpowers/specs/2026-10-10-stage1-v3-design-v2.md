# Stage1-v3 — design proposal v2

상태: 수정 C절과 native worktree 지원 범위를 확인받아 작성한 **최종 승인 전 설계**.
사용자는 CPU 전제 실험 결과를 본 뒤 “v3 전용 context의 VJP별 재계산 처리와 worktree 경로
호환성을 반영한 설계 v2” 진행에 `진행해`라고 답했다. 제품 수정이나 새 실험 실행 승인이 아니다.

이 문서는 [proposal v1](2026-10-10-stage1-v3-design.md)의 계약 전체를 상속하며,
아래 명시한 stochastic lifecycle·경로·검증 부분만 교체/추가한다. v1 SHA256은
`07069c90f11297f7c5408ff280ac9a182b317b540fbd88095e53332dedb1d47a`다.
v1 자체는 수정하지 않는다. v2 승인 대상은 이 문서와 해당 v1의 합성 계약이다.

## 1. 목적, 유지 범위와 관측 근거

목적은 동일 helper target을 쓰는 text-restricted R과 latent-body-off RV의 다음 GT 텍스트
score 차이를 학습하고, F→R Text-KL로 제한된 예측을 clean student에 묶는 것이다.
v1의 A/B, source/token/query mask, shared-prefix derivative, independent packed 실행,
joint R+KL VJP 후 별도 RV VJP, loss/reduction/projection/update 순서, tail/resume 요구를 유지한다.
core decoder 호출 예산 및 추가 세 번째 score traversal 금지도 유지한다.

CE0.1, frozen EMA1.0, one epoch 상당, two nodes/four ranks, micro1/acc4/global16,
LR5e-5/seed42/warmup10/BF16/DoRA32·64·.05/latent8/ZeRO1 등 v1 학습 프로필은 그대로다.
실제 retained/sampler/tail에 따른 budget 계산과 기존 coordinator 사용도 변하지 않는다.

[CPU 전제 실험](2026-10-10-stage1-v3-probe-outcome.md)은 두 layer·한 sample·두 block에서
checkpoint OFF 4조건의 수치 일치와 ON 4조건의 두 번째 VJP 진입 실패를 보였다.
첫 R+KL joint gradient는 8조건에서 일치했다. 현재 context manager는 재진입할 수 없으며,
fresh manager만으로 우회해도 기존 site별 recompute 최대1회 검사가 두 번째 재계산을 거부했다.
이 증거는 unchanged lifecycle 전제를 반박한다. 새로운 v2 lifecycle이 작동한다는 증거는 아니다.

기존 전체 CPU suite는 native worktree 경로 때문에 multinode 모듈 import에서 6개 수집 오류를
냈다. `multinode_artifacts.py:45–49`뿐 아니라 `ClusterSpec.default`, 원격 명령·작업 디렉터리,
전송 source/destination도 양 노드의 repository 절대 경로가 같다고 가정하는 호출부가 있다.
따라서 import 오류 제거만을 native-worktree 지원 완료로 간주하지 않는다.

## 2. 선택한 실행 변경과 대안

선택은 **v3 전용의 유한한 graph/VJP lifecycle**과 **명시적인 세 repository root**다.
이 방식은 승인받은 수학적 목적과 호출 예산을 유지하면서 실제 실패 원인을 수정 대상으로 삼는다.

대안은 모든 score/KL gradient를 한 번의 3-lane VJP로 합치거나 checkpoint를 끄는 것이다.
전자는 현재 two-lane+RV 실행 계약을 바꾸고 새 batching/memory 검증이 필요하다. 후자는 기존
Stage1 checkpoint 프로필과 메모리 조건을 바꾼다. 기존 검사를 전역으로 완화하는 방법은 ordinary
Stage1의 오류 검출을 바꾸므로 선택하지 않는다. 이 대안들은 자동 fallback이 아니다.

## 3. stochastic identity와 소유권

dropout seed 의미는 기존 그대로 `(base_seed, optimizer_global_step, canonical_slot_id, site)`다.
graph role, VJP 이름, packed row, 재계산 횟수를 seed에 추가하지 않는다. 같은 원본 token은
F/Gain-clean/R/RV와 recompute에서 같은 mask를 가져야 한다. selector/reference는 기존 eval/no-grad다.

각 microbatch의 v3 context는 immutable sample rows와 stochastic site inventory를 가진다.
counter의 identity는 seed와 별개로 다음 항목을 포함한다.

- graph role: `F`, `gain_clean`, `packed_replay`.
- checkpoint unit: decoder/vision 구분과 layer 식별자. 실제 참여하는 trainable stochastic site만 집계한다.
- phase: `original` 또는 `recompute`.
- backward role: `ordinary`, `joint_r_kl`, `rv`. original에는 backward role이 없다.

등록되지 않은 graph/unit/site/role은 오류다. original은 graph/unit/site당 정확히1회다.
recompute는 `(graph, unit, site, backward role)`당 최대1회다. 같은 VJP를 다른 이름으로 반복하여
검사를 우회하지 못하도록 backward role별 시작/완료 역시 microbatch당 최대1회로 제한한다.
joint R/KL의 두 cotangent lane은 **하나의 traversal**로 집계한다.

이것은 호출 검사를 끄는 변경이 아니다. ordinary Stage1 OFF 경로는 기존 context와
original1/recompute≤1 계약을 유지한다. v3 dispatch에서만 새로운 상태와 counter를 사용한다.
기존 dropout의 sample seed 및 mask 수식은 재사용하고, 관측 counter 연결만 v3 전용으로 분기한다.

## 4. 상태 전이와 실제 backward 경계

허용되는 순서는 다음과 같다.

```text
NEW -> F original -> gain_clean + packed_replay originals
    -> joint_r_kl -> rv -> ordinary F backward -> CLOSED
```

유효 auxiliary block이0이면 auxiliary originals/VJP를 건너뛰고 F backward로 간다.
beta가0인 경우에도 실행하는 VJP/관측 역할은 명시적으로 기록한다. 역할을 몰래 생략하거나
추가 traversal을 만들지 않는다. 오류가 나면 `FAILED`로 닫고 유효 완료 상태를 발행하지 않는다.

F의 backward는 기존 `Trainer.training_step`/DeepSpeed의 실제 backward 경계에 연결해야 한다.
`compute_loss` 안의 auxiliary VJP는 명시적 role scope로 실행한다. 단순히 public method 선언이나
callback 이름이 존재한다는 사실을 경계 증거로 쓰지 않는다. 계획 단계에서 부모 Trainer 호출과
wrapper를 따라가고, 구현 검증에서는 실제 진입/종료 witness를 요구한다.

| traversal | 재계산이 허용되는 graph |
|---|---|
| ordinary | F |
| joint_r_kl | gain_clean, packed_replay |
| rv | gain_clean, packed_replay |

각 graph의 원본 forward와 alias/buffer·dropout binding은 해당 checkpoint closure에 묶는다.
functional-call scope가 종료된 후에도 recompute가 실제 student parameter로 바뀌지 않도록,
원래 alias tensor와 buffer를 명시적으로 재바인딩한다. shared-prefix KV는 detach하지 않는다.

## 5. 재진입 가능한 checkpoint phase manager

v3의 `context_fn`은 PyTorch가 저장한 manager 객체에 대해 **순차적 재진입이 가능한 manager**를
제공한다. 한 번 쓰면 소모되는 generator manager를 반환하지 않는다. 각 `__enter__`에서 현재
microbatch와 graph를 확인하고 recompute 진입이면 명시적 backward role도 확인한다.
그 진입에 해당하는 ContextVar token/이전 phase를
보관한다. `__exit__`는 예외 여부와 관계없이 정확한 이전 상태를 복원한다.

동일 manager의 중첩/동시 진입, 잘못된 graph/VJP 조합, 닫힌 microbatch의 stale closure,
등록되지 않은 두 번째 VJP는 실행 전에 거부한다. 이는 현재 rank별 순차 실행을 위한 한정된
계약이며 일반적인 재진입·concurrency framework를 추가하지 않는다.

PyTorch checkpoint early stopping은 기본 동작을 보존한다. 그로 인해 방문하지 않은 site의
recompute count0은 허용하되, 실제 방문 site의 중복·잘못된 role은 거부한다. 디버그용
early-stop-OFF trace를 production 기본값으로 바꾸지 않는다. 검증은 기본 ON과 진단 OFF 모두를
실행하며, checkpoint 자체 OFF도 독립 수치 대조에 사용한다.

packed mask adapter도 관측 범위에 포함한다. source별 전체 mask를 seed로 재생성한 뒤 원본
token 좌표로 gather하며, 중복·재배열 좌표와 source valid length를 보존한다. native dropout을
호출하지 않는 packed site도 v3 recorder에 graph/unit/site/phase/VJP를 기록한다. native 호출
기록과 adapter 기록을 구별하고, 둘 중 하나의 무기록/중복이 수치 일치만으로 통과하지 않게 한다.

## 6. 세 root와 경로 계약

기존 coordinator/ClusterSpec/LaunchPlan을 확장해 아래 세 값을 launch identity로 명시한다.
새 coordinator·path registry·자동 checkout 검색·retry framework는 만들지 않는다.

| 의미 | Stage1-v3에서 제안한 값 |
|---|---|
| local repository root | `/root/.codex/worktrees/541e/ILVR` |
| remote container repository root | `/root/workspace/ILVR/.worktrees/stage1-v3` |
| remote host checkout root | `/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-v3` |

remote 값은 **실행 전 검증할 대상**이며 해당 checkout의 존재나 동기화를 이번 설계로 주장하지
않는다. local alias `.worktrees/stage1-v3`는 같은 물리 worktree인지 검증할 수 있지만, 그 문자열을
양 노드 경로가 같다는 근거로 쓰지 않는다. 원격 host/컨테이너·rank·network 정책은 바꾸지 않는다.

canonical 및 기존 in-tree worktree의 현재 root 조합은 기존과 같은 값으로 해석한다.
native root가 canonical 하위가 아닐 때 canonical main으로 fallback하거나 현재 경로를 원격에
그대로 복사하지 않는다. module import/CPU test 수집은 원격 root 유무와 독립적으로 가능해야 한다.
root 확정/검증은 launch-plan 생성·dry-run에서 수행하고, native 실행의 원격 root가 미지정이면
명시적 오류를 낸다. 새로운 사용자 CLI 이름은 구현 계획에서 기존 옵션 관례에 맞춰 확정한다.

명령/경로의 적용 원칙:

- local python/venv/PYTHONPATH/cwd·data·logs·results는 local root를 사용한다.
- remote container의 실행·node-info·소유 프로세스 제어·worker cwd·venv·입력·로그는 remote
  container root를 사용한다. remote host checkout root는 host filesystem 작업에만 사용한다.
- 상대 artifact 이름을 기존 validator로 검증한 뒤 목적지의 대응 root에 붙인다. checkpoint와
  evaluation-staging, result/log의 종류를 보존하고 단순한 전역 문자열 치환을 하지 않는다.
- resume push/pull 및 evaluation artifact helper 등 shared 함수가 서로 다른 root를 받으면
  실제 source는 해당 노드 root, destination은 상대 노드 root로 구성한다. v3에서 호출하지 않는
  기존 경로도 잘못된 root 조합을 조용히 받아들이지 않으며 지원하지 않으면 실행 전에 거부한다.
- 외부 pretrained input이나 symlink-backed data/checkpoint는 기존 허용된 저장 범위 검사를
  각 노드에서 적용한다. local `resolve()` 결과를 remote의 물리 경로로 취급하지 않는다.
- subprocess argv·docker `--workdir` 검증은 plan의 승인된 remote root에 묶는다. 임의 cwd를
  허용하도록 기존 안전 검사를 제거하지 않는다. node별 commit·코드·환경·입력 검증은 유지한다.

새 plan은 세 root를 serialize하고 수신자가 같은 identity를 검증한다. 구형 saved plan/resume는
저장된 동일-root 의미를 유지하거나 정확히 지원하지 못하면 거부한다. 저장된 plan을 로드하면서
새 root로 조용히 재해석하지 않는다. run ID와 현재 worktree의 checkpoints/results/logs 정책은 유지한다.

## 7. 구체적인 수정 경계

v1 Stage1/Qwen 통합에 더해 다음 연결점이 포함된다.

1. `stage1/equivalence.py` 인근의 v3 전용 context/recorder와 checkpoint manager;
   composition의 opt-in 설치 및 trainer 실제 backward 경계 연결.
2. Qwen packed checkpoint closure의 alias/buffer/dropout 바인딩과 명시적 graph/VJP 역할.
3. `multinode.py`의 plan/root 설정과 `multinode_runtime.py`의 node별 command/cwd 구성,
   `multinode_artifacts.py`의 import-time 계산 및 실제 전송 경로.
4. 현재 coordinator의 profile/옵션 전달과 그 영향 범위의 CPU·integration tests.

Stage2 목적함수나 학습 프로필을 바꾸지 않는다. shared coordinator 함수의 root 전달이 필요하면
기존 same-root Stage2 호출의 회귀 검증은 포함한다. native 경로 지원을 이유로 원격 checkout을
만들거나 동기화/학습을 실행하는 것은 이번 설계 작성 범위 밖이다.

## 8. 승인 전 증거와 이후 검증

기존 CPU 실험의 실패를 통과로 바꾸어 표시하지 않는다. v2의 중요한 잔여 전제는 재진입 manager와
역할별 counter가 실제 checkpoint/vmap/alias backward에서 수식과 호출 계약을 함께 지킨다는 점이다.
**새 전제 실험은 별도 명시적 승인이 있어야 실행한다.** 이번 v2 작성 승인을 실행 승인으로
확장하지 않는다. 실험 대신 확정하려면 이 불확실성을 감수한다는 명시적 waiver가 필요하다.

제안하는 최소 후속 CPU 검증은 기존 독립 oracle과 고정 오차 기준을 그대로 쓰되, scratch에서만
v2 lifecycle을 연결하는 것이다. 이전에 실패한 RV VJP까지 완료하고 두 VJP의 shared-prefix
gradient와 전체 projection을 비교한다. 기존 dropout/context의 ordinary mode는 별도 control로
남긴다. 실제 checkpoint 기본 early stopping ON/OFF를 모두 포함한다.

성공 조건에는 다음 행동 검증을 추가한다: graph/VJP별 정해진 original/recompute event,
중복 original·동일 VJP 재시작·중복 recompute·stale closure·잘못된 graph/VJP의 거부,
예외 후 ContextVar 복원, packed adapter의 관측 누락 검출. 오류 종류를 명시해 무관한 예외가
예상 실패로 통과하지 않도록 한다. dtype tolerance는 v1 실험의 값을 유지한다.

경로 수정의 후속 검증은 canonical/in-tree/native 세 입력에 대해 기존 entrypoint에서 plan을
구성하고, source/destination·cwd·python·PYTHONPATH가 각 노드 root를 사용하는지 실제 argv와
serialize/receive 왕복으로 확인하는 것이다. import/수집 회귀6개도 포함한다. 이 CPU command
검증은 원격 checkout 준비나 분산 실행 증거가 아니다. 그 증거는 승인된 실행 전 dry-run과
실제 두 노드 네 rank 테스트에서 따로 확보한다.

v1의 full suite, 실제 ZeRO1 update/master/moments, zero-query/tail/save-resume, 최종 TEST400
기준은 유지한다. 새로운 context나 path 조건에서 기존 결과를 재사용해 통과했다고 주장하지 않는다.

## 9. 동결과 승인 경계

자체 검토 후 proposal v2를 hash/commit으로 동결한다. root hate 및 비용 대비 필요한 macrothink의
독립 read는 모두 이 버전을 읽는다. 비판은 별도 commitment v2로 종합하고 이견·미실행 first-nail·
남은 불확실성을 명시한다. 최종 서면 승인 전 제품 구현/전체 구현 계획으로 진행하지 않는다.
새로운 material 변경은 다음 proposal 버전으로 남긴다. 기존 prototype의 Minor2개는 검증 설계에
반영했지만 과거 코드의 미해결 finding을 소급해 해결됐다고 표시하지 않는다.
