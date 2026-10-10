# Stage1-v3 — design commitment v2

상태: **CPU 전제 실험의 제한된 feasibility 결과를 반영했으며, 최종 서면 설계 승인 대기**.
처음 `진행해`는 v2 작성, 이후 `진행해`는 아래 CPU 실험과 root native 실행·독립 검토의 승인이다.
제품 구현·전체 구현 계획 승인으로 확대하지 않는다. [실험 결과와 한계](2026-10-10-stage1-v3-probe-v2-outcome.md)를 함께 검토한다.

승인 대상 설계는 [proposal v2](2026-10-10-stage1-v3-design-v2.md)와 그 문서가 상속하는
[proposal v1](2026-10-10-stage1-v3-design.md), 아래 검증 조건의 합성 계약이다.
v2는 commit `8ad9cb0f25c7a431038f79ab4a2338bec8f54df7`, SHA256
`424c2050661c7a00c48c956219f50e9288bfd019e6c0ceeaa0d87e0a751a0f7e`로 고정했다.
v1 SHA256은 `07069c90f11297f7c5408ff280ac9a182b317b540fbd88095e53332dedb1d47a`다.
세 fresh read와 root hate는 같은 v2를 읽었고 비판 사이에 proposal을 변경하지 않았다.

## 합의된 문제와 최종 설계

기존 Stage1의 opt-in v3로 R/RV의 GT text score 차이를 Gain으로 학습하고 F→R Text-KL을
일반 loss에 더한다. query별 text 제한, 현재 latent body 차단, helper target 재사용,
shared-prefix derivative, loss/reduction/projection/optimizer 순서와 tail/resume 계약은 v1 그대로다.
CE0.1·frozen EMA1.0·한 epoch 상당·2 nodes/4 ranks/global16·Stage1 ZeRO1을 유지한다.

v2는 이전 CPU 실험에서 드러난 checkpoint 재진입과 native worktree 경로 문제를 다룬다.
제품 설계의 변경은 다음 두 연결점으로 제한한다.

- v3 전용 graph/VJP lifecycle과 순차 재진입 checkpoint manager. RNG identity는 유지하고
  호출 identity만 graph/unit/site/phase/backward role로 구분한다. 원래 alias/buffer/dropout
  binding을 recompute에 보존하며 ordinary Stage1의 기존 context 검사는 유지한다.
- 기존 coordinator에 local·remote container·remote host의 repository root를 명시한다.
  각 노드의 실행·전송·resume 경로를 해당 root로 만들고, 고정 topology와 저장 범위 검증은
  유지한다. 원격 checkout 준비 여부는 실행 전 별도 검증 대상이다.

core 호출 예산과 joint R+KL 후 별도 RV VJP 계약을 유지한다. checkpoint를 끄거나 다른
VJP 방식으로 자동 전환하지 않는다. 두 변경의 acceptance는 분리한다: checkpoint CPU 증거가
원격 launch 준비를 입증하지 않으며, import/수집 성공도 올바른 원격 경로를 입증하지 않는다.

## 비판 결과: 양립하지 않는 대안부터

### divergent-incompatible: 더 작은 문제로 경계를 바꾸기

서로 다른 두 대안이 나왔다. 하나의 다수결 결론으로 합치지 않는다.

- `/root/v3_v2_fresh_a`, `/root/v3_v2_fresh_c`: 즉시 목표가 알고리즘 준비뿐이면 import를
  안전하게 만들고 native worktree의 실제 launch는 거부하는 경계가 더 작다. 그러나 확인받은
  native-worktree 지원을 축소하므로 현재 범위를 충족하지 않는다.
- `/root/v3_v2_fresh_b`: 3-lane VJP 하나로 보조 gradient를 모으면 두 번째 재계산 자체를
  없앨 수 있다. memory/operator 지원은 미검증이고, 상속한 two-lane+RV 계약과 다르므로
  새로운 설계 선택이 필요하다.

공통 질문은 호환성 요구를 그대로 해결할지, 그 요구를 발생시킨 실행 범위를 줄일지다.
Root는 확인받은 범위를 유지하면서 가장 작은 전제 실험으로 판단할 것을 권고한다.
위 대안들은 미해결 선택지로 보존하며 실패 시 자동 fallback으로 사용하지 않는다.

### divergent-compatible: 독립된 판정 기준을 두기

세 read가 지적한 공통 경계는 **경로 설정과 고정 topology의 분리**다. 현재
`src/multinode.py:139`는 cluster 전체를 `ClusterSpec.default()`와 비교한다. root가 달라지는
것을 허용하려고 이 검사를 단순 삭제하면 unrelated topology 검증도 사라질 수 있다.
host/container·rank·GPU/process 수·master/network가 달라진 경우를 각각 거부하는 음성 검증을
유지하고, repository root의 허용 여부를 별도로 판정해야 한다. 이 사실은 root도 source에서
확인했다. v2가 요구한 안전 경계의 구체화이며 현재 구현의 안전성을 주장하는 증거는 아니다.

서로 다른 보완점도 반영한다.

- A: checkpoint feasibility와 native launch compatibility의 결과를 별도로 보고한다.
- B: unequal-root 지원은 Stage1-v3 launch와 필요한 checkpoint 수집/resume부터 확인한다.
  무관한 evaluation/helper 조합은 실행 전에 명시적으로 거부할 수 있다. 같은 root의 기존
  경로는 회귀 검증한다. 모든 helper의 범용 portability 구현은 요구하지 않는다.
- C: expected stochastic-site inventory는 model/graph 계획에서 관측 전에 만든다. 실제
  recorder에 도착한 event로 expected 목록을 만들면 누락 판정이 순환 논리가 된다.
  auxiliary graph가 합법적으로 생략되는 경우도 사전에 구분한다.
- B의 추가 관측 조건: recorder가 붙인 role 이름과 별도로 실제 autograd 호출 경계를 세어,
  두 cotangent lane과 두 score traversal을 구별한다. label만으로 호출 예산을 증명하지 않는다.

Root가 확인한 source: `trainer.py:349–379`의 사전 site inventory와 부모 training_step 위임,
`equivalence.py:702,814–844`의 기존 counter/manager, `multinode.py:75,96,139`의 root/default/guard,
`multinode_artifacts.py:45–49,1433`의 import 및 전송 경로 가정.

### convergent: v3 격리와 관측 범위의 제한

A/B/C는 고정 실행 계약을 유지한다면 v3 전용 lifecycle과 독립 CPU oracle이 적절한 다음
경계라는 방향에 가까웠다. ordinary path 보존, 실제 backward 진입 관측, 별도 실험 승인에도
가까운 의견을 냈다. 이는 reassurance이며 gradient 동등성이나 실행 가능성의 증거가 아니다.
각 read는 위 incompatible 대안과 compatible 보완점으로 분류했고, 이 항목은 부수적 일치다.

## hate 결과와 증거 상태

```json
{
  "root": "graph/VJP별 counter와 재진입 manager가 검사 오류를 없애더라도 실제 alias·shared-prefix gradient를 보존한다는 증거는 아직 없다. 잘못된 role 또는 복원된 실제 parameter를 읽는 recompute까지 정상으로 인정하면 lifecycle 검사는 통과하면서 다른 목적함수를 미분할 수 있고, v2 실행 설계의 근거가 무너진다.",
  "first_nail": "별도 승인 후 기존 독립 CPU oracle을 유지한 작은 probe에서 v2 lifecycle을 연결하고, 두 번째 RV VJP까지 전체 gradient/projection을 비교한다. FP64/FP32와 dropout0/.05에 checkpoint OFF·ON(early-stop 기본 ON/OFF)를 조합하고, 실제 parameter와 alias 값을 구별하는 한 경우 및 중복/잘못된 role/stale context/예외 복원 검사를 포함한다. 고정 수치 기준과 명시적 native·packed event 판정을 모두 만족해야 한다. 실패하면 해당 전제를 다시 검토하며 제품 코드·optimizer·모델 저장·GPU/원격 실행은 하지 않는다."
}
```

**제안 작성 당시 v2 first-nail은 미실행이었다.** 이후 별도 승인으로 실행한 결과는 문서 끝에 반영했다.
[이전 v1 실험](2026-10-10-stage1-v3-probe-outcome.md)은 checkpoint
OFF 4조건의 수치 일치와 ON 4조건의 RV VJP 실패를 관측했다. 8조건의 첫 joint gradient 일치는
새 lifecycle이나 완료된 checkpoint RV gradient/projection을 입증하지 않는다. 기존 독립
검토의 Minor2개도 과거 probe에서 해결됐다고 소급 표시하지 않는다.

전체 CPU suite의 6개 수집 오류, 알려진 partial-accumulation resume 문제, 실제 Qwen/DoRA/
BF16/ZeRO1·네 rank update 검증도 미해결이다. 새 원격 checkout의 존재·동기화·환경 호환성은
이번에 검사하지 않았다. 설계 비판은 구현 review나 runtime witness를 대체하지 않는다.

## proposal 대비 변화

수식·계수·호출 예산·node root 선택·제품 수정 범위는 변경하지 않는다. 위 비판을 반영해
v2의 기존 요구를 판별하는 조건만 구체화했다: topology 음성 검증, 관측과 독립된 site inventory,
실제 autograd 경계, 두 호환성 결과의 별도 보고, 필요한 unequal-root 작업과 미지원 거부의 경계다.
동결 proposal은 그대로 보존한다. 이 commitment를 포함한 정확한 합성 계약이 이후 승인 대상이다.

## 별도 승인받아 수행한 CPU 전제 실험의 제안 기록

질문: 제안한 v3 lifecycle이 checkpoint 재계산 중 원래 alias와 shared-prefix derivative를
보존하면서, joint R+KL 후 RV VJP와 projection까지 독립 계산과 일치시키는가?

실험 범위는 기존 tiny two-layer/two-block oracle에 scratch lifecycle을 연결하는 것이다.
FP64/FP32 × dropout0/.05 × checkpoint OFF/ON을 비교하고 ON은 기본 early-stop ON과 진단 OFF를
각각 실행한다. 순수 gradient 비교의 FP64 허용오차는 `atol=1e-9, rtol=1e-7`, FP32는
`atol=1e-6, rtol=1e-4`로 유지한다. Text-KL 내부 FP32 계산 계약도 유지한다.

사전에 같은 원본 좌표의 mask, expected native/packed event와 actual autograd boundary를
정의한다. 모든 trainable gradient, nonzero prefix derivative와 projection을 판정에 포함한다.
별도의 한 조건에서는 alias와 live module parameter 값을 구별해, 잘못된 재바인딩이 같은 값
때문에 숨지 않게 한다. oracle과 alias의 기준값은 같게 유지한다. 이는 작은 fixture의 값 설정이며
optimizer 실행이 아니다. ordinary context control과 duplicate/wrong-role/stale/exception 복원,
누락 event 거부도 포함하고 예상 실패는 구체적인 오류 종류까지 확인한다.

throwaway 코드는 별도 scratch, 실행 로그는 `logs/`, 결과는 `results/`에 둔다. 제품 코드·
coordinator/path 수정, optimizer step, 모델 저장, GPU·원격 실행은 범위에 포함하지 않는다.
실패하면 원인을 보고하고 설계로 돌아온다. 무제한 수정 반복이나 다른 목적함수로 전환하지 않는다.
산출물은 관측된 일치/불일치, 명령·결과·원인과 독립 검토 기록이다.

제안한 방식은 **root native 실행 + 별도 독립 reviewer**다. 이 후속 probe의 범위와 방법을 한 번에
승인받아 수행했다. 이전/이번 probe의 방법 승인을 제품 구현 선택으로 확대하지 않는다.

제안 당시 실험 생략에는 명시적 uncertainty waiver와 정확한 written-spec 승인 의사가 필요했다.
이번에는 실험을 수행했으며 waiver를 추정하지 않는다. 아래 결과를 반영한 이 commitment를 최종
서면 승인받은 뒤에만 `superpowers:writing-plans`로 진행한다. 전체 구현 계획 검토와 구현 방법 선택,
TDD·독립 review·fresh verification 및 해당 finish gate는 이후에도 유지한다.

## 실행 증거를 반영한 최종 commitment

제품 설계는 동결 v2와 상속 v1 및 이 문서의 검증 조건 그대로다. [v2 CPU 실험 결과](2026-10-10-stage1-v3-probe-v2-outcome.md)는
12개 dtype/dropout/checkpoint 조건과1개 alias/live-parameter 대조의 수치·호출 일치를 보였다.
잘못된 alias binding의 음성 대조는 호출 검사와 mask가 맞아도 gradient 불일치로 거부됐다.
독립 검토는 이 작은 fixture의 feasibility 주장에 Critical/Important 없이 Minor1개를 남겼다.
최종 집중 테스트26개와 matrix13조건이 통과했다. 재계산 중 예외·binding 복원까지 직접 관측한
주장은 하지 않는다. 전체 CPU suite는 기존 경로 문제로6개 수집 오류가 있어 별도로 미통과다.

이 증거는 unchanged lifecycle의 v1 실패와 구분된다. 제안한 v3-only lifecycle로 같은 수치 계약을
작은 CPU fixture에서 실행할 수 있음을 뒷받침한다. 실제 Qwen/DoRA/Trainer/ZeRO·네 rank·tail/resume,
mutable buffer/tied weight, native/remote 경로, 성능과 일반적인 실패 복원은 후속 구현 검증 대상이다.
남아 있는 구현 검증 의무를 면제하거나 실험 코드를 제품 구현으로 승인하지 않는다.

최종 승인 대상은 **동결 proposal v2가 상속한 v1 계약 + 이 commitment의 검증 조건 + 연결된 실험
증거와 명시한 한계**다. hate/macrothink 이후 수식·프로필·호출 예산·수정 범위의 material 변경은 없다.
인간의 최종 서면 설계 승인은 아직 받지 않았다.
