# Stage1-v3 CPU 전제 실험 결과

2026-10-10. 연결된 [commitment v1](2026-10-10-stage1-v3-commitment.md)의 CPU probe만 실행했다.
사용자의 `진행해`는 root native 작성·실행과 독립 reviewer 검토를 승인했다.
전체 Stage1-v3 서면 설계/제품 구현/학습 실행 승인은 아니다.

## 판정

**작은 CPU fixture에서는 checkpoint OFF의 4조건이 일치했고, ON의 4조건은 두 번째 VJP에서
실행 실패했다. 현재 stochastic/checkpoint lifecycle을 그대로 재사용하는 전제는 통과하지 못했다.**
다른 lifecycle을 설계한 bridge까지 불가능하다는 뜻은 아니다.

| 조건 | 비교 결과 |
|---|---|
| FP64/FP32 model arithmetic × dropout0/0.05, checkpoint OFF | 4/4: R/RV score, Text-KL, 전체 parameter gradient와 projection이 고정 오차 기준 내 일치 |
| 같은 4조건, checkpoint ON | 4/4: 첫 R+Text-KL joint VJP gradient 일치 후 RV VJP에서 예외; RV gradient/projection 완료되지 않음 |
| dropout/RNG | 8/8: 관측된 원본 token mask 일치, global RNG 보존 |
| 호출 기록 별도 대조 | 8/8: 해당 fixture의 관측된 순서가 사전 정의한 단계와 일치; 실패는 다음 layer 재호출 전 발생 |

모델 arithmetic이 FP64인 경우에도 Text-KL은 설계대로 FP32로 계산했다. checkpoint OFF의
최대 절대 gradient/projection 오차는 FP64 조건 약1.18e-16, FP32 조건 약1.17e-7이었다.
고정 허용오차는 FP64 atol1e-9/rtol1e-7, FP32 atol1e-6/rtol1e-4이며 실행 후 바꾸지 않았다.

기준은 두 layer의 작은 causal model을 매 objective/block마다 독립적으로 full-sequence 계산한
것이다. packed 경로는 별도 segment/mask 배치를 사용했다. 두 latent block의 suffix 길이는
2와4이며 source token 좌표를 중복·재배열했다. source prefix embedding gradient가0이 아닌 것도
확인했다. 실제 저장소 SampleStableStage1Dropout은 clean/dense 경로에서 실행하고 packed
경로는 동일 sample/step/site seed의 source-shaped mask를 재생성해 gather하는 최소 adapter다.

## 원인 증거

실제 호출 경로는 `autograd.grad(negative)` → checkpoint `unpack_hook` → `recompute_fn` →
저장된 `recompute_context.__enter__`였다. 첫 joint VJP에서 한 번 사용한 Python
`_GeneratorContextManager`에 재진입하면서 다음 예외가 발생했다.

```text
AttributeError: '_GeneratorContextManager' object has no attribute 'args'
```

실패 전 관측 순서는 clean original0/1 → packed original0/1 → packed recompute1/0 →
clean recompute1/0이다. RV traversal의 다음 layer는 실행되지 않았다.
작은 별도 실제-context witness도 첫 backward 성공과 두 번째 backward 실패를 재현했다.
fresh manager를 새로 만들어 진입하는 별도 lifecycle witness는 현행 site별 최대1회 검사가
`second recompute invocation`으로 거부함을 보였다. 후자는 대체 bridge 성공 증거가 아니다.
프로토타입은 완전한 기록을 위해 checkpoint early stopping을 껐다. 이 설정을 포함한 이 작은
실험의 관측 결과이며, Qwen runtime 전체로 일반화하지 않는다.

## 독립 review와 검증

- Reviewer `/root/v3_cpu_probe_review`, fresh context, gpt-6-astra/high. 전체 source/test hash와
  source 경로를 확인했다. 제한된 실패 결론을 막는 Critical/Important 없음; Minor2개 미해결.
- Minor1: 보조 진단 test가 예외 분류를 엄격히 고정하지 않는다. 이번 실행의 정확한 traceback과
  별도 causal witness로 현재 원인은 확인했지만, 향후 무관한 예외를 같은 진단이 받아들일 수 있다.
- Minor2: 프로토타입의 `accepted`는 수치/mask/RNG 판정이며 전체 packed 호출 순서를 자동
  승인하지 않는다. root가 기록을 별도로 대조했으며 완전한 packed lifecycle 검증 주장은 하지 않는다.
- TDD: 빈 bridge에 대한 assertion4실패 → 독립 비교4통과. partial joint-gradient 증거 보존
  assertion2실패 → 보완 후 진단8통과. 검토 후 fresh focused8통과.
- 최종 8조건 실험은 exit1, 4통과/4실행실패. 이 exit1을 성공으로 재해석하지 않는다.
- 저장소 전체 CPU suite는 exit2, **수집 오류6개**로 중단됐다. 53개 GPU/distributed marker
  deselection 후 본 테스트 실행에 도달하지 못했다. 전체 suite 통과 주장은 없다.

수집 실패 파일은 모두 `tests/multinode/`의 다음 파일이다:

1. `test_artifacts.py`
2. `test_capacity.py`
3. `test_evaluation_coordinator.py`
4. `test_lifecycle_matrix.py`
5. `test_planned_checkpoint_stop.py`
6. `test_vision_gain_probe.py`

모두 기존 `src/multinode_artifacts.py:45–49` import 시 발생한다. 이 코드는 실제 repository
경로가 `/root/workspace/ILVR` 하위라고 가정해 `relative_to`를 호출하지만 native worktree는
`/root/.codex/worktrees/541e/ILVR`이다. alias symlink도 `Path.resolve()` 이후에는 이 가정을
만족하지 않는다. 제품 코드 수정은 이번 probe 권한에 포함되지 않아 원인만 기록했다.

## 근거 파일과 변경 identity

- [최종 matrix JSON](../../../results/stage1-v3-preparation/cpu-probe/final-matrix.json)
- [호출 기록 대조](../../../results/stage1-v3-preparation/cpu-probe/event-audit.json)
- [독립 검토 기록](../../../logs/stage1-v3-preparation/cpu-probe/independent-review.md)
- [CPU 전체 suite 로그](../../../logs/stage1-v3-preparation/cpu-probe/project-cpu.log)
- [최종 focused 로그](../../../logs/stage1-v3-preparation/cpu-probe/final-focused.log)
- [실행 brief](../../../logs/stage1-v3-preparation/cpu-probe-brief.md)
- [workflow](../../../logs/stage1-v3-preparation/workflow.md)

검토·실행 코드 기반 HEAD는 `0d79a8b738c5415bd173be3b18489f0eceeec29e`이며 제품 파일은
수정하지 않았다. throwaway 코드와 ledger는 `.superpowers/sdd/cpu-probe-brief/`에 보존했다.
`probe.py` SHA256 `399fff287508bd29b2f020b49926f6035db776fc668bf83da5b7cbae52364da9`;
`test_probe.py` SHA256 `608732f520fc1eee2202910a1a84569812d6fe4710047dc1967aaf6a7a27a13f`.
실험은 optimizer step/모델 저장/GPU/분산 학습을 실행하지 않았다. 전체 suite의 임시 경로는
`checkpoints/workflow-validation/stage1-v3-preparation/final-cpu-01/` 아래로 지정했다.

## 판단과 잔여 한계

실행 brief·ledger의 판단을 유지한다. 승인된 작은 전제 실험에 native 경로를 적용했으며
전체 구현 계획으로 확대하지 않았다(그 구분이 잘못되면 절차/범위가 어긋날 수 있다).
의도적으로 non-Git인 prototype은 commit diff 대신 실제 content hash로 review했다(잘못되면
검토 identity가 모호해지므로 reviewer와 root가 해시를 확인했다). 실패한 lifecycle을 이 실험에서
수정하지 않았다(실현 가능한 대안의 검증은 늦어지지만 승인받지 않은 설계 변경을 피한다).

독립 reviewer가 판단을 유보한 GPU/BF16/Qwen/DoRA/DeepSpeed, optimizer gradient 전달·mutation,
tail/resume, 성능/전체 호출 budget, multi-sample/padding/일반 mask/비채점 suffix 행은 미검증으로
남긴다. checkpoint ON의 RV gradient/projection도 미관측이다. 이를 넓게 주장하면 없는 증거를
호환성·정확성으로 오인시키게 된다. FP32 KL 및 packed mask adapter라는 구현 범위도 명시한다.
진단8통과나 reviewer의 제한된 긍정 판정이 이 한계를 해소하지 않는다.

## 다음 설계에서 확인할 변경안

현재 proposal/commitment v1은 최종 승인 단계로 진행하지 않는다. 수학적 목적함수·dropout seed·
호출 예산을 유지하려면, v3에 한정해 **원래 forward와 각 VJP traversal의 재계산을 구분하는
stochastic context 계약**이 필요하다. original은 graph/site별 한 번, recompute는 승인된
VJP/site별 한 번으로 검증하고, context 재진입도 해당 범위에서 안전하게 지원하는 방향을 제안한다.
기존 Stage1의 검사를 전역으로 느슨하게 만들지 않는 것이 변경 경계다. 이 방향의 실현 가능성은
아직 입증되지 않았다. native worktree를 지원하는 multinode 경로 처리도 별도 작은 수정 대상으로
설계에 포함해야 한다.

이는 **수정 설계안**이며 실행 승인이 아니다. 변경된 C절/경로 범위를 사용자와 확인하고 새
proposal/commitment를 작성·검토·승인받은 뒤 구현 계획으로 진행한다. 기존 A/B 확인 및 학습
프로필 선택은 유지한다. 전체 suite가 green이 아니며 구현도 시작하지 않았으므로 branch finish,
merge/PR 또는 worktree 정리는 하지 않았다.
