# V-off 동기화 변경 및 재평가 기록

이 문서는 기존 로컬 V-off 재현 실행과 동기화 수정 후 실행의 차이 및 평가 결과를 기록한다. 파일명은 사용자 요청을 따른 것이며, 이 기록만으로 실제 학습 실행 GPU가 B200이었다고 판단하지 않는다.

## 실행 및 평가 결과

| 구분 | 이전 로컬 실행 | 동기화 수정 후 실행 |
| --- | --- | --- |
| 학습 run | `voff-reproduction-full212-20260918-043839` | `voff-immediate-mask-b05-fixed-20260919-021026` |
| 코드 commit | `fe008ca6893de270625afc27838ad2ecbb9ecef6` | `1a63667bc9db3f54c5cfc350e874797e790e952a` |
| 평가 run | `eval-voff-stage2-test400-20260918-084050` | `eval-voff-immediate-test400-20260919-083208` |
| TEST 400 결과 | 200/400, 50% | 216/400, 54% |

이전 실행의 commit·4-rank 참여·beta·global batch는 [학습 시작 증거](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/full-training-start-witness-20260918.json)와 [양 노드 해석값](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/full-resolved-nodes-20260918.json)에 기록되어 있다. 수정 후 실행의 전체 유효 설정과 commit은 [수정 후 학습 기록](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/immediate-mask-fixed-launch.json)에 있다.

평가 결과 원문은 [이전 평가 요약](../.worktrees/voff-reproduction-20260917/results/eval-voff-stage2-test400-20260918-084050/summary.json), [수정 후 평가 요약](../.worktrees/voff-reproduction-20260917/results/eval-voff-immediate-test400-20260919-083208/summary.json), [수정 후 평가 검증](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/immediate-mask-evaluation-verification.json)에 있다. 동일한 400개 행을 짝지어 비교하면 이전 오답에서 수정 후 정답으로 바뀐 행이 41개, 이전 정답에서 수정 후 오답으로 바뀐 행이 25개다. 순증가는 16개, 즉 4%p다.

## 변경된 항목

1. preference beta를 `0.1`에서 `0.5`로 변경했다. preference weight는 `1.0`을 유지했다.
2. preference target scope를 `until_next_latent`에서 `immediate_text`로 변경했다. 원본 concatenated 문자열, token ID, CE label은 유지하고, 인접한 text/text 경계를 가로지르는 whole token만 preference previous/target 구간에서 제외했다. 전체 3,379개 retained sample 검증에서 입력·CE 동일성과 제외 토큰의 CE supervision 유지가 확인되었다([경계 수정 완료 증거](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/immediate-boundary-completion-evidence.json), [전체 collator 검증](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/immediate-boundary-final-collator.json)).
3. sampler를 `latent_balanced`에서 `reference_random`으로 변경했다. source의 global sample 순서와 update group을 맞추고, 마지막 불완전 update는 실제 sample을 반복하지 않는 zero-loss padding으로 채웠다. tail에서도 고정 global-batch 분모를 유지했다([sampler 구현 기록](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/sampler-implementation-report.md), [4-rank 검증](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/sampler-alignment-verification-20260918.json)).
4. sidecar가 clean embedding graph를 재사용하던 경로를 context group마다 독립적인 embedding lookup을 한 번 수행하는 경로로 바꿨다([embedding 정렬 구현 기록](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/embedding-implementation-report.md)).
5. clean CE 계산과 clean logit projection을 sidecar scoring보다 먼저 수행하도록 순서를 옮겼다. 계산식과 sidecar 인자는 유지했다([clean-first commitment](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/clean-first-commitment-v1.md), [독립 review](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/clean-first-review.md)).

## 유지된 조건

- Stage 1 초기화는 두 실행 모두 `/root/workspace/ILVR/checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431`이다.
- 두 실행 모두 4 ranks, per-device microbatch 1, gradient accumulation 4, effective global batch 16, 212 updates를 사용했다.
- learning rate `5e-5`, seed `42`, warmup 10 updates, preference weight `1.0`, entropy weight `0`을 유지했다.
- `clean_prefix` V-only preference에서 현재 step latent 본문에 대한 attention 차단을 유지했다. START/END와 원본 이미지 접근은 유지했다.
- 두 평가는 같은 TEST 400개와 `temperature=0`, `top_p=1`, `max_new_tokens=4096`, `max_sequence_length=128000`, `repetition_penalty=1`, `external_full_latent_block`, vision ablation 없음 설정을 사용했다([수정 후 평가 실행 기록](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/immediate-mask-evaluation-launch.json)).

## 남아 있는 차이와 해석 한계

이번 실행에서는 다음 차이를 변경하지 않았다.

- 현재 경로의 explicit layer K/V와 비교 대상 경로의 differentiable `DynamicCache` 사용 차이
- 현재 sidecar scorer에 추가로 적용되는 non-reentrant activation checkpoint
- 현재 경로가 scored row를 먼저 선택한 뒤 LM head를 적용하는 반면 비교 대상은 suffix hidden 전체를 projection한 뒤 scored row를 선택하는 차이

이 차이들의 추적 근거는 [mechanism difference isolation](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/mechanism-difference-isolation-20260918.md)과 [remaining differences audit](../.worktrees/voff-reproduction-20260917/results/voff-reproduction-workflow/remaining-differences-audit-20260918.md)에 있다.

이번 비교는 여러 변경을 한 번에 적용한 두 학습 실행의 결과이며 factorial ablation이나 변경별 인과 배분 실험이 아니다. 따라서 4%p 상승을 embedding-gradient 수정 또는 `immediate_text` 하나의 효과로 귀속할 수 없다. 수정 후 54%가 비교 대상의 54%와 수치상 같다는 사실도 전체 학습 메커니즘이나 gradient·optimizer trajectory의 완전한 동등성을 뜻하지 않는다.
