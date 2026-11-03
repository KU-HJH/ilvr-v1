# V-off 경계 토큰 및 서버 간 비교

## 추가 확인: loss 수식은 같지만, preference의 정답 범위가 다르다

**앞서 'OFF 차단 규칙과 loss가 같다'고만 요약한 것은 불충분했다.** 같은 span을 넣은 planner 비교에서는 일치했지만, 원본 데이터에서 그 span을 만드는 규칙이 다르다. GPU 수나 초기 checkpoint를 제외해도 아래 방법론 차이가 실제로 있다.

| 항목 | 현재 서버 | 타 서버 |
|---|---|---|
| preference target 범위 | latent 바로 뒤의 `text` 항목 **하나** | 다음 latent 전까지의 **모든 텍스트** |
| 풀이 뒤 별도 항목인 최종 정답 문장 | **preference에서 제외** | **preference에 포함** |
| 일반 CE의 최종 정답 문장 | 학습함 | 학습함 |
| 연속 text 항목 연결 | 사이에 공백+줄바꿈 ` \n` 삽입 | 구분자 없이 그대로 연결 |
| loss 수식과 집계 | 토큰 평균 log-prob → step별 preference → step 평균 | 같은 수식·집계. 단, 평균에 들어가는 target가 다름 |
| OFF에서 START/latent/END 처리 | START/END 유지, 현재 latent 본체만 attention 차단 | 같은 규칙 |
| ON에서 END 및 위치 | END 입력 0, 원래 위치 유지 | 블록 앞 텍스트에서 다시 예측, 뒤쪽 위치 당김 |

실제 TRAIN index0을 같은 processor와 원본 이미지로 양쪽 **실제 데이터 준비·collator 코드**에 넣어 확인했다. 현재 preference target은 풀이 **67토큰**, 타 서버는 `The final answer is: 70°`까지 **76토큰**이었다. 입력 길이는 현재213/타 서버212, CE 감독 토큰은94/93이었다. 별도 final-answer text가 뒤따르는 구조는 전체3450개 중3449개, **현재 retained3379개 중3378개**에서 확인했다. 이는 source 구조의 전수 집계이며 타 서버의 역사적 retained ID 목록까지 확인했다는 뜻은 아니다.

같은 target와 FP32 logits를 주면 두 loss 함수의 값과 gradient는 비교 실행에서 정확히 일치했다. 즉 **수식이 다른 것이 아니라 수식에 들어가는 정답 텍스트가 다른 것**이다. 타 서버 preference는 최종 정답 문장까지 비교하고, 현재는 바로 뒤 풀이 항목까지만 비교한다. 현재 모델이 최종 정답을 전혀 학습하지 않는다는 뜻은 아니다. 양쪽 모두 정상 CE에는 최종 정답이 포함된다.

이 차이는 loss가 모델에 주는 학습 신호를 바꾸는 확인된 방법론 차이다. 다만 이것이 정확도 역전의 단일 원인이라는 인과 검증은 아직 없다. 상세 근거: [방법론·loss·토큰 처리 검증](../results/methodology-loss-token-audit-20260913/verification.md).

## 최신 확인 — `/raid`의 ours1 자료 배치 이후 (2026-09-13)

실제 타 서버 자료를 `/raid/workspace/ILVR/ILVR/.worktrees/ours1`에서 확인했다. 코드는 그 아래 `source/`, 실제 실행 설정은 `training/`, 평가 원출력은 `evaluation/`에 있다. **이전 절의 '타 서버 코드·216/400 원출력 미확인' 상태는 해소됐다.** 아래의 과거 조사 기록은 새 자료 배치 전 상태다.

### START/END를 유지하는 OFF baseline 비교

| 항목 | 현재 서버 | 타 서버 |
|---|---|---|
| V-off에서 차단하는 것 | 현재 step의 latent 본체 | 동일 |
| START/END | 유지. END 자리부터 같은 정답 텍스트를 예측 | 동일 |
| 원본 이미지·이전 step latent | 유지 | 동일 |
| 위치 번호 / 입력 항목 삭제 | 원래 위치 유지, 삭제 없음 | 동일한 원리. 두 step CPU 비교에서 attention·입력·위치 일치 |
| 학습 목표의 수식 | 각 서버가 선택한 GT의 clean/V-off 비교 + clean CE | 수식·계수 동일. **서버 간 GT target 범위는 위 추가 확인처럼 다름** |
| 초기 Stage1 run | `stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225`, checkpoint-215 | `stage1-dora-frozen-b1-lr5e5-zero1-1epoch-exact32768-20260902`, checkpoint-431 |
| Stage2 GPU / 누적 | 4 GPU × microbatch1 × accumulation4 | 1 GPU × microbatch1 × accumulation16 |
| 한 번의 업데이트에 모이는 수 | 16 | 16 |
| sampler 경로 | multi-rank용 step-count grouping | single-rank의 상위 Trainer sampler |
| Stage2 학습 설정 | lr5e-5, seed42, 212 step, DoRA r32/alpha64/dropout0.05 | 동일한 값 |
| activation checkpointing | global gradient checkpointing ON | global OFF, latent rollout만 선택적으로 ON |
| tokenizer / chat template / processor 설정 | 비교 기준 | JSON 내용 동일 |
| 평가 반복 억제 | **repetition_penalty=1.05 적용** | **적용 코드 없음** |
| 정답 수 / 정확도 | 175/400, 43.75% | **216/400, 54.00% — 원출력 재채점 확인** |

**직관적인 결론:** OFF에서 '무엇을 못 보게 하는가'는 같은 규칙이다. 그러나 같은 Stage1 run에서 출발한 것도 아니고, Stage2의 batch 구성과 sampler 경로, 평가의 반복 억제 처리도 다르다. 따라서 10.25%p 차이를 START/END 유지 방식 차이로 설명하면 안 된다. 각 차이가 성능에 얼마나 기여했는지는 동일 초기 checkpoint와 평가 조건을 고정한 대조 실행 없이는 분리할 수 없다.

### ON의 차이도 이제 코드로 확인됨

현재 서버 ON은 END 입력을 0으로 바꾸되 END 자리와 위치 번호를 남긴다. 타 서버 ON은 **현재 visual block 앞의 마지막 텍스트 토큰을 다시 입력해 다음 문장 첫 토큰을 예측**하고, 뒤쪽 텍스트의 위치 번호를 앞으로 당긴다. 타 서버도 공유 clean KV 텐서 자체를 통째로 삭제하는 것은 아니다. V-off 분기의 참조 범위·입력·위치를 바꿔 블록을 건너뛴 유효 문맥을 구성한다.

타 서버 ON 원출력 역시 원본 TEST 정답으로 재채점하여 **201/400**을 확인했다. OFF/ON 모두 저장 match와 재채점 불일치 0개, error 0개였다. 두 서버 ON은 다른 조작이지만, 이것만으로 성능 역전의 원인이 입증된 것은 아니다.

근거와 비교 실행 범위: [새 자료 검증 상세](../results/baseline-v-off-new-ours1-20260913/verification.md).

---

## 사용자 제공 상대 서버 결과로 비교 대상 확정

사용자가 상대 SSH 서버에서 가져온 `/raid/workspace/ILVR/ILVR`의 최근 작업 결과로 다음 표를 제공했다. 상대 수치는 **사용자 제공 결과**이며, 해당 두 run의 원출력과 실행 코드를 독립적으로 연결·재채점한 상태는 아니다. 이전 절들의 ‘상대 두 결과 미식별’은 수치를 받기 전 조사 상태다. 이제 비교할 수치는 확정됐고, 남은 미확인점은 그 결과와 실제 구현·실행 설정의 연결이다.

| 자료 | OFF | ON | ON − OFF |
|---|---|---|---:|
| 상대 서버: 사용자 제공 | `attention_mask`: latent만 차단, 경계 유지. **216/400, 54.00%** | `remove_full_visual_block`: 표 설명상 latent와 START/END 모두 제거. **201/400, 50.25%** | **−15개, −3.75%p** |
| 현재 ours1: 기존 원출력 확인 | V-off latent 구간 차단, 경계 유지. **175/400, 43.75%** | V-off 경계 접근까지 차단하고 END 재생 입력을 0으로 처리. **199/400, 49.75%** | **+24개, +6.00%p** |

**우선 구분해야 할 구현:** 현재 ours1은 토큰 시퀀스에서 블록을 삭제하지 않는다. 기존 위치를 사용하는 재생 분기를 만들고 첫 END embedding을 0으로 바꾼다. 상대 표의 ‘제거’가 실제 시퀀스 삭제를 의미한다면, 두 ON은 서로 다른 조작이다. 다만 `remove_full_visual_block`이라는 이름과 설명만으로 상대 구현의 위치 번호 재계산, 첫 target의 예측 위치, latent 재계산 여부까지 단정하지 않는다. 이 식별자로 사본과 그 주요 Git 브랜치를 검색했으나 해당 구현을 아직 찾지 못했다.

또 하나의 중요한 관측은 **OFF 기준 모델부터 216 대 175, 즉 41개(10.25%p) 차이**가 난다는 점이다. 따라서 ON 조작 차이 하나만으로 두 서버 결과를 설명할 수 없다. 초기 checkpoint, 학습 이력·설정·난수 조건, 평가 모델·prompt·채점 조건을 대응시켜 확인해야 한다. 이것은 해당 조건 중 특정 하나가 달랐다는 증거가 아니라, 아직 같은 기준 모델을 비교한다고 볼 근거가 부족하다는 뜻이다.

현재 결론은 **상반된 수치는 확인됐지만 동일한 실험의 재현 실패로 확정되지 않았고, 정확도 역전의 원인도 아직 입증되지 않았다**는 것이다. 가장 먼저 연결할 항목은 상대 `remove_full_visual_block` 코드와 두 결과를 만든 run/checkpoint이며, 현재 ours1의 경계 zeroing 효과를 그 원인으로 대신 설명하지 않는다.

2026-09-13 갱신. 현재 서버 자료와 `/raid/workspace/ILVR/ILVR` 사본의 코드·로그·평가 원출력을 대조했다. 제공된 사본은 읽기만 했으며, 학습이나 GPU 평가를 새로 실행하지 않았다.

자료 출처는 사용자 확인에 따라 `/raid/workspace/ILVR/ILVR` = **상대 SSH 서버에서 가져온 사본**으로 확정한다. 이 사본이 상대 서버 자료라는 사실을 의심하거나 SSH 접속 정보를 다시 요구할 이유는 없다. 현재 미해결점은 접속이 아니라, 그 사본 안에서 사용자가 말한 상반된 결과의 두 run과 원출력을 아직 식별·연결하지 못한 것이다. 상대 사본의 완료 400문항 결과는 아래 2절의 5개 파일에서 정답 수 173·158·195·199·228을 다시 확인했다. 이를 현재 서버의 175·199 경계 옵션 비교와 혼용하지 않는다.

## 재검토 결과: START/END 관련 설정을 분리해야 한다

**이 문서의 과거 표현 ‘경계 포함/제외 학습’은 모델 전체에서 START/END를 유지/제거했다는 뜻이 아니다.** 아래 두 현재 run은 정상 학습·추론에 START/END를 모두 사용한다. ‘포함/제외’라는 축약은 오직 V-off 비교 분기의 추가 경계 차단 여부를 가리킨다. 이를 다른 서버와의 학습 차이라고 대신 설명한 것은 부적절했으며, 서버 간 성능 역전의 원인으로 확정하지 않는다.

| 실제 비교 항목 | 현재 175/400 모델 | 현재 199/400 모델 | 근거 |
|---|---|---|---|
| V-off만 사용하는 preference | 켬 | 켬 | 두 launch의 `STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true` |
| V-off 추가 경계 차단 | 미지정: 기본 False | True | `STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES` launch·기본값 |
| 경계 토큰 CE supervision | True | True | 두 저장 config의 `supervise_latent_boundaries` |
| 정상 분기의 START/END | 사용 | 사용 | 입력·trainer 호출 경로 |

서로 다른 두 설정의 정확한 의미는 다음과 같다.

- `supervise_latent_boundaries=False`: START/END의 **label만 −100**으로 바꿔 직접 CE supervision에서 제외한다. 입력 토큰을 삭제하거나 attention을 차단하지 않는다.
- `visual_off_suppress_latent_boundaries=True`: **V-off 분기에서만** 차단 범위를 START/PAD/END까지 넓히고, 첫 재생 입력인 END embedding을 0으로 만든다. 이 설정은 위 supervision 설정과 별개다. 단, 차단 범위 확장과 END 입력 0 처리는 현재 이 **하나의 flag에 함께 연결**되어 있으며 각각 독립된 두 knob는 아니다.

현재 ours1의 기존 CPU 테스트 7개를 다시 실행해 모두 통과했다(`7 passed in 9.56s`). 여기에는 supervision True/False의 label 검사, V-only 분기 선택, 경계 차단 계획, 경계 embedding 변화에 대한 off-logit 불변성, off 분기 gradient 차단, **실제 trainer→model 경로로 flag 전달** 검사가 포함된다. 경계 embedding을 바꾸면 정상 clean 출력은 바뀌고 차단된 off 출력은 유지되는 것도 확인했다. 이는 작은 CPU 모델에서의 실행 검증이며, 과거 전체 GPU 학습 재현이나 성능 인과 검증은 아니다.

코드 근거: [별도 설정 필드](/root/workspace/ILVR/.worktrees/ours1/src/step_preference/config.py:74), [supervision의 label 처리](/root/workspace/ILVR/.worktrees/ours1/src/stage2/data.py:213), [V-off 옵션의 trainer 전달](/root/workspace/ILVR/.worktrees/ours1/src/stage2/trainer.py:613), [차단 범위·END zero flag 연결](/root/workspace/ILVR/.worktrees/ours1/src/step_preference/sidecar.py:50). 실행 기록: [CPU 재검토](../logs/training-mechanism-audit-20260913/boundary-mode-check.txt).

## 성능 차이에 대한 핵심 분석

이 절은 실제 결과가 확보된 **현재 ours1의 경계 포함·제외 두 학습**을 분석한다. 아직 식별하지 못한 다른 서버의 실험을 대신 설명하는 것은 아니다. 학습률·데이터 차이가 아니라, **V-off에서 경계까지 차단하는 조작이 preference 학습 신호를 어떻게 바꿨는지**를 추적했다.

### 학습 중 무엇이 달라졌나

| 학습 진행률 | 경계 포함 preference loss | 경계 제외 preference loss | 경계 제외 V-off 정답 로그확률 |
|---|---:|---:|---:|
| 약 9% | 0.69312 | 0.66267 | −0.94 |
| 약 19% | 0.69312 | 0.38246 | −10.49 |
| 약 28% | 0.69316 | 0.02155 | −39.89 |
| 약 57% | 0.69315 | 0.00082 | −73.18 |
| 마지막 기록 | 0.69313 | 0.000126 | −90.70 |

경계 포함에서는 정상 풀이와 V-off의 점수가 거의 갈라지지 않았다. 경계 제외에서는 V-off 점수가 급격히 낮아져, ‘정상 풀이가 V-off보다 좋아야 한다’는 조건을 일찍 만족했다. 마지막 clean 로그확률은 −0.2470 대 −0.2495이고 CE는 0.2124 대 0.2150으로 비슷했다. **Preference loss 감소의 관측상 큰 변화는 clean 개선이 아니라 V-off 점수 하락이다.** 값들은 writer-rank의 로그 구간 평균이며, 고정 문항의 학습 전후 측정이나 전체 rank 평균은 아니다.

이 loss는 차이가 충분히 벌어지면 더 벌리라는 압력이 작아진다. 따라서 경계 제외 버전은 초반에 비교 분기를 크게 불리하게 만든 뒤, 후반에는 preference의 점수 수준 압력이 크게 약해지는 양상이다. ‘loss가 0.693에서 일정하니 gradient도 0’이라는 뜻은 아니며, 전체 모델 gradient를 측정하지 않고 후반 학습이 정확히 CE-only와 같다고 말할 수도 없다.

수식으로 확인하면, 한 구간에서 차이를 $\Delta=s_{\mathrm{clean}}-s_{\mathrm{Voff}}$라 할 때 다음 관계가 있다.

$$
\ell=\log(1+e^{-\beta\Delta}),\qquad
\left|\frac{\partial\ell}{\partial\Delta}\right|
=\beta\sigma(-\beta\Delta)\leq\beta\ell.
$$

따라서 마지막 경계 제외 로그의 평균 loss 0.00012596과 $\beta=0.1$로부터, 해당 로그 구간의 평균 점수 차이 계수는 최대 약 $1.26\times10^{-5}$다. 비교를 위해 차이 0인 단일 구간의 계수는 0.05다. **이는 점수에 대한 계수의 상한이지, 모델 파라미터 gradient norm의 상한은 아니다.**

경계 제외 변경에는 START·latent 접근 차단 확장과 END 재생 입력 0 처리가 함께 들어 있다. 현재 두 run만으로 그 둘 중 어느 조작이 효과를 주도했는지는 분리되지 않는다. 비교 분기를 손상시키는 지름길을 학습했을 가능성은 로그와 부합하지만, 독립적인 원인 분리 실험으로 확정된 것은 아니다.

### 실제 정답률 상승은 어디서 나왔나

| 문제 유형 | 경계 포함 정답 | 경계 제외 정답 | 오답→정답 | 정답→오답 | 순변화 |
|---|---:|---:|---:|---:|---:|
| creation | 63/100 | 68/100 | 14 | 9 | +5 |
| deletion | 36/100 | 55/100 | 24 | 5 | **+19** |
| selection | 36/100 | 35/100 | 0 | 1 | −1 |
| update | 40/100 | 41/100 | 7 | 6 | +1 |
| 전체 | 175/400 | 199/400 | 45 | 21 | +24 |

순증가 24개 중 19개가 deletion에서 발생했다. 모든 시각 추론 유형이 고르게 좋아진 결과는 아니다. 정상 평가에서 경계 포함은 400/400, 경계 제외는 399/400이 EOS로 종료됐으며, 경계 제외에서 길이 제한에 걸린 1문항은 양쪽 모두 오답이다. **이번 +24개는 길이 제한 오답을 해소해서 생긴 상승이 아니다.**

deletion의 실제 개선 사례도 확인했다. 아래는 원출력의 진술과 결과이며, 중간에 언급된 얼굴 수를 이미지에서 독립적으로 다시 세어 검증한 것은 아니다.

| TEST index | 경계 포함 출력 | 경계 제외 출력 | 확인 가능한 변화 |
|---|---|---|---|
| 103 | 6단계에서 10·10·10·10·10·1을 말하고 최종 41, 오답 | 같은 개수 진술 후 최종 51, 정답 | 중간 개수는 같은데 마지막 집계가 바뀜 |
| 107 | latent 9블록, 최종 70, 오답 | latent 6블록, 최종 58, 정답 | 반복 단계 수와 최종 답이 함께 바뀜 |
| 108 | latent 5블록, 최종 49, 오답 | latent 4블록, 최종 39, 정답 | 반복 단계 수와 최종 답이 함께 바뀜 |

**현재 증거로 말할 수 있는 결론:** 경계 차단을 강화하면서 preference의 학습 신호가 크게 달라졌고, 최종 점수 상승은 주로 deletion의 집계·반복 단계 변화에서 나타났다. ‘경계 토큰을 빼면 시각 이해가 전반적으로 좋아진다’는 결론은 지지하지 않는다. 이 학습 신호 변화가 +6%p 상승을 얼마나 설명하는지, 다른 seed나 다른 서버에서도 같은지까지는 두 run만으로 확정할 수 없다.

계산 근거: [loss 곡선·유형별 성능·사례 집계](../results/training-mechanism-audit-20260913/performance-audit.json), [400문항 대응표](../results/server-comparison-20260912/paired_full400.csv), [경계 포함 원출력](../results/server-comparison-20260912/evaluation_full400/boundary_included/merged.jsonl), [경계 제외 원출력](../results/server-comparison-20260912/evaluation_full400/boundary_excluded/merged.jsonl).

## 최종 비교 대상 정정: `ours1` worktree 기준

사용자가 비교 대상을 `ours1` worktree로 명확히 지정했다. **아래의 `ours-trial1` 비교는 다른 구현에 대한 조사이므로, 요청한 서버 간 성능 역전의 근거에서 제외한다.** 그 구현의 ramp·dropout·비교 범위 차이를 `ours1` 두 학습의 차이라고 설명해서는 안 된다.

| 확인 위치 | 결과 |
|---|---|
| `/raid/workspace/ILVR/ILVR/.worktrees/` | `vcsd-ilvr-stage2`만 존재; `ours1` 없음 |
| 해당 사본의 Git worktree 등록·local/remote refs | `ours1` 없음 |
| `/root/workspace/ILVR/.worktrees/ours1` | 존재; HEAD `b580a393a5860ca74f9bd3f2944036cf59cb2e49` |

추가로 `/raid` 아래 디렉터리를 깊이 8까지 검색했고, 원본 `/raid/workspace/ILVR/ILVR-backup-20260828.tar.zst`(223,468,323,722 bytes)의 tar 목록도 끝까지 검사했다. `pipefail`을 적용한 목록 검사 명령은 exit 0으로 끝났다. 압축본에서 확인된 최상위 worktree 디렉터리 항목은 `ILVR/.worktrees/vcsd-ilvr-stage2/`뿐이며 `ours1` 항목은 없었다. 모델 파일을 추출하거나 수정하지 않았다.

`/raid/workspace/ILVR/checkpoints/ours1-...` 디렉터리들은 실제로 존재한다. 그러나 현재 `/root/workspace/ILVR/.worktrees/ours1/checkpoints`가 바로 그 저장소로 연결되어 있다. 경계 제외 run 디렉터리를 양쪽 경로로 `stat`한 device·inode는 모두 `2049:608968611`로 같았다. 따라서 이 두 경로를 독립 서버의 서로 다른 실험으로 간주할 수 없다. **지정 경로를 몰라서가 아니라, 현재 접근 가능한 사본에서 상대 `ours1` worktree가 식별되지 않은 상태다.**

현재 존재하는 `ours1`에서 실제 두 학습의 `launch.sh`를 직접 비교했다. export 설정 차이는 `STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=true` 추가뿐이다. 그 밖에는 run ID·분산 통신 port·공백 차이였다. **두 launch는 초기 Stage1 checkpoint, 데이터, DoRA dropout 0.05, 학습률 5e-5, seed 42, preference 가중치 1·beta 0.1, attention 설정, 212 step이 같다.** 이것은 실행 스크립트의 비교이며, 이 사실만으로 모든 런타임 상태·커널·난수열까지 동일하다고 단정하지 않는다.

근거: [경계 포함 launch](/root/workspace/ILVR/.worktrees/ours1/logs/ours1-ctx-pref-vonly-direct-s1-dora-egb16-lr5e5-default-s212-r2-20260910t022900z/launch.sh), [경계 제외 launch](/root/workspace/ILVR/.worktrees/ours1/logs/ours1-ctx-pref-vonly-boundaryoff-direct-s1-dora-egb16-lr5e5-s212-20260911t085315z/launch.sh).

이 두 run은 이미 확인한 현재 서버의 175/400·199/400 모델이다. 다른 서버의 `ours1` worktree는 제공된 사본에서 확인되지 않아, **다른 서버와 학습이 무엇이 다른지는 아직 판정할 수 없다.** 상대 `ours1`의 실제 경로 또는 그것이 포함된 사본이 필요하다.

## 참고로 보존: 비교 대상이 아닌 `ours-trial1` 조사

이번 요청은 **서버 사이에서 학습이 무엇이 달라 성능 방향이 바뀌었는지** 확인하는 것으로 해석했다. 아래 E2 추론 실험은 그 원인을 입증하는 자료로 사용하지 않는다.

앞선 조사에서는 디스크에 있는 `vcsd-ilvr-stage2` 작업 폴더를 중심으로 확인했다. 재조사에서 작업 폴더가 사라진 **`ours-trial1` 브랜치의 Git 코드**를 찾았다. 커밋은 `f33c1390ab4533bc33513e895c685eb54f8090a0`이다. 여기에 evidence-window preference 구현이 실제로 남아 있다. 따라서 사본 전체를 ‘CE-only 학습’으로 설명하는 것은 부정확하다. 아래 3절의 CE-only 설명은 `vcsd-ilvr-stage2`에 한정된다.

### 새로 확인한 학습 코드 차이

| 항목 | 사본 Git의 `ours-trial1` reference-free EW | 현재 ours1 경계 제외 V-off | 확인 수준 |
|---|---|---|---|
| 경계 차단 | START·PAD·END를 attention key/value 차단 범위에 포함 | 경계까지 차단하고 **첫 END 재생 입력도 0으로 변경** | 코드와 CPU 구성요소 검사 |
| 비교하는 조합 | 기본 설정은 각 풀이 구간에 대해 앞선 모든 latent 블록을 각각 비교 | 각 step의 대응 latent 블록과 직후 풀이를 비교 | parser·선택 코드; 사본에서 2블록 예제로 3개 조합 확인 |
| preference 시작 시점 | 제공 config는 처음 10%에서 0, 다음 20% 동안 증가, 이후 1 | 현재 context objective는 설정값 1을 그대로 적용; 별도 preference ramp 없음 | 사본 기본 config와 현재 objective 코드; 상대 실제 run 설정은 미확보 |
| dropout | 활성 EW 비교 전에 양수 dropout을 검사하여 거부 | DoRA dropout 0.05 사용 | 사본 실행 전 검사 코드·현재 adapter 설정 |
| 점수 계산 경로 | 정상 CE 외에 별도 full/차단 SDPA forward | 정상 prefix를 공유하는 sidecar, 현재 실행은 flex | trainer 호출 경로 |
| 기본 loss 형태 | CE + reference-free sigmoid preference, beta 0.1 | CE + reference-free sigmoid preference, beta 0.1 | **이 기본 수식 자체는 같은 계열** |

핵심은 **‘경계를 attention에서 가리는 것’과 ‘END 입력을 0으로 바꾸는 것’이 다르다**는 점이다. 다음 풀이의 첫 단어는 바로 앞 END 위치의 출력으로 예측한다. 사본 구현은 attention을 차단해도 END의 입력값이 residual 경로에 남는다. 현재 경계 제외 구현은 그 입력값부터 0으로 바꾼다. 즉 두 구현은 같은 ‘경계 OFF’ 대조군이 아니다.

이 차이는 CPU에서 실제 함수로 확인했다. 현재 재생 입력 함수에 END 벡터 `[12, 13, 14]`를 넣으면 경계 포함은 그대로, 경계 제외는 `[0, 0, 0]`이 된다. 사본 decoder의 실제 forward에 attention·MLP 출력만 0으로 만드는 시험용 구성요소를 연결하면 END 입력이 residual을 통해 그대로 남는다. 이는 **코드 경로 차이의 증거**이며, 실제 모델에서 정확도 역전의 원인을 확정하는 실험은 아니다.

현재 학습에서 경계 제외 V-off 점수가 −90.70까지 낮아진 점을 고려하면, 우선 분리해서 검증할 후보는 **END 입력 0 처리로 비교 분기가 훨씬 강하게 손상되는 효과**다. preference는 정상 점수를 높이는 것뿐 아니라 차단 점수를 낮춰도 줄어들므로, 이 조작은 학습 신호의 성격을 바꿀 수 있다. 다만 상대 run의 loss 추이와 최종 ON/OFF 결과가 연결되지 않아 ‘이것 때문에 반대로 나왔다’고 확정할 수는 없다.

Git에 남은 검증 기록은 7B 모델·`comt_ckpt` 초기값·sample 920을 이용한 구현 smoke 등을 기록한다. 현재 3B DoRA 전체 학습과 짝지을 최종 성능 실험은 아니다. **찾아낸 코드가 반대 성능을 낸 바로 그 run에 쓰였는지부터 아직 확인이 필요하다.** 두 run의 경로 또는 수치를 받으면 그 checkpoint→실행 config→loss 로그→평가 결과를 연결해야 한다. 재학습은 시작하지 않았다.

근거와 Git 코드 발췌: [학습 메커니즘 재조사 증거](../results/training-mechanism-audit-20260913/source-evidence.md). 기존 [인계 문서](/raid/workspace/ILVR/ILVR/documents/ours-trial1/guidance1.md)는 브랜치를 찾는 단서로 사용했으며, 위 차이는 Git 코드도 직접 확인했다.

## 기존 평가 수치 요약

| 비교 | 현재 확인한 결과 |
|---|---|
| 학습 V-off 분기에서 경계 토큰 포함 → 제외 | 정상 평가 175/400 → 199/400 (+6.00%p) |
| 평가에서 latent 계산 허용 → 금지, 경계 포함 Preference 모델 | 고정 패널 40/99 → 40/99; 서로 다른 9개 손실·9개 이득 |
| 평가에서 latent 계산 허용 → 금지, 경계 제외 Preference 모델 | 완료: 고정 패널 44/99 → 7/99; 정답 39개 손실·2개 이득 |
| 독립적인 다른 서버와 성능 방향 비교 | 제공 사본에서 Stage1 결과는 확인했으나, 동일한 V-off 경계 ON/OFF 실험 쌍은 찾지 못함 |

경계 제외 학습에서도 clean branch와 정상 평가는 latent를 사용한다. 이를 평가 시 latent 계산 제거와 혼동하면 서로 반대인 결과처럼 읽을 수 있다. 실제 서버 간 역전인지 확인하려면 양쪽의 latent ON/OFF 구현, 같은 sample index별 원출력, checkpoint와 생성 설정을 맞춰야 한다.

9월 12일 시점의 수집 자료는 [비교 자료 안내](../results/server-comparison-20260912/README.md)에 있다. 그 자료의 E2 ‘진행 중’ 표기는 당시 상태이며, 위 표와 아래 결과가 최신이다. 전체 400문항 수치는 [문항별 CSV](../results/server-comparison-20260912/paired_full400.csv), [공통 재채점 결과](../results/server-comparison-20260912/comparison.json), [학습 metric](../results/server-comparison-20260912/training_metrics.json)에서 확인할 수 있다.

원격 rank-0에서 가져온 로그는 **현재 run에 참여한 같은 분산 학습의 로그**다. 반대 결과가 나온 독립 실험의 자료로 간주하지 않는다.

## 1. 같은 ‘latent 제외’라는 말이 서로 다른 실험을 가리킨다

Interleaved vision-language reasoning에서는 글로 풀이하다가 START로 latent 계산을 시작하고, 여러 latent 상태를 만든 뒤 END를 지나 다시 글로 풀이한다. 다음 세 조작은 이 흐름의 서로 다른 부분을 바꾼다.

| 조작 | 실제로 바꾸는 것 | 그대로 남는 것 |
|---|---|---|
| 제공 사본의 `supervise_latent_boundaries=False` | START/END를 정답으로 예측하는 직접 CE 항만 제외 | 입력의 START/END, latent 계산, attention 경로 |
| 현재 ours1의 V-off 경계 제외 학습 | 비교용 V-off 분기에서 해당 START·latent 구간의 attention 접근을 막고, 첫 재생 입력인 END embedding을 0으로 만듦 | 정상 학습 분기와 정상 추론의 START/latent/END; 시퀀스 길이·위치 |
| 현재 E2 `no_latent_execution` | 추론 중 START 선택을 금지하여 latent 블록 자체를 생성하지 못하게 함 | 원본 이미지 입력과 텍스트 추론 |

따라서 **경계 제외 V-off로 학습해서 점수가 올랐다는 것은, 추론에서도 START/END와 latent 계산을 없애면 좋아진다는 뜻이 아니다.** E2는 경계 토큰만의 역할과 latent 내용의 역할도 분리하지 못한다.

제공 사본에서 `supervise_latent_boundaries=False` 옵션의 동작은 확인했지만, 이 옵션을 False로 사용한 성능 대조 실험은 찾지 못했다. 확인한 Stage2 실행 로그는 True였다. 현재 두 V-off 모델의 정상 학습 설정도 `supervise_latent_boundaries=True`다.

코드 근거: [사본의 label 생성](/raid/workspace/ILVR/ILVR/.worktrees/vcsd-ilvr-stage2/src/stage2/data.py:123), [현재 V-off 범위](/root/workspace/ILVR/.worktrees/ours1/src/step_preference/sidecar.py:50), [END 입력을 0으로 만드는 실행 함수](/root/workspace/ILVR/.worktrees/ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py:124), [E2 START 차단](/root/workspace/ILVR/.worktrees/ours1-visual-causal/src/evaluation/visual_interventions.py:209).

## 2. 제공 사본에서 실제로 확인한 성능

아래는 파일에 기록된 모델 구분이다. 모두 원출력을 현재 공통 채점기로 재채점했으며, 문항별 정오 판정 변경은 0건이었다.

| 사본에 남은 평가 | 정답 수 | 전체 400문항 정확도 | V-off 경계 ON/OFF 대조인가? |
|---|---:|---:|---|
| Stage1 Full, 3 epoch | 173/400 | 43.25% | 아니오: Full/DoRA 비교 |
| Stage1 DoRA, 3 epoch | 158/400 | 39.50% | 아니오: Full/DoRA 비교 |
| `trained_3b_1epoch_results` | 195/400 | 48.75% | 해당 ON/OFF 쌍 없음 |
| `trained_7b_1epoch_results` | 199/400 | 49.75% | 해당 ON/OFF 쌍 없음 |
| `comt_ckpt_results` | 228/400 | 57.00% | 해당 ON/OFF 쌍 없음 |

근거: [Full/DoRA 원결과 폴더](/raid/workspace/ILVR/ILVR/results/stage1-evaluation/full-vs-dora-3epoch-20260828), [기타 Stage1 원결과 폴더](/raid/workspace/ILVR/ILVR/results/table1_stage1_comt).

TRAIN·TEST JSONL은 양쪽 서버 사본의 SHA-256이 각각 동일하다. TEST의 빈 정답 2개도 동일하게 index 57·93이다. 다만 JSONL 일치는 모든 이미지 파일의 픽셀, 모델 초기값, 실제 생성 prompt까지 같다는 증거는 아니다.

| 파일 | 양쪽에서 일치한 SHA-256 |
|---|---|
| TRAIN.jsonl | `5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8` |
| TEST.jsonl | `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618` |

사본의 루트뿐 아니라 `.worktrees/vcsd-ilvr-stage2`도 조사했다. 해당 브랜치 참조는 `14b0717d110284b1c87cf2463c5f9c6cd362ec68`이며, 여기서 확인한 실제 평가 기록은 8월 28일의 Stage1 결과다. 브랜치 참조만으로 복사된 작업 파일이 모두 커밋과 동일하다고 가정하지 않았다. `results/stage2-evaluation/.../TEST-10.jsonl`은 평가 입력이고, 완성된 예측 결과가 아니다.

**현재 확보한 사본으로는 ‘동일 실험인데 서버에 따라 효과가 반대’라는 주장을 확인할 수 없다.** 반대 결과가 없었다는 뜻이 아니라, 그 ON/OFF 두 run의 설정과 원출력이 이 사본에서 식별되지 않았다는 뜻이다.

## 3. 학습 메커니즘은 무엇이 다른가?

| 항목 | 제공 사본에서 확인한 구현·실행 | 현재 ours1 V-off 두 run |
|---|---|---|
| Stage1 학습 | EMA 교사가 만든 helper-image 목표와 latent 표현을 맞추는 항 + CE | 비교 중인 두 run은 동일 Stage1 DoRA checkpoint-215에서 시작 |
| Stage2 목표 | 정상 풀이의 CE | 정상 풀이 CE + clean/V-off 점수 차이 preference |
| Stage2 latent 생성 | 앞부분의 hidden state를 다음 latent 입력으로 순차 대입 | 정상 latent 흐름에 추가로 V-off 비교 점수를 계산 |
| V-off를 일부러 낮추는 학습 신호 | 확인한 Stage2 경로에는 없음 | 있음: clean 점수를 높이거나 V-off 점수를 낮추면 preference loss 감소 |
| 확인한 3-epoch Stage1의 학습률·기간 | 1e-5, 3 epoch | 현재 비교 대상은 Stage2, 5e-5, 212 step이므로 직접 대응하지 않음 |
| attention 경로 | 해당 Stage1 실행은 FlashAttention2 | V-off 학습은 language SDPA, vision FlashAttention2, 비교 분기 flex |
| 평가 경로 | 위 Full/DoRA 수치는 Stage1 `model.generate` 경로 | 현재 Stage2 continuous-latent 추론; E2는 SDPA 대조군과 비교 |

코드 근거: [사본 Stage1 loss](/raid/workspace/ILVR/ILVR/.worktrees/vcsd-ilvr-stage2/src/stage1/trainer.py:341), [사본 Stage2 trainer 진입점](/raid/workspace/ILVR/ILVR/.worktrees/vcsd-ilvr-stage2/src/stage2/trainer.py:230), [사본 Stage2 rollout 및 CE](/raid/workspace/ILVR/ILVR/.worktrees/vcsd-ilvr-stage2/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:2024), [현재 preference 합산](/root/workspace/ILVR/.worktrees/ours1/src/stage2/trainer.py:502). 학습 설정 근거: [사본 Stage1 launch manifest](/raid/workspace/ILVR/ILVR/.worktrees/vcsd-ilvr-stage2/outputs/stage1-timed/dora-3epoch/launch-manifest.json).

현재 목적함수를 간단히 쓰면 다음과 같다. 각 점수는 해당 풀이 구간의 정답 토큰 평균 로그확률이며, 여러 구간에서 loss를 평균한다.

$$
L = L_{\mathrm{CE}} + \frac{1}{N}\sum_{i=1}^{N}
\left[-\log\sigma\!\left(0.1\left(s_{\mathrm{clean},i}-s_{\mathrm{Voff},i}\right)\right)\right]
$$

직관적으로는 ‘정상 풀이가 차단된 풀이보다 점수가 높아야 한다’는 학습이다. **정상 풀이를 개선하지 않고 차단된 풀이만 더 못하게 만들어도 이 조건을 만족할 수 있다.** 이것이 실제 정답률 향상의 원인이라는 결론은 아니며, loss 자체가 허용하는 방향이라는 뜻이다.

| 마지막으로 기록된 학습 metric | 경계 포함 V-off | 경계 제외 V-off |
|---|---:|---:|
| clean 정답 로그확률 | −0.2470 | −0.2495 |
| V-off 정답 로그확률 | −0.2474 | −90.6995 |
| Preference loss | 0.69313 | 0.000126 |

로그에서는 clean 점수보다 V-off 점수가 극단적으로 달라졌다. 그러므로 preference loss 감소를 그대로 ‘시각 추론이 좋아졌다’는 증거로 읽으면 안 된다. 위 값은 각 run 마지막 로그 구간의 평균이지, 동일 문항에 대한 학습 전후 비교가 아니다.

CPU에서 실제 `context_preference_loss` 함수도 확인했다. 두 점수를 −0.25로 놓으면 loss는 0.693147이고, clean 점수에 대한 미분은 −0.05, V-off 점수에 대한 미분은 +0.05였다. 즉 loss 함수는 두 점수 모두에 위 방향의 신호를 준다. 이는 실제 모델 전체의 gradient 크기나 성능 변화 원인까지 입증하는 검사는 아니다.

사본의 실제 label 생성 함수와 latent parser를 이용한 CPU 검사에서는 `prompt, START, PAD, PAD, END, text` 입력을 그대로 둔 채, False일 때 START/END의 label만 −100으로 바뀌었다. ‘경계 supervision 제외’를 ‘경계 정보 제거’라고 해석할 수 없음을 확인한 검사다.

## 4. 새로 완료된 E2가 보여주는 것

| 비교 | 정상 latent 허용 | 추론에서 latent 생성 금지 | 해석 |
|---|---:|---:|---|
| 경계 포함 V-off로 학습한 Preference 모델 | 40/99 | 40/99 | 정답 9개 손실과 9개 이득이 상쇄 |
| 경계 제외 V-off로 학습한 Preference 모델 | 44/99 | 7/99 | 정답 39개 손실, 2개 이득: 37.37%p 하락 |

두 번째 행은 `visual-causal-boundaryoff-resume-20260912t041628z`의 완료 결과다. E2의 100문항에서 latent 블록 수는 전부 0이고 실행 오류는 0건이다. 정답이 비어 있는 1문항을 제외해 99문항으로 비교했다. E0 전체 파일은 E5용 추가 문항을 포함한 114개이므로, 그 전체 점수를 100문항 E2와 직접 비교하지 않았다.

이번 조사에서 두 원출력을 공통 채점기로 재채점한 결과 정오 판정 변경은 0건이었다. E2의 100개 index에 대응하는 E0와 prompt identity·checkpoint 필드가 각각 100/100 일치했다. E2는 71/100문항에서 4,096토큰 제한으로 종료됐고, 나머지 29개는 EOS로 종료됐다. E0 파일의 114문항은 모두 EOS로 종료됐다. 따라서 하락에는 출력 길이 제한에 걸리는 현상이 동반된다. 이 결과를 단순히 ‘latent를 없애니 같은 길이의 풀이에서 추론 능력만 약해졌다’고 해석해서는 안 된다.

근거: [E2 대응 비교](/root/workspace/ILVR/.worktrees/ours1-visual-causal/results/visual-causal-boundaryoff-resume-20260912t041628z/comparisons/e2-vs-e0-sdpa/summary.json), [완료 상태와 조건별 결과](/root/workspace/ILVR/.worktrees/ours1-visual-causal/results/visual-causal-boundaryoff-resume-20260912t041628z/summary.json), [E2 실행 설정](/root/workspace/ILVR/.worktrees/ours1-visual-causal/results/visual-causal-boundaryoff-resume-20260912t041628z/e2/metadata.json).

**현재 확실한 결론:** V-off 비교 분기의 경계까지 차단하며 학습한 모델은 정상 평가 점수가 더 높았지만, 그 모델에서 추론 중 latent 생성을 금지하면 성능이 크게 떨어졌다. 서로 모순되는 결과가 아니다. 다만 이 하락만으로 유용한 시각 정보에 대한 의존인지, START/END와 익숙한 풀이 형식에 대한 의존인지 분리할 수 없다. 다른 서버의 반대 결과 원인을 확정하려면 그 서버의 정확한 ON/OFF 두 run과 원출력이 추가로 필요하다.
