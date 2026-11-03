# Ours1 V-off 비교 종합: 구현 차이, 실행 근거와 적합성 판단

작성일: 2026-09-13.

이 문서는 지금까지 작성한 V-off 구현 비교, 방법론·학습 토큰 비교, 과거 평가 결과 검토와 구현 적합성 판단을 통합한다. **확인된 사실과 설계 권고를 구분하며, 권고를 구현하거나 새 실험을 승인한 문서는 아니다.** 아래 테스트·평가 수치는 기존 조사에서 확보한 기록이고, 이번 문서 정리 작업에서 학습·평가·테스트를 다시 실행하지 않았다.

understood as: 지정된 두 ours1의 비교 결과와 후속 적합성 판단을 현재 저장소의 documents/diff-ours1.md에 통합하고, 과거 잘못된 대응 관계는 정정하여 보존한다.

**권고 요약:** 현재 ours1의 실행 기반에서 **기본 latent-body attention mask + reasoning-only preference + 전체 assistant 응답의 clean CE**를 주 방법으로 유지한다. **전체 visual block 생략 ablation은 /raid 방식**을 사용하고, **현재의 END zeroing은 별도 ablation으로 구분**한다. 이는 연구 목적에 대한 적합성 판단이며, 해당 구성이 최고 정확도를 낸다는 실험적 결론이 아니다.

| 판단 대상 | 권고 |
|---|---|
| 직전 latent 본체의 국소적 기여 | 양쪽의 기본 attention mask가 적절 |
| START·latent·END 전체 생략 효과 | /raid의 full-block removal이 적절 |
| END zeroing | 기본 V-off 또는 full-block deletion으로 취급하지 않음 |
| Preference target | 의미상 다음 reasoning block을 선택; 별도 final-answer 항목은 clean CE로 학습 |
| 입력 렌더링 | 우열 미확인. 비교 조건 전체에서 하나의 형식으로 고정 |
| K/V 공유와 checkpointing | 양쪽 gradient 공유는 타당. 현재 실행 기반 유지 권고; 속도/VRAM 우위는 미측정 |
| 진단 집계 | 현재의 구간 평균에 분산 rank 집계를 결합하는 방향 권고 |
| 정상 추론 | 현재의 명시적 길이 제한 처리 권고; repetition penalty는 평가 protocol에 맞춰 고정 |

구현의 세부 내용은 1–9절, 기존 성능 기록과 혼동 정정은 10–11절, 판단 근거와 권고 범위는 12–13절에 정리한다.

**결론: 두 구현을 동일한 V-off 학습으로 취급할 수 없다.** 기본 latent-body 차단 규칙은 같고, 통제된 CPU 입력에서는 logits와 loss도 일치한다. 그러나 실제 학습 입력의 렌더링과 preference target 범위가 다르다. 경계 제거 옵션을 켜면 재생 입력, prefix 경계, 위치 번호까지 다른 intervention이 된다. 캐시 전달, activation checkpointing, 진단값 집계 및 추론 controller도 다르다.

조사일: 2026-09-13. 아래에서 **A**는 `/raid/workspace/ILVR/ILVR-ours1`, **B**는 `/root/workspace/ILVR/.worktrees/ours1`이다. 현재 상위 저장소 자체가 아니라 요청한 ours1 worktree를 비교했다.

- A는 `.git`이 없는 소스 사본이다. 따라서 이 디렉터리의 commit 또는 원 서버의 uncommitted 상태는 증명할 수 없다. A의 AGENTS.md가 지시하는 `/NHNHOME/.../ILVR/AGENTS.md`는 이 호스트에 없다. 접근 가능한 A 지침과 사용자가 제공한 지침을 적용했다.
- B의 HEAD는 `b580a393a5860ca74f9bd3f2944036cf59cb2e49`이다. 조사 시작 시 tracked 변경은 없었고, 기존 untracked 문서·캐시가 있었다.
- 이전에 조사했던 `/raid/workspace/ILVR/ILVR/.worktrees/ours1/source`를 대체 비교 대상으로 사용하지 않았다. 이번 probe는 지정된 A의 `transformers`, `src`, trainer를 직접 import했다.
- 비교한 Stage2/step_preference/Qwen 소스와 주요 진입점 38개의 [SHA-256 목록](../results/v-off-implementation-audit-20260913/source-hashes.json), 차이가 있는 공통 파일의 [diff](../results/v-off-implementation-audit-20260913/diffs/), [재현 probe](../results/v-off-implementation-audit-20260913/probe.py), [실행 결과](../results/v-off-implementation-audit-20260913/runtime-comparison.json)를 남겼다. 모델·학습 소스, 의존성, checkpoint는 수정하지 않았다.

## 1. 실제 학습 진입점과 공통 구조

A의 경로:

`run_stage2_training.sh → src/main.py → Stage2DataCollator → Stage2Trainer.compute_loss → build_context_sidecar_execution_plan → model.forward(stage2_sidecar_plan=...) → _forward_stage2 → pack_stage2_sidecar_plan → clean latent rollout → final clean decoder → retain_stage2_prefix_state → execute_stage2_sidecar_groups → trainer의 loss 합산`

B의 경로:

`run_stage2_training.sh / multinode launcher → src/main.py → Stage2DataCollator → Stage2Trainer.compute_loss → _compute_context_preference_loss → build_context_sidecar_plan → model.forward(stage2_shared_prefix_inputs=...) → _forward_stage2 → clean latent rollout → final clean decoder(K/V capture) → score_sidecars → trainer의 loss 합산`

양쪽의 활성 context-preference 경로는 같은 GT를 clean/off 조건으로 teacher forcing하여 채점한다. rejected 문장 생성, frozen reference 채점, off 분기에서의 latent 재생성은 이 경로에 없다. 이름에 `dpo_loss`, `chosen_entropy`, legacy negative helper가 남아 있다는 사실은 해당 기능이 현재 실행된다는 증거가 아니다. B는 `legacy_dpo`를 명시 선택하는 별도 경로도 보존한다.

두 구현 모두 먼저 clean prefix를 반복 실행하여 각 latent 입력을 직전 hidden state로 채우고, 완성된 clean 입력을 다시 decoder에 통과시킨다. **V-off용 K/V는 이 최종 clean forward에서 가져온다.** 이후 모든 step의 V-off suffix를 한 modality forward에 묶는다. 각 suffix는 자기 step 이전의 clean history를 참조하며, 다른 off suffix나 clean target의 미래 K/V를 참조하지 않는다.

따라서 V-only에서는 latent recurrence를 제외한 prediction decoder 실행이 `clean 1회 + V-off 1회`다. T-off까지 선택하면 3회다. 이는 전체 decoder가 2/3번만 호출된다는 뜻이 아니다. 예를 들어 latent 2개씩인 2-step probe에서는 recurrence 4회, clean prediction 1회, V-off prediction 1회가 관측됐다. B의 context 경로는 query budget이 작아도 modality를 chunk하지 않는다.

근거: [A trainer](/raid/workspace/ILVR/ILVR-ours1/src/stage2/trainer.py:351), [A model](/raid/workspace/ILVR/ILVR-ours1/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:2332), [B trainer](/root/workspace/ILVR/.worktrees/ours1/src/stage2/trainer.py:586), [B model](/root/workspace/ILVR/.worktrees/ours1/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:2473), [B sidecar 그룹 실행](/root/workspace/ILVR/.worktrees/ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py:193).

## 2. 기본 V-off: 차단 대상은 같지만 실제 학습 전체의 동등성은 아님

한 step의 입력을 다음처럼 표시한다.

`P · START · Z_k · END · t1 · t2 · ... · tn`

`P`는 원본 이미지·질문·이전 clean history이고, `Z_k`는 latent 본체다.

A의 `ablation=attention_mask`와 B의 `visual_off_suppress_latent_boundaries=False`는 모두 다음을 수행한다.

1. `[latent_start, latent_end)`의 **현재 latent 본체** K/V에 대한 직접 attention을 차단한다.
2. suffix 입력은 `END, t1, ..., t(n-1)`이다. 각 행은 `t1, ..., tn`을 예측한다.
3. clean END의 K/V는 사용하지 않는다. END **embedding**을 다시 넣고 off 조건에서 END hidden state부터 새로 계산한다.
4. clean START는 참조 가능하다. 원본 이미지, 질문, 이전 text·latent 이력도 유지된다.
5. suffix 각 행은 자기 branch의 이전 행과 자신을 볼 수 있다. 다른 step의 off 결과는 참조하지 않는다.

B는 원래 clean position_ids의 해당 행을 선택한다. A는 clean END의 세 축 position에 `0,1,...`을 더한다. canonical 연속 텍스트 구간에서는 같다. 임의의 비연속 position_ids까지 동일하다고 일반화하지 않는다.

입력 embedding을 만드는 방법도 다르다. B는 latent를 채운 clean embedding 배열에서 END/GT text 행을 선택한다. A는 suffix token ID를 `embed_tokens`에 다시 넣는다. canonical END/일반 text에서 이번 동일 가중치 probe는 같은 값을 만들었다. 멀티모달 특수 토큰이 suffix에 들어오는 임의 입력까지 동등성을 검증한 것은 아니다.

근거: [A planner](/raid/workspace/ILVR/ILVR-ours1/src/step_preference/packing.py:284), [A position·입력 검증](/raid/workspace/ILVR/ILVR-ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_sidecar.py:127), [B chosen branch](/root/workspace/ILVR/.worktrees/ours1/src/step_preference/sidecar.py:50), [B 입력 선택](/root/workspace/ILVR/.worktrees/ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py:124).

## 3. 경계 제거 ON: 서로 다른 intervention

A는 `STEP_PREFERENCE_ABLATION=remove_full_visual_block`, B는 `STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=true`를 사용한다. **이 두 ON을 같은 옵션으로 대응시키면 안 된다.**

| 항목 | A: full visual block 생략 | B: boundary suppression |
|---|---|---|
| 제거/차단 범위 | `[prev_text_end, target_text_start)`의 START+latent+END | `[latent_start-1, latent_end+1)`의 START+latent+END |
| 재생 시작 행 | START 직전 마지막 text | 원래 END |
| 첫 입력 | 직전 text의 실제 embedding | END embedding을 선택한 뒤 0으로 덮어씀 |
| 이후 입력 | 동일 GT의 마지막 토큰 전까지 | 동일 GT의 마지막 토큰 전까지 |
| 첫 target 예측 위치 | 직전 text의 position | 원래 END의 position |
| 위치 번호 | visual block 길이만큼 앞으로 당긴 local positions | 원래 위치 유지 |
| clean prefix 범위 | 재생할 직전 text보다 앞 | END보다 앞에서 START/latent 차단 |
| 직전 text 처리 | clean K/V 대신 그 token을 재생 | clean K/V를 계속 참조 |
| 공유 clean K/V 배열 | 물리 삭제하지 않음 | 물리 삭제하지 않음 |
| clean CE와 다음 step의 clean history | 그대로 | 그대로 |
| T-off 병용 | `visual_text`에서도 허용, text 분기는 유지 | V-only일 때만 허용 |

B는 END의 **입력**을 0으로 만들 뿐, 그 query 행을 삭제하지 않는다. 따라서 위치 간격과 한 행의 decoder 계산이 남는다. 0 입력에서 나오는 hidden state/logits까지 0이라는 뜻도 아니다. 또한 이후 GT query는 이 off branch에서 재생된 END 행을 참조할 수 있다.

A는 이전 text 토큰으로 첫 target을 예측하는 문맥을 구성한다. source 전체의 토큰 배열, 모든 clean cache, 모든 과거 latent를 삭제하는 구현은 아니다. local branch에서 현재 visual block을 건너뛰는 것이다.

직접 관측한 두 step의 예:

```text
clean IDs: [11, START, Z, Z, END, 12, 14, START, Z, Z, END, 13, 15, 16]
positions:   0    1    2  3   4    5   6     7    8  9  10   11  12  13

기본 양쪽 step0: [END, 12]       at [4, 5]      → targets [12, 14]
A ON step0:      [11, 12]        at [0, 1]      → targets [12, 14]
B ON step0:      [zero(END), 12] at [4, 5]      → targets [12, 14]

기본 양쪽 step1: [END, 13, 15]       at [10, 11, 12] → targets [13, 15, 16]
A ON step1:      [14, 13, 15]        at [6, 7, 8]    → targets [13, 15, 16]
B ON step1:      [zero(END), 13, 15] at [10, 11, 12] → targets [13, 15, 16]
```

A의 기존 테스트에는 실제 토큰 배열에서 block을 삭제한 독립 forward와 sidecar logits/DoRA gradients를 비교하는 검사가 있다. 이번 CPU 실행에서 통과했다. 이 검증 범위는 tiny 모델·dropout 0이며, production 이미지/DoRA dropout/GPU에서 모든 실행이 물리 삭제 재계산과 같다고 증명하지는 않는다.

근거: [A 경계 이동](/raid/workspace/ILVR/ILVR-ours1/src/step_preference/packing.py:299), [A 삭제 oracle 테스트](/raid/workspace/ILVR/ILVR-ours1/tests/step_preference/test_optional_visual_removal.py:33), [B zeroing과 검증](/root/workspace/ILVR/.worktrees/ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py:35), [B 옵션 제약](/root/workspace/ILVR/.worktrees/ours1/src/step_preference/sidecar.py:221).

## 4. 실제 preference target와 입력 렌더링이 다름

**기본 OFF에서도 학습이 달라지는 직접적인 차이다.** 양쪽 모두 자신의 clean/off 사이에서는 같은 GT를 사용하지만, **A와 B가 GT의 같은 구간을 점수화하는 것은 아니다.**

A는 tokenized latent block 위치를 기준으로 `END 다음 → 다음 START 직전`, 마지막 step에서는 `END 다음 → assistant 종료 marker 직전`을 target으로 삼는다. 사이에 여러 text plan 항목이 있으면 모두 들어간다.

B는 sequence_plan의 개별 문자 구간과 tokenizer offset mapping을 대조한다. latent의 **바로 다음 text 항목 하나**만 target으로 삼는다. 별도 final-answer text 항목이 뒤따르면 그 항목은 preference/entropy에서 빠진다. 다만 clean CE에는 여전히 포함된다.

B의 main은 preference가 켜지면 `adjacent-text-space-lf-v1`을 선택하여 연속 text 항목 사이에 `" \n"`을 삽입한다. A는 모든 plan 항목을 그대로 이어 붙인다. 이 차이는 clean CE 입력·supervised token 수·final answer의 위치에도 영향을 준다.

원본 `data/TRAIN.jsonl`을 직접 센 결과, **3,450행 중 3,449행**에 `latent → text → text`처럼 하나의 latent 뒤에 연속 text가 있었다. 첫 사례의 추가 text는 `The final answer is: 70°`다. 이 수치는 원본 TRAIN 전체의 구조 통계이며, 양쪽 역사적 run의 retained subset가 동일하다는 증명은 아니다.

동일한 실제 tokenizer/processor, 동일한 원본 이미지와 TRAIN 첫 행을 양쪽 실제 collator에 넣은 결과:

| 항목 | A | B |
|---|---:|---:|
| 전체 input 길이 | 212 | 213 |
| clean CE supervised 토큰 | 93 | 94 |
| 첫/마지막 step target interval | `[134,210)` | `[134,201)` |
| preference target 토큰 수 | 76 | 67 |
| target에 `The final answer is: 70°` 포함 | 예 | 아니오 |
| 이전 text 시작 | 111 | 112 |

A의 이전 text 시작에는 assistant header 뒤 newline도 들어간다. B는 실제 plan text 시작을 사용한다. 이 시작 차이는 특히 T-off 범위에 영향을 준다. V-off의 이전 text 끝과 latent 위치는 이 샘플에서 같았다.

A는 latent marker를 기준으로 span을 정하기 때문에 B의 fast-tokenizer/정확한 plan-boundary offset 계약을 요구하지 않는다. B는 token이 plan 경계를 가로지르거나 image-placeholder 확장 뒤 tokenization이 processor 입력과 다르면 거절한다. 따라서 tokenization/수용 가능한 입력의 계약도 다르다.

근거: [A target 도출](/raid/workspace/ILVR/ILVR-ours1/src/step_preference/spans.py:60), [A 이어붙이기](/raid/workspace/ILVR/ILVR-ours1/src/stage2/data.py:134), [B rendering 선택](/root/workspace/ILVR/.worktrees/ours1/src/main.py:498), [B 즉시 인접 text 선택](/root/workspace/ILVR/.worktrees/ours1/src/step_preference/spans.py:227), [B collator](/root/workspace/ILVR/.worktrees/ours1/src/stage2/data.py:340), [실제 collator 결과](../results/v-off-implementation-audit-20260913/runtime-comparison.json), [전체 데이터 구조 통계](../results/v-off-implementation-audit-20260913/dataset-structure.json).

## 5. loss 수식, gradient와 해석

활성 V-only 경로의 공통 수식은 다음이다.

\[
s_k^c=\frac{1}{|T_k|}\sum_j\log p_\theta(t_{k,j}\mid c_k,t_{k,<j}),\quad
s_k^v=\frac{1}{|T_k|}\sum_j\log p_\theta(t_{k,j}\mid c_k^{-V},t_{k,<j})
\]
\[
L=L_{CE}+\lambda_{pref}\frac1K\sum_k[-\log\sigma(\beta(s_k^c-s_k^v-m))]
+\lambda_H\frac1K\sum_k\frac1{|T_k|}\sum_j\max(0,m_H-(H^v_{k,j}-H^c_{k,j})).
\]

- preference는 각 step에서 sigmoid를 적용한 뒤 step 평균을 낸다. 전체 token 합이나 평균 margin에 sigmoid를 한 번 적용하는 방식이 아니다. entropy도 step 내부 token 평균 후 step 평균이다.
- active context 경로에서 CE 계수는 양쪽 모두 1이다. B에 남아 있는 일반 `lambda_ce` 필드만 보고 이 경로에서 가중치가 적용된다고 해석하지 않는다.
- clean/off 점수 둘 다 autograd에 연결된다. clean K/V를 detach한 fixed teacher로 사용하지 않는다.
- 현재 step latent 본체에 대한 직접 attention edge를 차단한다. 하지만 이전 clean latent/text, 원본 이미지와 질문은 남는다. **V-off를 모든 시각 정보 제거 또는 이미지 없는 baseline으로 읽으면 안 된다.**
- boundary suppression에서 control embedding 변화에 대한 불변성과 gradient 차단을 확인한 B 테스트는 1-step 조건이다. 여러 step에서 과거 control token의 영향까지 모두 지워졌다는 증거로 확대하지 않는다.
- preference loss는 clean 정답 점수를 올려도, off 정답 점수를 낮춰도 감소한다. 따라서 작은 preference loss 자체는 clean 추론 정확도 향상을 입증하지 않는다.
- 두 서버의 수식이 같아도 4절의 `T_k` 범위 차이 때문에 실제 목적함수는 달라진다. A는 final-answer 토큰에도 clean/off gap 및 entropy 신호를 주고 B는 해당 별도 항목에는 clean CE만 준다.

진단/API 차이도 있다. A의 primitive는 FP32 변환 후 loss·pair loss·margin을 구조체로 반환하고 음수 preference margin을 거절한다. B의 primitive는 scalar tensor를 반환하며 유한한 음수 margin도 허용한다. A primitive는 빈 tensor에 differentiable zero를 반환하지만 실제 sidecar packing은 비어 있는 modality를 거절한다. B trainer는 reasoning step이 없으면 clean만 계산하고 auxiliary metric을 0으로 처리한다. 기본 FP32/비음수 margin/비어 있지 않은 정상 입력에서는 이 API 차이가 이번 loss 일치를 깨지 않았다.

근거: [A loss](/raid/workspace/ILVR/ILVR-ours1/src/step_preference/dpo_loss.py:27), [A aggregation](/raid/workspace/ILVR/ILVR-ours1/src/stage2/trainer.py:478), [B loss](/root/workspace/ILVR/.worktrees/ours1/src/step_preference/dpo_loss.py:122), [B aggregation](/root/workspace/ILVR/.worktrees/ours1/src/stage2/trainer.py:498).

## 6. 캐시·메모리·checkpointing·backend의 차이

| 항목 | A | B |
|---|---|---|
| 최종 clean forward | `use_cache=True`, DynamicCache 생성 | `use_cache=False`, layer별 K/V 별도 capture |
| sidecar로 전달 | 보존한 tensor를 새 DynamicCache wrapper로 재구성 | `stage2_prefix_key_values` tuple |
| suffix K/V 결합 | cache.update 경로 | attention에서 prefix/suffix `torch.cat` |
| 분기 간 상태 공유 | wrapper는 분리, 원래 clean tensor 공유 | 동일 clean tuple을 읽고 분기별 suffix 결합 |
| autograd | producer tensor를 보존하고 differentiability 검사 | capture tensor와 checkpoint 인자로 gradient 연결 |
| off 출력 | branch별 full-vocabulary logits를 반환, trainer에서 축약 | score 함수 안에서 mean logp·token entropy로 축약하여 반환 |
| sidecar activation checkpoint | 별도 wrapping 없음 | score+LM head+축약 함수를 기본 checkpoint, RNG 보존 |
| native decoder checkpointing | sidecar 사용 중 train 상태에서 금지 | clean/latent layer checkpointing 지원; suffix는 score checkpoint 사용 |
| 별도 latent checkpointing | `stage2_latent_activation_checkpointing`, 전체 latent-prefix 함수 재실행 | 이 별도 옵션 없음; native layer checkpoint 사용 |
| language/sidecar backend | eager, sdpa, flex_attention; split 옵션이면 clean SDPA + sidecar Flex | language eager/sdpa, sidecar dense/flex 별도 옵션 |
| vision backend | split 옵션이 FlashAttention2를 선택 | vision backend를 별도 인자로 선택 |
| padding | 전체 clean 길이의 cache와 binary valid mask | 유효 root 길이까지만 K/V 사용, contiguous right-padding 요구 |

A에 존재하는 `differentiable_cache` rollout 선택지만 보고 V-off sidecar가 latent recurrence까지 cache로 최적화한다고 해석하면 안 된다. `stage2_sidecar_plan`이 들어오면 top-level은 `_forward_stage2`의 clean no-cache rollout 경로로 분기한다. 이번 probe에서도 latent recurrence는 매번 prefix 전체를 처리했다.

B에서는 sidecar가 clean vocabulary scoring보다 먼저 실행되고, A에서는 clean vocabulary scoring 뒤 sidecar가 실행된다. decoder clean→sidecar 순서는 같다. LM head가 동일하다는 사실만으로 모든 dropout/RNG/메모리 실행도 같다고 일반화하지 않는다.

관측: B는 기본 설정에서도 backward 중 V-off decoder 1회 재실행했다. A는 checkpoint 없는 기본 설정에서 backward decoder 재실행이 없었다. A selective-latent checkpoint를 켜면 backward에서 4개 latent prefix decoder가 재실행됐다. B의 native checkpoint는 layer 단위이므로 decoder 진입 hook 숫자만으로 layer 재실행을 세어서는 안 된다.

Flex의 head-dimension padding, mask 생성과 vision rotary 구현에도 소스 차이가 있다. 이 보고서의 동일 logits/gradient 수치는 **CPU SDPA text-only tiny model** 비교다. 실제 이미지 tower, Flex/Flash GPU kernel, bf16 및 DoRA dropout 0.05에서 같은 수치·성능·메모리 사용량을 확인한 것은 아니다. 성능 우열이나 절대 VRAM 차이는 측정하지 않았다.

근거: [A cache 보존](/raid/workspace/ILVR/ILVR-ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_prefix_cache.py), [A replay](/raid/workspace/ILVR/ILVR-ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_sidecar.py:337), [A top-level guard](/raid/workspace/ILVR/ILVR-ours1/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:2762), [B capture/dispatch](/root/workspace/ILVR/.worktrees/ours1/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1569), [B scoring/checkpoint](/root/workspace/ILVR/.worktrees/ours1/transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py:157).

## 7. 옵션·운영 설정과 로그 해석

| 목적 | A | B |
|---|---|---|
| 활성화 | `STAGE2_PHASE=stage2b` | `STEP_PREFERENCE_ENABLED=true`, `STEP_PREFERENCE_MODE=stage2b`, `STEP_PREFERENCE_OBJECTIVE=context_preference` |
| V-only | `STEP_PREFERENCE_MODALITIES=visual` | `STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true` |
| 기본 V-off | `STEP_PREFERENCE_ABLATION=attention_mask` | boundary suppression 기본 False |
| 경계 옵션 | `...ABLATION=remove_full_visual_block` | `STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=true` |
| preference 계수 | `STEP_PREFERENCE_LAMBDA_PREFERENCE` | `STEP_PREFERENCE_WEIGHT` |
| entropy 계수 | `STEP_PREFERENCE_LAMBDA_ENTROPY` | `STEP_ENTROPY_WEIGHT` |
| backend 조합 | `STAGE2_SPLIT_ATTENTION=true` | language/vision/sidecar 각각 설정 |

둘 다 설정 기본 modality는 V+T이고, V-only는 명시적으로 골라야 한다. A의 comparison launcher에서 `off`라는 인자는 **V-off 자체를 끈다는 뜻이 아니라 경계 제거 옵션을 끈 V-off**다.

이번에 직접 읽은 A comparison launcher의 `COMPARISON_INITIALIZATION=stage1` 분기와 B 기존 V-only launch는 beta 0.1, preference weight 1, margin 0, entropy weight 0, latent size 8, DoRA r32/alpha64/dropout0.05, LR 5e-5, 212 step을 설정한다. 반면 초기 Stage1 경로, 분산 batch 구성, checkpointing은 다르다. A는 1 process×micro1×accum16, B는 4 ranks×micro1×accum4 구성이다. A launcher는 초기화 인자를 생략하면 stage2a 분기로 간다. 이 스크립트 설정 비교를 모든 과거 run의 runtime 동일성 증거로 취급하지 않았다.

진단 집계는 A가 logging 대상 microbatch의 최신 값에 대해 rank 평균을 내는 방식이고, B는 microbatch마다 누적하여 logging 구간의 local 평균을 기록한다. metric 이름도 A `loss/context_preference_all`, B `loss/context_preference` 등으로 다르다. **두 로그의 같은 step 숫자만 맞춰 loss를 직접 비교하면 집계 단위가 어긋날 수 있다.**

근거: [A launcher](/raid/workspace/ILVR/ILVR-ours1/scripts/run_visual_boundary_comparison.sh:1), [B launch](/root/workspace/ILVR/.worktrees/ours1/logs/ours1-ctx-pref-vonly-direct-s1-dora-egb16-lr5e5-default-s212-r2-20260910t022900z/launch.sh:1), [A diagnostics](/raid/workspace/ILVR/ILVR-ours1/src/stage2/trainer.py:762), [B metrics](/root/workspace/ILVR/.worktrees/ours1/src/stage2/trainer.py:469), [B log 평균](/root/workspace/ILVR/.worktrees/ours1/src/stage2/trainer.py:1416).

## 8. 일반 추론에서의 동작

두 경계 옵션은 학습용 V-off 비교 분기에 적용된다. 정상 inference에서 START/latent/END를 자동으로 제거하지 않는다. 두 controller 모두 정상적으로 START를 선택하면 continuous latent block을 계산한다. 따라서 training boundary ON을 inference latent-disable과 동일시하면 안 된다.

다만 inference 구현은 다르다.

| 항목 | A Stage2 controller | B continuous-latent controller |
|---|---|---|
| repetition penalty | 적용 코드/필드 없음 | 적용; dataclass 기본 1.05, builder는 전달값/generation config도 읽음 |
| config 기본 max_new_tokens | 64 | 4096 |
| max_new_tokens 직전 START | block 전체를 더해 예산을 초과할 수 있음 | 남은 예산에 맞춰 suffix 제한 후 종료 |
| max_sequence_length 직전 START | 완성 block이 한도를 넘으면 ValueError | 용량 범위까지만 suffix 추가하고 stop reason 반환 |
| 생성 결과 | sequences, optional latent states | 추가 stop_reason, 생성 수·START 수, 선택적 post-END logits |
| latent-disable | 해당 인자 없음 | 별도 `disable_latent_generation` 지원 |

생성 길이·penalty는 외부 평가 진입점의 명시 설정으로 달라질 수 있다. 여기서는 구현과 기본값을 비교했으며, 특정 400문항 정확도 차이의 원인으로 확정하지 않았다.

근거: [A controller](/raid/workspace/ILVR/ILVR-ours1/src/stage2/inference.py:16), [B wrapper](/root/workspace/ILVR/.worktrees/ours1/src/stage2/inference.py:14), [B generation](/root/workspace/ILVR/.worktrees/ours1/src/evaluation/continuous_latent.py:183).

## 9. 이번 실행으로 확인한 범위

CPU 비교는 A/B의 실제 Transformers package와 실제 trainer를 각각 별도 process에서 import했다. 2-layer tiny Qwen, FP32, SDPA, attention dropout 0, 동일 token/label/span을 사용했다. 모든 초기 parameter를 이름별 고정 난수로 채우고 전체 parameter byte SHA-256이 양쪽에서 같은지 검증했다. 모델과 gradient 배열은 process 간 메모리로만 전달했고 weights 파일은 만들지 않았다. trainer의 `return_outputs=True` 진단 경로를 사용했다.

| 비교 모드 | clean logits 최대 절대 차이 | V-off logits 최대 절대 차이 | total loss 차이 | parameter gradient 최대 절대 차이 |
|---|---:|---:|---:|---:|
| 기본 V-off | 0 | 0 | 0 | 1.7416e-7 |
| 각자의 경계 ON | 0 | 0.713393 | 0.00284243 | 0.0215909 |
| 기본 V-off + 각자의 checkpointing | 0 | 0 | 0 | 1.7416e-7 |

gradient의 None/non-None 불일치는 없었다. 기본 V-off의 CPU 일치는 **입력과 target까지 강제로 같게 했을 때**의 결과이며, 실제 collator 차이를 상쇄한 검증이다. 경계 ON 수치는 비동등성을 보여주는 witness이지 학습 품질의 우열이 아니다.

선택한 기존 테스트도 한 번씩 실행했다.

- B: `test_context_sidecars.py`, `test_context_preference.py` → **45 passed, 2 skipped**. [로그](/root/workspace/ILVR/logs/v-off-implementation-audit-20260913/current-tests.txt)
- A: `test_optional_visual_removal.py`, `test_visual_only.py`, `test_context_preference.py`의 CPU 관련 선택 → **17 passed, 4 skipped, 2 deselected**. GPU 검사는 CUDA를 숨겨 skip했고, 외부 원 서버 경로를 전제하는 launcher/metadata CLI 검사는 선택에서 제외했다. [로그](/root/workspace/ILVR/logs/v-off-implementation-audit-20260913/raid-tests.txt)
- 동일 실제 processor·원본 이미지의 첫 TRAIN row에 대한 양쪽 collator 실행 성공. [probe 출력](/root/workspace/ILVR/logs/v-off-implementation-audit-20260913/probe-run.txt)

새로 GPU 학습, 전체 benchmark 평가, 원 서버 환경 재현을 실행하지 않았다. 이 조사로 확정한 것은 **target/렌더링 차이, boundary intervention 비동등성, cache/checkpoint/추론 구현 차이**, 그리고 한정된 CPU 기본 모드의 수치 일치다. 기존 보고서에서 수식이나 기본 attention 차단 규칙이 같다는 이유로 학습 전체가 같다고 읽힐 수 있었던 부분은 이 범위로 좁혀야 한다. 서버 간 성능 차이의 원인별 기여량은 아직 분리 검증되지 않았다.


## 10. 기존 성능 비교 기록과 학습 신호

이 절은 기존 조사 기록의 요약이다. 소스 비교 대상 A와 별개로, 타 서버의 과거 run 설정·평가 원출력은 `/raid/workspace/ILVR/ILVR/.worktrees/ours1/` bundle에서 확인했다. 소스 사본 디렉터리 A 자체에 과거 run의 weights·실행 이력이 모두 들어 있다는 의미는 아니다. 이 출처 차이를 유지한다.

| 서버 / 학습용 V-off 옵션 | 정상 추론 평가 정답 수 | 정확도 | 같은 서버의 ON−OFF |
|---|---:|---:|---:|
| 현재 ours1: 기본 mask, 경계 suppression OFF | 175/400 | 43.75% | — |
| 현재 ours1: boundary suppression ON | 199/400 | 49.75% | +24개, +6.00%p |
| 타 서버 ours1: attention_mask | 216/400 | 54.00% | — |
| 타 서버 ours1: remove_full_visual_block | 201/400 | 50.25% | −15개, −3.75%p |

타 서버의 216/201은 최초에는 사용자 제공 수치였지만, 이후 원출력의 400행을 원본 TEST의 `original_final_answer`와 공통 scorer로 재채점했다. 저장 정오 판정과의 불일치 및 오류는 각각 0개였다. 따라서 최신 상태에서는 **“타 서버 코드/원출력 미확인”을 미해결 문제로 반복하면 안 된다.**

다만 두 서버의 ON은 3절처럼 서로 다른 intervention이다. 기본 OFF부터 216 대 175로 41개(10.25%p) 차이가 있으며, 4절의 target/렌더링도 다르다. 이 수치만으로 경계 처리 하나가 성능 역전을 일으켰다고 결론 낼 수 없다.

기존 run 설정 대조에서 확인한 추가 차이:

- Stage1 초기 artifact는 현재 `stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/prepared-checkpoint-215`, 타 서버는 `stage1-dora-frozen-b1-lr5e5-zero1-1epoch-exact32768-20260902/checkpoint-431`이었다. 같은 가중치라는 검증은 없다.
- Stage2 global batch는 16으로 같았지만 현재 4 ranks×micro1×accum4, 타 서버 1 rank×micro1×accum16이었다. sampler 경로와 update에 포함되는 sample 순서까지 동일하다는 증거는 없다.
- 현재는 native gradient checkpointing, 타 서버는 selective latent checkpointing이었다.
- 양쪽 retained 수는 3,379/3,450으로 보고됐지만 타 서버의 역사적 retained ID 목록은 확보되지 않아 동일 subset/순서를 증명하지 못했다.
- 기존 방법론 조사에서 현재 175/400 run의 length manifest와 raw sequence_plan을 대조한 결과, 현재 retained 3,379개 중 3,378개에서 latent 뒤 연속 text 및 별도 final-answer 구조가 확인됐다. 이것은 원본 구조 집계이며 양쪽 전수 processor 실행 결과가 아니다.
- tokenizer/chat-template/processor JSON 내용이 같아도 렌더링 규칙과 추론 repetition penalty가 다르므로 실제 입력·출력 절차가 같다고 볼 수 없다.

현재 서버의 마지막 기록에서 기본 V-off 평균 정답 log-probability는 약 −0.2474, boundary suppression ON은 약 −90.6995였다. clean 점수는 각각 약 −0.2470/−0.2495, CE는 0.2124/0.2150으로 기록됐다. 이는 writer-rank의 logging 구간 평균이며, 동일 고정 문항에서의 전후 비교가 아니다. 관측상 preference 감소의 큰 변화는 off 점수 하락이었다. 타 서버도 마지막 step200 snapshot에서 ON preference가 약 0.000026803으로 작아졌지만 정상 평가 정확도는 OFF보다 낮았다.

이 기록은 **preference loss 감소를 clean 정확도 향상의 증거로 대체하면 안 된다는 점**을 보여준다. 현재 모델이 END zeroing을 특정 shortcut으로 이용했다는 인과적 증명이나, 어느 요인의 기여가 몇 %p인지에 대한 분해는 아니다. 서버별 loss의 집계 방식도 7절처럼 다르다.

근거: [타 서버 baseline·원출력 검증](../results/baseline-v-off-new-ours1-20260913/verification.md), [방법론·retained 구조 집계](../results/methodology-loss-token-audit-20260913/verification.md), [기존 비교와 학습 로그 분석](analysis_v-off_compare.md).

## 11. 혼동하면 안 되는 설정과 과거 비교의 정정

| 설정/실험 | 실제 의미 | 의미하지 않는 것 |
|---|---|---|
| `supervise_latent_boundaries=False` | START/END의 label을 −100으로 만들어 직접 CE supervision에서 제외 | 입력 삭제 또는 attention 차단 |
| 현재 boundary suppression ON | V-off 분기에서 경계 접근 차단 + END 입력 zeroing | 정상 clean 학습/추론에서 latent 제거 |
| 타 서버 full-block removal ON | 해당 V-off suffix의 유효 문맥에서 현재 visual block 생략 | 공유 clean cache나 과거 모든 visual 정보 삭제 |
| inference latent-disable(E2) | 생성 중 START 선택을 막아 latent block 생성을 금지 | 학습용 V-off masking과 동일한 intervention |

이전에 확인한 현재 175/199 두 학습은 `supervise_latent_boundaries=True`였다. 따라서 “경계 포함/제외 학습”이라는 축약은 V-off 분기의 추가 차단을 가리켜야 하며, 모델 전체의 START/END supervision 유무로 읽으면 안 된다.

별도 inference E2 비교 기록은 다음과 같다.

| 현재 모델 | 정상 latent 허용 | latent 생성 금지 | 비교 범위 |
|---|---:|---:|---|
| 기본 V-off 학습 | 40/99 | 40/99 | 100문항 중 gold가 있는 99개 |
| boundary suppression 학습 | 44/99 | 7/99 | 같은 방식의 99개 |

두 번째 비교는 `visual-causal-boundaryoff-resume-20260912t041628z`의 결과다. E2는 100문항 모두 latent block 0개, 실행 오류 0개였고 71/100문항이 4,096-token 제한으로 종료됐다. 공통 채점기로 재채점했을 때 정오 판정 변경은 없었고, 대응 E0/E2의 prompt identity와 checkpoint 필드가 100/100 일치했다는 기존 검증 기록이 있다. E0 전체 파일은 추가 문항을 포함한 114행이어서 그 전체 점수를 E2 100행과 직접 비교하지 않았다.

이 결과는 정상 추론에서 latent 생성 경로를 막았을 때 해당 모델의 출력·성능이 달라졌음을 보여준다. 유용한 시각 내용 의존과 익숙한 START/END 형식 의존, 출력 길이 제한 영향을 분리하지는 못한다. 서버 간 400문항 ON/OFF 차이의 원인으로 대신 사용할 수 없다.

과거 비교의 정정 원칙:

1. 초기 `/raid/.../ours-trial1` 또는 `vcsd-ilvr-stage2` 조사는 요청한 ours1과 다른 구현에 대한 자료다. 그 차이를 두 ours1의 차이로 대입하지 않는다.
2. 양쪽 경로가 같은 checkpoint symlink/device/inode를 가리킨 경우에는 독립적인 두 서버 결과로 세지 않는다.
3. “상대 ours1이 없다”, “216/201 원출력이 미확인이다”라는 과거 문구는 자료 배치 전 상태다. 현재는 코드 및 원출력 확인으로 해소됐다.
4. “기본 차단 규칙과 loss가 같다”는 결과는 동일 span/target 조건의 주장으로 한정한다. 실제 collator target 차이가 확인된 뒤에도 학습 전체가 같다고 결론 내리지 않는다.
5. 아직 미해결인 것은 실제 GPU 학습의 동등성과 성능 차이의 원인별 기여량이다.

근거: [기존 조사와 정정 이력](analysis_v-off_compare.md), [E2 대응 결과](/root/workspace/ILVR/.worktrees/ours1-visual-causal/results/visual-causal-boundaryoff-resume-20260912t041628z/comparisons/e2-vs-e0-sdpa/summary.json), [E2 설정](/root/workspace/ILVR/.worktrees/ours1-visual-causal/results/visual-causal-boundaryoff-resume-20260912t041628z/e2/metadata.json).

## 12. 어떤 구현이 적절한가: 판단 근거

이 절은 확인된 코드·실행 결과를 바탕으로 한 **권고**다. 사용자의 구현 변경 승인이나 실험 결과를 뜻하지 않는다. 판단 기준은 “clean history를 유지한 채 직전 latent의 다음 reasoning에 대한 국소적 기여를 학습한다”는 연구 목적이다. 초기 문서의 generated-negative/reference-DPO 설명은 후속 same-GT context-preference 계약으로 대체됐으므로 판단에 재도입하지 않는다.

### 12.1 기본 V-off는 양쪽 모두 적절

latent-body attention mask는 clean/off의 token scaffold와 logical positions를 유지하면서 현재 latent 접근을 차단한다. 앞서 같은 입력·target을 준 CPU 비교가 일치했으므로 `/raid`와 현재 중 한쪽만 기본 masking이 옳다고 판단할 근거는 없다.

기본 off 점수가 clean과 비슷하다고 masking 오류라고 단정해서도 안 된다. 원본 이미지, 질문, 이전 reasoning, teacher-forced target prefix는 의도적으로 남아 있다. 이 실험은 total visual information removal을 측정하지 않는다.

### 12.2 전체 블록 제거에는 /raid가 적절, END zeroing은 별도 조건

“START·latent·END가 없는 local 문맥”을 구성하려는 목적에는 `/raid`의 block 이전 text 재생과 position 압축이 더 명확히 대응한다. 반면 현재 ON은 END 입력을 0으로 바꾸고 그 query 행과 원래 위치를 남긴다. 두 가지를 같은 ‘삭제’라고 부르면 안 된다.

현재 ON은 latent·경계 접근 차단 외에 **첫 target을 예측하는 입력 자체를 인위적으로 변경**한다. clean/off 점수 차이가 커져도 latent 내용의 기여와 첫 입력 변경의 영향을 분리할 수 없다. 현재 loss는 off 정답 확률을 낮춰도 줄어들기 때문에 모델이 분기 차이를 활용할 가능성을 배제할 수 없다. 이는 확인이 필요한 위험이며, 실제로 그 shortcut을 학습했다는 결론은 아니다.

`/raid` full removal도 위치 간격·경계·첫 예측 문맥을 함께 바꾸므로 기본 latent-content 실험을 대신할 수 없다. 따라서 **주 방법은 기본 mask, 전체 block removal은 보조 ablation, END zeroing은 추가적인 별도 ablation**이라는 구분을 권한다.

### 12.3 Preference target은 현재 reasoning-only를 권고

다음 reasoning block을 scoring unit으로 삼으려는 목적에는 별도 final-answer 항목을 분리하는 현재 방식이 더 적절하다. `/raid`처럼 둘을 합치면 reasoning token과 answer token의 gap이 같은 평균에 들어가 학습 신호와 해석이 달라진다.

final-answer 토큰은 앞의 GT reasoning을 teacher-force한 상태에서 채점된다. 따라서 그 likelihood를 포함하는 것만으로 모델이 스스로 reasoning을 생성하여 올바른 답에 도달하는 능력을 직접 측정했다고 볼 수 없다. 이 conditioning은 설계된 비교이지 그 자체로 leakage 오류는 아니다.

단, 단위의 기준은 **의미 있는 reasoning 구간**이어야 한다. 데이터가 하나의 reasoning을 여러 text 항목으로 나눴다면 “항상 첫 text 항목만”이라는 구현 규칙은 재검토해야 한다. 이번에 확인된 reasoning text + 별도 final-answer text 구조에서는 현재 분리가 타당하다. final-answer preference가 별도 목표라면 명시적인 항/가중치로 비교하는 방향을 권고하며, 이 변경은 아직 승인·구현하지 않았다.

### 12.4 입력 형식·실행 기반·로그·추론

- **입력 형식:** `" \n"`과 단순 연결 중 어느 쪽이 더 높은 정확도를 내는지는 검증되지 않았다. V-off 실험과 독립적으로 하나의 rendering protocol을 정하고 모든 대조군에 동일하게 적용해야 한다. 기존 실행 결과를 해석하면서 구분자를 사소한 차이로 지우면 안 된다.
- **Cache:** DynamicCache인지 tuple인지가 방법론의 옳고 그름을 결정하지 않는다. differentiability, branch 격리, 동일한 clean producer tensor 사용이 기준이며 양쪽 모두 이에 대응하는 구현이 있다.
- **Scoring/checkpointing:** sidecar 내부에서 logp/entropy를 축약하고 checkpointing을 지원하는 현재 기반을 유지하는 것이 합리적이다. 이는 유지·메모리 관리 구조에 대한 권고이며 실측 속도 또는 VRAM 우위 주장은 아니다.
- **로그:** 현재의 logging 구간 평균 방식에 `/raid`의 rank 집계 개념을 결합하되, sample/step/token 중 분모를 명확히 하는 방향이 적절하다. 단순한 mean-of-means가 항상 전역 step 평균인 것은 아니므로 실제 분모가 같다는 조건 또는 합/개수 집계가 필요하다. 아직 구현된 변경은 아니다.
- **추론:** 생성 budget을 명시적으로 지키고 종료 사유를 반환하는 현재 방식이 적절하다. 단, 부분 latent block을 남기는 현재 종료 동작까지 모든 downstream consumer에 적합하다고 일반화하지 않는다. repetition penalty 1.05가 본질적으로 더 좋은 설정이라는 근거는 없고 평가 protocol에 맞춰 고정해야 한다.

근거: [기본 intervention 정의](ILVR_stepwise_modality_dpo_spec_ko.md:238), [logical position 계약](ILVR_stepwise_modality_dpo_spec_ko.md:803), [same-GT context-preference 명세](TODO-ours1-20260906.md:40), [직전 적합성 판단에 사용한 구현 비교](../results/v-off-implementation-audit-20260913/report.ko.md).

## 13. 권고 구성과 남은 확인 범위

권고 구성은 다음과 같다.

```text
실행 기반:       현재 ours1
주 V-off:        현재 latent 본체의 attention만 차단, 경계·위치 유지
Preference:     다음 reasoning 구간의 same-GT clean/off 비교
Clean CE:       전체 assistant 응답, 별도 final-answer 포함
보조 ablation:  /raid의 full visual block removal
별도 ablation:  현재 END embedding zeroing
실험 공통 조건: 동일 initialization, 데이터/target, rendering, 평가 controller와 설정
```

어느 방식이 최고 정확도를 내는지, target/구분자/boundary/초기화/평가 설정이 각각 몇 %p에 기여하는지는 아직 분리되지 않았다. 이를 판단할 추가 대조에서는 각 요인을 독립적으로 바꾸고, clean 평가 정확도와 clean/off 점수를 함께 봐야 한다. 작은 preference loss나 off 점수 붕괴를 성공 지표로 단독 사용하지 않는다. 추가 GPU 실험은 이 문서 정리 요청에 포함되지 않으며 실행하지 않았다.

원본 근거 묶음:

- [구현 비교 보고서·코드 연결](../results/v-off-implementation-audit-20260913/report.ko.md)
- [실제 trainer/모델 CPU 비교 및 collator 결과](../results/v-off-implementation-audit-20260913/runtime-comparison.json)
- [재현 probe](../results/v-off-implementation-audit-20260913/probe.py), [비교 대상 소스 hashes](../results/v-off-implementation-audit-20260913/source-hashes.json)
- [CPU 테스트·실행 로그](../logs/v-off-implementation-audit-20260913/)
- [방법론·loss·token 추가 검증](../results/methodology-loss-token-audit-20260913/verification.md)
- [타 서버 baseline과 평가 원출력 검증](../results/baseline-v-off-new-ours1-20260913/verification.md)
- [과거 조사·정정·성능 분석 기록](analysis_v-off_compare.md)

## 추가 조사: Stage 1 메모리 피크 (2026-09-14)

사용자가 보고한 타 서버 약 95GB와 로컬의 더 낮은 메모리 사용량은 아직 동일 측정 기준의 전체 실행 피크로 대조되지 않았다. 확보된 타 서버 자료에는 해당 Stage 1 실행의 rank별 peak allocated/reserved 기록이 없다. 따라서 특정 구현 차이를 원인으로 확정하지 않는다.

사용자는 95GB 관측 실행이 아래 타 서버 run이 맞을 것으로 답했다. 따라서 해당 run을 잠정 비교 대상으로 사용하되, 정확한 run 식별과 95GB 수치는 원본 메모리 로그로 확인된 값과 구분한다. 추가 검색에서도 제공된 archive의 logs/results에서 이 run의 원본 Stage 1 메모리 로그를 찾지 못했다.

- 비교 대상 과거 로컬 실행은 `stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225` / step 215, 타 서버 실행은 `stage1-dora-frozen-b1-lr5e5-zero1-1epoch-exact32768-20260902` / step 431이다. 양쪽 microbatch는 GPU당 1이다. global batch 16/8 및 optimizer step 215/431 차이를 GPU당 activation 메모리가 절반이라는 뜻으로 해석할 수 없다.
- 로컬 과거 실행의 [ownership 기록](../logs/multinode/stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/initial/ownership-rank1.json)은 commit `5e77288a795ebeeacc841b8ef610839c16481add`를 지정한다. 그 commit의 `src/stage1/composition.py`는 DoRA에 `use_reentrant=True`를 설정한다. 현재 재현 코드의 non-reentrant 설정과 다르다. 타 서버 제공 소스 snapshot은 ZeRO-1 DoRA에 non-reentrant를 설정하지만, 그 snapshot이 과거 학습 시작 시점의 코드와 동일하다는 보장은 없다.
- 과거 로컬 로그의 CLI `max_seq_length_train=8192`만으로 8k 절단을 주장하면 안 된다. 해당 commit의 SFT 설정은 32768이고 custom dataset/collator 경로를 사용한다. 현재 collator도 `truncation=False`다. 양쪽 실제 처리 token 길이와 image grid를 대조한 기록은 없다.
- 확보된 양쪽 Stage 1 processor 설정의 `min_pixels=3136`, `max_pixels=12845056`은 같다. 설정상 이미지 최대 해상도를 낮췄다는 설명은 뒷받침되지 않는다.
- 현재 Stage 1 microbatch 1은 모델의 전체 vocabulary logits → FP32 → cross entropy 경로를 사용한다. trainer의 token-chunk CE는 batch가 2 이상인 경로이므로 이번 메모리 차이의 근거로 쓰면 안 된다. 길이 32768, vocabulary 151668인 FP32 logits 한 개의 이론적 크기는 약 18.51GiB다. 이는 텐서 크기 계산이며 실제 입력 길이나 관측 피크가 아니다.
- 조사 시점 새 multi-node 재현 실행은 아직 학습 중이었다. 두 노드 네 GPU의 `nvidia-smi` 순간 사용량은 각각 54757, 54519, 53861, 53951MiB였다. 이 값은 전체 실행의 peak allocated/reserved 측정이 아니다.

원인을 확정하려면 95GB가 관측된 정확한 실행·rank·측정 도구를 연결하고, 같은 입력의 token 길이/image grid 및 teacher forward/student forward/backward 단계별 peak allocated/reserved를 비교해야 한다. 이 조사에서는 실행 중인 학습 코드를 변경하거나 추가 GPU 실험을 시작하지 않았다.

### 추가 증거로 좁힌 원인: 8월 28일 Full/DoRA 보고서는 microbatch 4

후속 요청에 따른 [Stage 1 메모리 상세 비교 보고서](diff-ours1-stage1-memory.md)에 실제 실행 순서, 27개 증감 항목, 공통 대형 텐서, CPU probe와 검증 한계를 정리했다.

사용자가 추가로 지정한 [타 서버 보고서](/raid/workspace/ILVR/ILVR/results/stage1-evaluation/stage1-full-vs-dora-20260828/report.md:23)는 위 9월 2일 batch-1 / checkpoint-431 실행과 다른 실험이다. 보고서 23행의 최대 rank peak allocated는 Full **162.64GiB**, DoRA **143.95GiB**이며, 51–54행은 양쪽 모두 **microbatch 4, accumulation 2**라고 명시한다. 앞선 batch-1끼리의 설명을 이 보고서에 적용하면 잘못이다.

원본 [DoRA evidence](/raid/workspace/ILVR/ILVR/.worktrees/vcsd-ilvr-stage2/outputs/stage1-timed/dora-3epoch/evidence.json:83)에서도 microbatch 4, world size 2, global batch 16, 3 epochs, 648 steps, ZeRO-3 config를 확인했다. rank별 peak allocated는 154567055360 / 153118799872 bytes다. 현재 재현은 microbatch 1, world size 4, accumulation 2, global batch 8, 1 epoch, 431 steps, ZeRO-1이다.

기록된 실행 commit `ca1183e13a0f63018cba210f81640b526bc43a44`의 실제 소스에서 다음 메모리 증가 경로를 확인했다.

1. `src/stage1/data.py`는 batch의 최대 길이로 padding한다. GPU당 네 샘플을 함께 처리하므로 decoder tensor가 `[4, T_max, H]`, logits가 `[4, T_max, V]`가 된다. 현재 microbatch 1은 샘플 간 padding이 없다.
2. `src/stage1/targets.py`는 teacher에도 batch 전체를 넣고 teacher 출력이 살아 있는 동안 helper image vision forward를 수행한다. 이 경로 역시 당시 GPU당 네 샘플의 텐서를 처리한다.
3. `src/stage1/trainer.py`는 labels를 포함한 batch를 모델에 넘긴다. Qwen forward는 전체 logits를 FP32로 바꾸고 `logits[..., :-1, :].contiguous()`로 CE 입력을 만든다. batch 4에서는 이 sequence slice가 비연속이어서 추가 복사된다. batch 1에서는 같은 slice가 연속이므로 이 추가 복사가 필요 없다. CPU tensor probe `[B, 8, 16]`에서 B=1은 storage 공유, B=4는 storage 복사를 직접 확인했다. 이는 해당 연산의 증거이며 GPU 피크 기여량 측정은 아니다.

따라서 이 보고서의 143.95GiB와 현재 약 54GB 관측치 사이에는 **GPU당 동시 샘플 수 4→1, padding, full-logits/CE 임시 텐서 크기 및 복사 차이**라는 구체적인 설명이 있다. 현재 launch/config 및 조사한 Stage 1 코드에 54GB 메모리 상한을 지정한 설정은 없다. checkpointing은 decoder block 내부 activation을 줄이지만 이 전체 LM-head/CE 텐서를 제거하지 않는다. ZeRO 단계와 checkpoint 구현도 달라 정확한 GiB 차이의 요인별 기여량까지 분리한 것은 아니다. 현재 관측치는 `nvidia-smi` 순간 사용량이므로 보고서의 전체 실행 peak allocated와 측정 기준도 구분한다.

이 추가 조사는 새 GPU 학습 없이 원본 evidence, 실행 commit 소스, 작은 CPU tensor 연산으로 수행했다. 앞서 잠정 연결한 9월 2일 batch-1 실행 자체의 95GB 원인은 이 batch-4 보고서만으로 확정하지 않는다.

### 재현 평가 완료: 203/400 (50.75%)

`stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210`은 두 노드·네 GPU에서 global batch 8, optimizer updates 431로 학습을 완료했다. 네 평가 shard의 성공 종료와 전체 index 0–399의 중복·누락 없음, merged `match` 합계 203을 확인했다.

| Stage 1 실행 | 정답 / 전체 | 정확도 |
|---|---:|---:|
| 과거 로컬 global batch 16 / 215 updates | 183/400 | 45.75% |
| 타 서버 재현 대상의 과거 결과 | 201/400 | 50.25% |
| 현재 global batch 8 / 431 updates 재현 | **203/400** | **50.75%** |

현재 결과는 과거 로컬 대비 +5.00%p, 타 서버 과거 결과 대비 +0.50%p(2문항)다. 이번 한 번의 평가에서 타 서버와 유사한 정확도 수준에 도달했다. batch/update 조정 외 dropout context 수정 및 과거 실행·평가 경로 차이도 있으므로 개선 전부를 단일 요인의 인과효과로 분리하거나, 타 서버와 완전히 동일한 학습·예측을 재현했다고 주장하지 않는다. 두 blank-gold 문항은 기존 비교와 같이 전체 400개 분모에 포함했다.

근거: [평가 summary](../results/eval-stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/summary.json), [문항별 결과](../results/eval-stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/merged.jsonl), [학습 완료 witness](../results/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/initial-train-witness.json).

학습 시간은 로컬 machine의 두 process 로그 중 긴 `Stage 1 Trainer time` 기준으로 기존 **2434.031초(40분 34초)**, 현재 **2809.702초(46분 50초)**였다. 현재가 **375.671초(6분 16초), 15.43%** 더 걸렸다. 둘 다 두 노드·네 GPU의 1 epoch 실행이며 accumulation 4→2, updates 215→431로 변경됐다. 이는 `Trainer.train()` 구간으로 모델 로딩·평가와 종료 후 별도 final save 시간은 제외한다. 각 실행의 별도 final save는 약 3초다. 근거: [기존 로그](../logs/multinode/stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/initial/stderr-rank1.log:31), [재현 로그](../logs/multinode/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/initial/stderr-rank1.log:27).

### Stage 2 ON/OFF 학습 시간: 서로 다른 두 비교

현재 repository의 기존 multi-node 실행 rank0 stdout의 `train_runtime`과 producer evidence를 대조했다. 아래는 Stage 1 재현 checkpoint로 새로 학습한 결과가 아니라 기존 Stage 2 실행 기록이다.

| 비교 | OFF | ON | ON−OFF |
|---|---:|---:|---:|
| Preference loss weight 0→1, 423 updates, 4 GPUs, global batch 8 | 12056.8236초 = 3시간 20분 57초 | 12008.5233초 = 3시간 20분 9초 | −48.3초 (−0.40%) |
| V-off boundary suppression false→true, 212 updates, 4 GPUs, global batch 16 | 11050.9244초 = 3시간 4분 11초 | 11024.2464초 = 3시간 3분 44초 | −26.7초 (−0.24%) |

첫 비교는 `ours1-ctx-{ce,pref}-dora-lr5e5-20260907-050655`, 두 번째는 `...default-s212-r2-20260910t022900z`와 `...boundaryoff-...s212-20260911t085315z`다. 작은 시간 차이를 ON의 속도 우위로 일반화하지 않는다.

주의: 첫 비교의 OFF도 실제 CLI에서 `step_preference_enabled=True`이고 `step_preference_weight=0.0`이다. 기록된 commit `003138784b25af7ceb7d0722ab0903aee8dc0ccc`의 `_compute_context_preference_loss`는 weight로 sidecar 실행을 생략하지 않고, visual/textual 점수를 계산한 뒤 `_reduce_context_objective`에서 preference 항에 0을 곱한다. OFF 런타임 로그에도 visual/textual off 점수와 3 logical scoring groups가 남아 있다. 따라서 이 결과는 **preference 계산 자체를 제거한 CE 전용 실행의 속도 비교가 아니다.** 두 번째 비교는 양쪽 모두 preference loss가 활성화된 상태다.

근거: [수집한 rank0 원본 요약 로그](../results/stage2-training-time-comparison-20260914/rank0-training-summaries.log), [시간·step 비교 JSON](../results/stage2-training-time-comparison-20260914/comparison.json), [당시 Stage 2 trainer 소스](../results/stage2-training-time-comparison-20260914/historical-stage2-trainer.py:577).
