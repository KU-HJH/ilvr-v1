# 중간 visual latent의 원본 이미지 참조 분석

상태: **design v1 — 서면 승인 대기**. 대화에서 섹션 1–3을 승인받았다 (2026-09-25). 이 문서 승인은 구현 계획(writing-plans) 작성 권한만 부여한다. 구현과 GPU 실행은 계획 승인 뒤에 한다. 모든 상대 경로는 작업 폴더 기준이다.

## 1. 목적과 범위

질문: interleaved reasoning 중간에 생성되는 visual latent (START, Body 8개, END)가 **계산될 때** 원본 입력 이미지에서 정보를 가져오는가?

- 주분석 (A, 메커니즘): latent query가 원본 image key에 두는 attention을 기술적으로 측정한다 (③). 인과 검증으로 attention knockout (①)과 image 교란 activation patching (②)을 포함한다.
- 보조분석 (B, 정보 내용): latent state가 이미지 고유 정보를 담는지 민감도와 식별 가능성으로 측정한다.
- 보조 자유 생성: latent 생성 구간 image mask 하의 정답률 변화 1회.

기존 분석과의 관계:

- 2026-09-10 visual-causal E3는 자유 생성에서 latent 위치의 image 접근을 막았지만 이전 text hidden을 통한 간접 경로를 배제하지 못했고, 다른 CE 모델(Stage 2A s1matched)을 사용했다.
- 2026-09-22 image-block 분석은 image를 **END 이후 text query에서만** 차단했고 정답 궤적을 teacher-force했다. latent 자체의 계산 시점 image 참조는 측정하지 않았다.
- 이번 분석은 latent 계산 시점을 직접 다루며, 직접/간접 경로를 분리한다.

범위 밖: 재학습, 다른 모델 비교 (D), TEST400 전체 확장, 이미지 관련 영역의 정답 라벨 기반 정량 평가.

## 2. 모델·코드·입력

- 작업 폴더: `/root/workspace/ILVR/.worktrees/latent-image-attention-20260925`, branch `feat/latent-image-attention-20260925`, 기준 `feat/ce-block-analysis-20260921` `e28ca6d607eb02f90c0015de0541621cfdbb666b`. 이 branch의 replay 경로는 CE-matched 모델에서 이미 검증됐다. ours1 코드는 이 모델의 학습 코드와 동등하다는 보장이 없어 쓰지 않는다.
- 모델: `checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212` (CE-only, preference off, global batch 16, 212 updates, TEST400 212/400). base: `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`. prepared manifest와 adapter 해시를 기록한다.
- 데이터: `data/TEST.jsonl`, SHA256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`.
- 표본: `/root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json` (SHA256 `fb1f15fe…96aa`)의 `requested_indices` 100개. 재추출하지 않는다. 구성: deletion 30, selection 30, creation 25, update 15, 모두 이미지 1장.
- 궤적: 모델 자신의 greedy 출력. `/root/workspace/ILVR/.worktrees/voff-reproduction-20260917/results/eval-stage2-ce-matched-test400-20260919-155759/merged.jsonl` (SHA256 `76380b6e…96d5`)의 `generated_suffix_token_ids`를 100문항에 대해 사용한다. 이 100문항에서 latent block은 총 391개, block 없는 문항 0개, 정답 49개다. 평가 설정은 기본값 (greedy, max new tokens 4096, max seq 128000, repetition penalty 1, `external_full_latent_block`)이다.
- latent 입력 embedding은 저장되어 있지 않으므로 prefix에서 recurrence로 재계산한다. §6 게이트 G5를 통과하지 못한 문항은 제외 사유와 함께 기록한다. 제외가 10문항을 넘으면 중단하고, latent embedding까지 저장하는 재생성을 별도로 제시한다.
- attention backend: 언어 모델 SDPA (기존 replay 요구사항), vision FlashAttention2.

## 3. 주분석 ③: attention 측정

teacher-forced 1회 forward에서 각 layer attention의 RoPE 적용 후 Q, K와 V를 hook으로 받는다. 필요한 query 행에 대해서만 softmax(QKᵀ/√d + mask)를 직접 계산한다. 대상은 36 layer × 16 head (KV head 2, GQA)다.

query 그룹:

- latent: START, Body 1–8, END를 위치별로 구분한다.
- 대조 text 1: 같은 block 직전 reasoning text token k=8개 (block 사이 text가 8개 미만이면 있는 만큼).
- 대조 text 2: block 직후 reasoning text token k=8개.

key 영역 분할 (모든 key를 빠짐없이 한 영역에 배정):

1. system/template 및 attention sink 위치 (시퀀스 첫 token 등)
2. 원본 image token (`<|image_pad|>` 위치; vision start/end는 1번에 포함)
3. 질문 text
4. 이전 reasoning text
5. 이전 latent block
6. 현재 block (자기 자신 포함)

지표 (query × layer × head 단위):

- image mass: 영역 2의 softmax 합.
- image enrichment: image mass ÷ (영역 2 key 수 ÷ 전체 visible key 수).
- value-weighted mass: Σ_j a_ij‖W_O·v_j‖ 중 영역 2의 비율. sink 보정용이며 판정 주지표다.

집계: layer별 곡선 (latent vs 대조 text 1·2), head별 상위 목록, 문항 단위 cluster bootstrap 95% CI (10,000회, seed 42). 판정: latent의 value-weighted image 비율이 두 대조군보다 CI상 유의하게 높은 layer가 있으면 "latent는 image를 직접 참조한다"는 기술적 증거로 본다. 인과 증거는 §4에서 판단한다.

"image-참조 head 집합" H: 전체 (layer, head) 중 latent Body query의 value-weighted image 비율 평균 상위 5% (29개). ①의 부분 knockout에 사용하며 §3 실행 결과로 고정 기록한 뒤 ①을 실행한다.

정성 자료: 대표 문항 6개 (카테고리별 최소 1개)의 latent별 image patch attention heatmap (merge 후 grid 좌표).

## 4. 인과 검증

### 공통: latent 재계산과 지표

latent 입력 embedding은 직전 위치의 hidden state다. 모든 개입 조건에서 개입 상태로 latent embedding을 순서대로 재계산한다. text token은 저장된 궤적으로 고정한다. 개입은 latent query (START, Body)에만 적용하고 text query는 원상태로 둔다 (단, ②의 교란 run 자체는 전체 forward).

지표:

1. latent 변화: 마지막 layer hidden의 cosine 거리 (Body 위치 평균). clean 반복 실행 차이를 noise floor로 삼아 대비로 보고한다.
2. 직후 text ΔNLL과 token flip 비율 (다음 block 또는 answer 전까지의 text, 09-22 지표와 같은 정의).
3. 후속 block 전파: 이후 block들의 latent 변화.
4. answer ΔNLL: "The final answer is:" 이후 answer token의 ΔNLL.

### ① attention knockout

- K-all: 전 layer에서 latent query → image key 차단.
- K-H: §3의 head 집합 H에서만 차단.
- 대조 (a) no-op: 같은 4D additive mask 형식, 차단 없음. clean과 일치해야 한다 (G3).
- 대조 (b) random: 같은 query에서 image가 아닌 key를 image key 수만큼 무작위 차단 (영역 1과 현재 block 제외, seed 42). mask 자체의 OOD 효과 보정용이다.

### ② image 교란 activation patching

교란 이미지 (원본과 같은 processed grid, 즉 같은 image token 수):

- blur: 원본에 Gaussian blur (σ = 짧은 변의 8%).
- mean: 원본의 채널 평균 단색 이미지.
- swap: 같은 카테고리의 다른 100문항 이미지를 원본의 processed 해상도로 resize. 같은 grid의 짝이 있는 문항은 47개뿐이라 resize로 전 문항을 덮는다. 후보는 seed 42로 결정적으로 고르고 선택 목록을 기록한다.

교란 이미지는 실행 중 결정적으로 생성하며 `data/`에 저장하지 않는다.

원본 run과 교란 run 모두 같은 궤적으로 teacher-force한 뒤, latent 계산 시 layer별 K/V를 다음처럼 조합한다.

| 조건 | image KV | 이전 text·latent KV |
|---|---|---|
| total | 교란 | 교란 |
| direct | 교란 | 원본 |
| indirect | 원본 | 교란 |

image token은 image와 그 앞 prefix만 보므로 image KV는 이미지만의 함수다. 보고: 세 조건의 지표와 direct 비율 = direct 효과 ÷ total 효과. 상호작용 가능성이 있으므로 direct + indirect = total을 가정하지 않는다.

## 5. 보조분석 B와 자유 생성

- 민감도: ②의 total run에서 latent 변화와 대조 text 변화 (같은 정의의 hidden cosine 거리)를 비교한다. latent가 text보다 유의하게 크게 변하면 latent가 이미지 의존 정보를 더 싣는다는 증거로 본다.
- 식별 가능성: 각 문항에서 원본 + 같은 카테고리 swap 후보 4개 (N=5, resize 방식)로 각각 전체 run을 계산한다 (궤적 고정). 후보 i마다 latent 표현 z_i (Body 평균 hidden)와 이미지 표현 r_i (같은 layer에서 image token 위치 평균 hidden)를 얻는다. 문항 안에서 후보 평균을 빼 질문 공통 성분을 제거한 뒤, z_i와 cosine이 가장 큰 r_j가 j=i인 비율을 우연 수준 1/5과 비교한다 (layer 18과 마지막 layer). latent의 이미지 간 차이가 이미지 표현의 차이와 정렬되는지 묻는 지표다.
- 자유 생성: latent 생성 구간 (START 이후 END까지)의 query에서만 원본 image key를 차단한 greedy 생성. 100문항, 기본 평가 설정. clean 대비 exact McNemar. ours1-visual-causal `src/evaluation/visual_interventions.py`의 E3 mask 로직만 가져와 이 branch의 생성 경로에 맞춘다.

## 6. 검증 게이트

본실행 전 GPU smoke 2문항 (latent block 수가 1개인 문항과 3개 이상인 문항)에서 모두 통과해야 한다.

- G1: 직접 계산한 softmax(QKᵀ)·V → W_O 결과가 SDPA attention 출력과 허용 오차 (bf16 기준 상대 1e-2, 기록) 안에서 일치.
- G2: 각 query의 key 영역 분할이 visible key 전체를 정확히 한 번씩 덮고 mass 합이 1 (±1e-3).
- G3: no-op mask 조건이 clean과 hidden·logit 모두 수치 일치.
- G4: 두 KV를 모두 원본 run에서 가져온 patching이 clean과 일치 (재계산 경로 검증).
- G5: 저장 궤적 replay 시 teacher-forced greedy argmax가 저장 token과 일치하는 비율과 첫 불일치 위치를 문항별로 기록한다. 첫 latent block 이전에 불일치가 있거나 전체 일치율이 99% 미만인 문항은 제외한다.

CPU 단위 테스트 (모델 없이): key 영역 분할, mask 구성, KV 조합 index, 교란 이미지 생성의 결정성·grid 보존, bootstrap.

## 7. 산출물과 실행

- 코드: `src/evaluation/latent_image_attention.py` (③), `src/evaluation/latent_image_interventions.py` (①②), `src/evaluation/latent_image_content.py` (B), CLI `scripts/analyze_latent_image.py` (prepare / smoke / main / report 단계). 기존 `vision_block_inputs.py`, `vision_block_runtime.py`를 재사용하고 동작을 바꾸지 않는다.
- 결과: `results/latent-image-attention-<ts>/` — `manifest.json` (입력·해시·설정·commit), `attention/`, `interventions/`, `content/`, `freegen/`, `summary.json`, `report.md`. 모델 artifact를 만들지 않는다.
- 로그: `logs/latent-image-attention-<ts>/`.
- 실행: inference 분석이므로 multinode 학습 정책 대상이 아니다. 로컬 B200 2장에서 문항 단위 shard. 순서는 ③ → (H 고정) → ① → ② → B → 자유 생성.
- 보고: 기술적 증거 (③), 인과 증거 (①②), 정보 내용 (B), 정답률 영향 (자유 생성)을 구분한다. 실행하지 않았거나 게이트를 통과하지 못한 항목은 통과로 표시하지 않는다.
