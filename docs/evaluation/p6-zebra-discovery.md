# P6 Zebra-CoT 2D 조사 — 2026-09-13

상태: **2026-09-14 원본 400개·이미지 400개 확보 및 로컬 전수 검사 완료. 상세 설계 commitment v1 승인·계획 작성 완료, P6 구현 진행 중.**

understood as: 완료한 P5 EMMA 다음으로 P6 Zebra-CoT Jigsaw/Search 오프라인 평가를 진행한다. 기존 공통 평가 경로와 고정 원본을 재사용하고, 실제 정답 형식과 평가 표본을 확인한 뒤 설계를 확정한다. `계속 진행하자`를 아직 제시하지 않은 P6 상세 설계 승인으로 해석하지 않는다.

## 원본과 표본의 확인 범위

- 소스 비교 전에 `references.lock.json`의 독립 저장소 7개 root/origin/commit/clean 상태와 잠근 파일 33개 hash를 확인했다. [검증 결과](../../results/evaluation/p6-discovery-20260913-01/reference-verification.json).
- Bagel-Zebra-CoT 고정 checkout `bbf53c327889a1764d27989d117d1a6b73ffdd3f`에서 전용 Jigsaw/Search scorer를 찾지 못했다. 이는 [기존 audit](audit.md)의 범위를 유지하며, 모든 공개 구현에 대한 부재 주장으로 확대하지 않는다.
- 기존 [스키마](../../data/evaluation/zebra_2d_source_schema.json)의 dataset은 `multimodal-reasoning-lab/Zebra-CoT`, revision은 `f6d1defc169d19180a69f517925ed6cbb06a1c97`이다. 원본 split 명칭은 두 과제 모두 `train`이다.
- [ILVR 논문 §4.1](https://arxiv.org/html/2512.05665v1)은 Jigsaw/Search 각 200문항과 자유응답 평가용 Qwen2.5-VL 72B judge를 명시한다. 논문의 정확한 400개 ID, 추출 규칙, judge 프롬프트는 이번 확인에서 확보하지 못했다.
- 로컬 파일명 검색과 기존 audit에서 Zebra 평가 manifest를 찾지 못했다. `data/TEST.jsonl`은 기존 CoMT 평가 대상이다. 파일명 검색만으로 모든 파일 내용에 Zebra 표본이 없다고 단정하지 않는다.

| 과제 | README 선언 행 수 | parquet 수 | 전체 parquet bytes | 첫 shard 실제 행 수 |
|---|---:|---:|---:|---:|
| Jigsaw | 21,899 | 26 | 12,050,671,761 | 843 |
| Search | 30,000 | 27 | 12,844,156,433 | 1,112 |

공개 두 과제 전체는 선언상 51,899개, 약 24.9 GB다. 전체를 내려받거나 전수 검증한 결과가 아니다.

## 물리 스키마 조사

고정 revision의 README를 내려받아 기존 SHA-256 `be594e52ab48fac3f7865d49efd50592b81cd5318e5b9e41441e6972c44c7ca1`과 일치함을 확인했다. 먼저 각 과제 첫 parquet의 footer만 HTTP Range로 읽었다. 응답의 206 상태, 정확한 Content-Range와 길이를 검사하고 PyArrow로 footer를 해석했다. 이 단계에서 이미지나 reasoning 본문은 다운로드하지 않았다.

자료는 모두 [data/evaluation/p6-source-discovery-20260913-01/](../../data/evaluation/p6-source-discovery-20260913-01/)에 있다. `first-shard-metadata.json`은 두 파일에 한정된 물리 스키마·행 수·텍스트 열 위치 관찰이다. 원본 parquet 전체 파일이나 준비된 평가 manifest가 아니며, 전체 53개 파일의 스키마 일치를 증명하지 않는다.

두 첫 shard 모두 `Question`, `Text Reasoning Trace`, `Final Answer`는 nullable string이다. `problem_image_1`은 bytes/path 구조다. Jigsaw는 reasoning image 1–4, Search는 reasoning image 1을 포함한다. 평가 입력으로 사용할 후보는 `Question`과 `problem_image_1`이며, 정답과 reasoning 열은 입력에서 제외해야 한다.

## 답안 형식과 남은 결정

기존 각 3개 preview에서 Jigsaw 답은 선택지 문자이고 Search 답은 사물명·날짜 같은 자유 텍스트다. preview는 고정 revision과 일치 여부가 미확인이다. 이 여섯 사례를 전체 형식의 근거로 사용할 수 없다.

이후 고정 revision의 두 첫 shard에서 PyArrow/Fsspec으로 `Question`, `Final Answer` 열만 읽어 원본 텍스트를 조사했다. reasoning·이미지 열은 해석하거나 평가 입력으로 추출하지 않았다. 원본 텍스트는 위 데이터 디렉터리의 `jigsaw-first-shard-text.jsonl`, `search-first-shard-text.jsonl`이며, 각 파일의 `shard_row`는 원본 첫 shard의 0-based 위치다. 이는 준비 manifest 또는 최종 평가 표본 선정이 아니다.

- Jigsaw 843개: A 196개, B 221개, C 198개, D 228개. 빈 답과 null 답이 없다.
- Search 1,112개: 서로 다른 답 769개, 빈 답과 null 답이 없다. 숫자를 포함한 답 263개, 슬래시 포함 답 10개, 최대 길이 141자다. 날짜(`12/02`, `3/1/01`), 백분율(`8%`), 금액(`$0.9M`), 단어와 문장이 관찰된다. 숫자 포함 여부는 답안 유형 분류가 아니다.
- 두 텍스트 집합의 모든 Question에 `[problem_image_1]` 표시가 있다. 이미지 bytes의 존재·정합성·선택지 내용은 아직 검증하지 않았다.
- [첫 shard 텍스트 요약](../../data/evaluation/p6-source-discovery-20260913-01/first-shard-text-summary.json)은 관찰된 두 파일에만 적용된다. 전체 공개 데이터 또는 향후 선정할 400개가 같은 계약을 만족한다고 일반화하지 않는다.

초기 진행 설명의 순열·좌표 가정은 원본 근거가 없었으므로 철회했다. Search의 `12/02` 같은 날짜에 EMMA의 분수 비교를 적용하거나, Jigsaw의 선택지 수를 gold 문자에서 추론하면 안 된다. 추출기는 정답을 보지 않아야 하며, 의미 동등성을 검증하지 않은 규칙 점수를 논문의 judge 점수와 동일시하지 않는다.

표본 구성은 다음 세 방향을 제시했다.

1. 고정 원본에서 정답과 무관하게 Jigsaw/Search 각 200개를 선정한 구현 검증용 표본 — 권장. 논문 동일 표본과 OOD는 미확인으로 표시한다.
2. 공개 두 과제 전체 51,899개 — 검증 범위가 넓지만 수집·이미지 저장 비용이 크다.
3. 사용자가 제공하는 논문 평가 ID — 정확한 목록과 원본 연결이 가능하면 이를 사용한다.

선정 규칙, 실제 선정 대상의 정답 형식 전수 조사, 채점 규칙과 제한, 소스 입력 누출 방지, 두 과제별 집계는 후속 설계에서 고정해야 한다. 아직 scorer, adapter, config, 평가 manifest를 만들지 않았다. 데이터의 `ood_verified=unknown`, `paper_protocol_verified=false` 상태를 유지한다.

## 400개 원본 확보 승인과 선정 — 2026-09-14

사용자가 `고정한 200개씩, 총 400개로 검증 (권장)`을 명시적으로 선택했다. 이는 원본 확보·표본 고정의 승인이고, 아직 제시하지 않은 세부 scorer 설계의 승인은 아니다.

전체 원본 53개 shard의 footer에서 총 51,899행을 확인했다. `ilvr-zebra-2d-source400-v1`, 고정 revision, config, 0-based 전역 행 번호를 LF로 연결한 UTF-8 bytes의 SHA-256을 오름차순 정렬하여 각 과제의 앞 200개를 선정했다. 전역 순서는 고정 parquet 경로의 사전순과 각 파일 내부 행 순서다. 답·질문·이미지·모델 출력을 선정 기준으로 사용하지 않는다. 문제가 발견돼도 임의로 다른 표본으로 대체하지 않는다.

수집 전 고정한 [selected-ids.json](../../data/evaluation/zebra-source400-20260914-02/selected-ids.json)의 SHA-256은 `92e776e6b7e7e646c7f3b23c05c87b0993af44821faecac789a0e29ef2737927`이다. 400개는 53개 shard 모두에 걸쳐 있다. 자세한 규칙·원본 행 수는 같은 데이터 디렉터리의 `selection-rule.json`, `source-inventory.json`에 있다.

수집은 승인된 조사용 일회성 스크립트이며 운영용 adapter/preparer가 아니다. 선택한 행의 `Question`, `Final Answer`, `problem_image_1`을 원본 투영 자료로 저장한다. 압축 row group 단위 접근 때문에 네트워크에서 읽는 문제 이미지 열의 범위는 400개보다 넓다. 평가용으로 보존할 이미지는 선택된 400개뿐이며 reasoning 열은 읽어 해석하거나 보존하지 않는다.

최초 shell background 실행은 PID만 반환한 뒤 로그·데이터를 생성하지 못하고 종료됐다. 정확한 종료 신호는 관찰하지 못했다. [실패 기록](../../logs/evaluation/p6-source400-20260914-01/launch-failure.json)을 보존하고, 이전 조사에서 동작했던 managed exec 세션으로 fresh run ID `p6-source400-20260914-02`를 사용했다. 소스 읽기/선정 로직은 바꾸지 않았다.

## 채점 범위 제안 — 검토 중

세 접근을 제시했다: 보수적 규칙 비교(권장), 정규화 없는 strict exact match, 동의어·의미 비교 포함. 권장안은 Jigsaw 선택지 문자 비교, Search의 대소문자·공백·문장 끝 마침표 정규화 후 전체 답 비교다. 날짜/숫자 변환·부분 문자열·동의어 비교를 하지 않는다. 이는 자체 modified profile이며 논문의 LLM judge와 동등하다고 주장하지 않는다.

예컨대 Search의 `knife`와 `Knife.`는 일치시키되 `A knife.`와 `knife`, `1/2`와 `0.5`는 일치시키지 않는 제안이다. 실제로 의미가 같은 응답을 틀리게 판정할 수 있다는 한계가 있다. 반대로 strict exact match는 단순 표기 차이도 실패하며, 의미 비교 확장은 별도 정답성 근거와 평가 프로토콜 결정이 필요하다.

기존 공통 prepare/score CLI, 누락·인프라 오류 처리와 Jigsaw/Search별 집계를 재사용하는 방향이다. 이미지 안에 있는 선택지를 정답에서 추측한 텍스트로 만들지 않는다. 추출기의 marker/boxed 문법, Jigsaw 허용 라벨, 데이터 오류 정책과 CPU 검증 범위는 실제 400개 조사 결과와 함께 후속 상세 설계에서 제시한다.

## 선택된 400개 조사 결과

수집 프로세스와 별도 로컬 검증 명령 모두 exit 0이다. [원본 투영 자료](../../data/evaluation/zebra-source400-20260914-02/source-records.jsonl), [수집 요약](../../data/evaluation/zebra-source400-20260914-02/acquisition-summary.json), [검증 결과](../../results/evaluation/p6-source400-20260914-02/source-validation.json), [검증 로그](../../logs/evaluation/p6-source400-20260914-02/validation.log)를 남겼다.

- 과제별 200개, 총 400개. 선택 규칙을 `heapq.nsmallest`로 별도 재계산해 저장된 ID 집합과 일치를 확인했다. shard 행 위치와 전역 행 번호의 대응도 확인했다.
- 문제 이미지 400개 모두 JPEG, 서로 다른 byte hash 400개, 총 125,791,653 bytes. 저장된 전체 이미지의 해시·크기·디코딩을 확인했다.
- 400개 Question 모두 문제 이미지 1개 표시가 있고 reasoning image나 다른 image slot 표시가 없다. 원본 투영 record는 선정 메타데이터와 Question/Final Answer/problem_image_1만 포함한다. 실제 추론 입력의 누출 방지 검증은 adapter 구현 뒤 별도로 해야 한다.
- Jigsaw의 정답은 A 62개, B 40개, C 45개, D 53개다. 정답별 균형을 맞추기 위해 표본을 교체하지 않았다.
- Search는 서로 다른 정답 167개다. 숫자를 포함하는 답 47개, 슬래시 포함 답 1개, 최대 길이 178자다. 날짜 `9/25/87`, 금액 `$ 11 billion`, 단어·문장형 답이 관찰된다.
- 원본 투영 JSONL SHA-256은 `1a533ce69f426cce9397251df976cf4331118f530b342713747f2ebdae3d4f25`다. 이는 확보한 투영 자료의 고정 identity이며, 53개 원본 parquet 전체 bytes의 hash 검증은 아니다.

Search 전역 행 2172는 tandem bicycle의 색을 묻고 원본 답은 `The bicycle is red.`다. 제안한 보수적 전체 문자열 비교를 구현하면 `red`는 불일치 판정된다. 이 한계는 실제 원본에 근거한 정책상 예상이며 아직 없는 scorer를 실행한 결과가 아니다. 따라서 규칙 기반 구현 검증 지표와 의미 정답률을 구분해야 하며, 비교 범위 확인 질문에 이 사례를 함께 제시한다.

이번 완료 범위는 원본 확보와 데이터 검사다. scorer CPU 계약 테스트, adapter로 만든 운영 manifest, GPU 추론, 전체 모델 평가, 원본 judge 환경 비교, 논문 프로토콜 검증은 수행하지 않았다. P6 전체 완료나 모델 성능 재현으로 표시하지 않는다.

## 비교 범위 승인과 세부 설계 — 2026-09-14

사용자가 `보수적인 규칙 기반 비교 (권장)`을 선택했다. 앞서 제시한 문장형 gold와 짧은 의미 정답의 불일치 한계를 유지한다. 승인된 표본/비교 범위를 다시 묻지 않고, 추출 문법·입력/target 경계·준비/보고서 계약·검증 계획을 [design commitment v1](../superpowers/specs/2026-09-14-ilvr-zebra-2d-offline-design.md)에 구체화했다.

동일한 frozen proposal에 hate/prism을 적용했고, macrothink는 범위가 명시적으로 결정된 상황에서 추가 fresh-read의 효용이 작아 생략했다. 명세의 검증 주장에는 mandela를 적용했다. 문서 전체 복사본에 대한 별도 shower 독해의 세 가지 모호함을 수정했다. 비평 결과·first_nail 미실행 이유·이미 승인된 한계·남은 확인은 명세 8절에 있다. [독해 기록](../../results/evaluation/p6-design-20260914-01/cold-read.json)과 [현재 Reference 확인](../../results/evaluation/p6-design-20260914-01/reference-verification.json)을 남겼다.

세부 설계의 최종 인간 승인 뒤 writing-plans로 파일별 구현/검증 명령을 고정한다. 이번 단계에서 scorer/adapter/config/manifest를 구현하지 않았고 기존 P1–P5 코드·worktree·데이터 symlink를 유지했다.

후속 실행 승인: 사용자가 최종 설계를 보고 `진행하자`로 승인했다. [상세 계획](../superpowers/plans/2026-09-14-ilvr-zebra-2d-offline-plan.md)을 작성하고 기존 worktree에서 SDD로 구현을 시작했다. 새 scorer/공통 계약 연결, 고정 원본 준비/전수 대조의 두 작업으로 진행한다. 기존 schema/report CPU baseline 12개가 통과했다. 완료 여부는 작업별 실제 증거와 최종 검증 후 갱신한다.

## P6 오프라인 구현 완료 — 2026-09-14

승인한 `zebra_2d_rules_v1` scorer와 공통 reader/report/prepare CLI 연결을
완료했다. 기본 운영 산출물은 `data/evaluation/zebra-2d400-v1/`이며 Jigsaw와
Search 각각 200개, 총 400개와 이미지 400개, 제외 0개다. 최종 수정 뒤 실제
이미지를 pixel decode하여 별도 준비한 manifest도 같은 hash를 얻었다.
Zebra 집중 83개 및 최종 P1–P6 scoped CPU 633개가 통과했고, 원본 400개와
준비 산출물의 ID·원문·target·이미지 bytes/hash 대응을 확인했다.
[검증 기록](validation.md)과 [프로토콜·사용법](protocols.md#zebra-cot-2d-고정-400개-오프라인-프로토콜)에
완료 범위와 명령을 남겼다. 앞 절들의 미실행 표시는 당시 조사/설계 단계의
기록이다. 현재도 실제 모델 평가, 원본 judge 환경, GPU, paper sample identity와
OOD 검증은 수행하지 않았다. P6 오프라인 완료를 논문 성능 재현으로 확대하지 않는다.
