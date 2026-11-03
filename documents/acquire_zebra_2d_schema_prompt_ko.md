# Codex 프롬프트 — Zebra-CoT Jigsaw/Search 원본 경로·스키마 확보

현재 **ILVR 작업 루트**에서 설치된 Superpowers와 프로젝트 지침을 따르되, 이번에는 **공식 데이터 경로·필드 정의와 소량의 답안 예시만 확인·저장**하라. 평가 구현, 데이터 전체 다운로드, 모델 실행은 하지 않는다.

## 공식 출처

- 데이터셋: `https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT`
- 메타데이터: `https://huggingface.co/api/datasets/multimodal-reasoning-lab/Zebra-CoT`
- 공식 필드 정의: 데이터셋 `README.md`의 YAML `dataset_info`와 `configs`.
  시작 URL: `https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT/raw/main/README.md`
- `Reference/Bagel-Zebra-CoT/`는 참고 코드 저장소다. 원본 데이터가 그 안에 저장되어 있다고 가정하지 않는다.

| 대상 | 정확한 config | split | 데이터셋 내부 파일 경로 |
|---|---|---|---|
| Jigsaw | `2D Visual Reasoning - Visual Jigsaw` | `train` | `2D Visual Reasoning - Visual Jigsaw/train-*.parquet` |
| Search | `2D Visual Reasoning - Visual Search` | `train` | `2D Visual Reasoning - Visual Search/train-*.parquet` |

두 config의 확인 기준은 `Question: string`, `Final Answer: string`, `problem_image_1: image`다. 실제 공식 메타데이터와 대조하며 대소문자·공백을 보존하라. `string`이라는 선언을 A–D, 정수, 좌표 등 답안 값의 전체 규칙으로 해석하지 않는다.

## 수행할 작업

1. 현재 ILVR 루트에서 작업한다. 기존 `Reference/`와 ILVR 코드를 수정하지 않고 새 worktree·브랜치도 만들지 않는다. 이미 확보한 해당 데이터의 revision이 있으면 유지하고, 없으면 Hub 메타데이터에서 현재 commit SHA를 확인한다. 코드 저장소 SHA를 데이터셋 SHA로 사용하지 않는다.
2. **확인한 데이터셋 SHA**의 README와 위 두 폴더의 파일 목록만 조회한다. config/split, 공식 features, 실제 Parquet 상대 경로와 해당 SHA에 고정된 다운로드 URL, 라이선스를 기록한다. URL의 공백 등은 정상 인코딩하고 Parquet 본문은 다운로드하지 않는다. README 조회 형식은 `/datasets/multimodal-reasoning-lab/Zebra-CoT/raw/{SHA}/README.md`, 다운로드 URL 형식은 `/datasets/multimodal-reasoning-lab/Zebra-CoT/resolve/{SHA}/{파일경로}`다. `{SHA}`를 실제 값 없이 남기거나 추측한 값으로 채우지 않는다.
3. 답안 예시는 공식 미리보기 API `https://datasets-server.huggingface.co/first-rows`에 `dataset=multimodal-reasoning-lab/Zebra-CoT`, 표의 `config`, `split=train`을 전달하여 config별 한 번씩 조회한다. 응답에서 **각 과제 최대 3개**의 `row_idx`, `Question`, `Final Answer`만 원문 그대로 보존한다. 이미지 URL을 따라 다운로드하지 않고 `Text Reasoning Trace`와 `reasoning_image_*`의 내용은 저장하지 않는다. 질문·정답이 `truncated_cells`에 포함된 행은 완전한 예시로 사용하지 않는다. 미리보기의 revision이 입증되지 않으면 `sample_revision=null`, `revision_verified=false`로 기록하고 고정 SHA의 예시라고 주장하지 않는다. API 실패 시 예시는 빈 목록과 실패 이유로 남기고, 대용량 다운로드로 우회하지 않는다.
4. 결과는 **`docs/evaluation/zebra_2d_source_schema.json` 한 파일**에 저장한다. 데이터셋 ID·revision·조회 시각·출처 URL·라이선스와, config별 split·원본 파일 경로/URL·features·입력 필드·정답 필드·제외 필드·텍스트 예시·확인 상태를 포함한다. 기존 파일이 있으면 검증 후 재사용하며 다른 revision이나 사용자 수정은 임의로 덮어쓰지 않는다. 조회 실패나 스키마 불일치는 성공으로 표시하지 않는다.
5. JSON을 다시 읽어 두 config와 필수 필드·dtype·원본 파일 경로가 기록되었는지 검증한다. 저장 경로, 확인한 revision, config별 예시 수와 미확인 사항만 보고하고 멈춘다. 소수 예시를 전체 답안 규칙이나 ILVR 논문의 공식 평가 split으로 일반화하지 않는다.

**범위 밖:** Hugging Face 데이터셋 clone, 전체 shard 다운로드, 이미지 저장, 의존성 설치, 전용 수집 스크립트 작성, 평가 데이터 변환·채점기·judge 구현, 자동 commit/push. 이 작업의 결과는 원본 위치와 공개 스키마의 확보이지 평가 준비 완료가 아니다.
