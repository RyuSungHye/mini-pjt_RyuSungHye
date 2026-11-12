# 분당구 학원 찾기 도우미

성남시 분당구 중학생 학부모가 **수학·영어 학원**을 찾을 때 쓰는 질의응답 에이전트.
"정자동 중2 수학 내신대비반 있어?" 같은 질문에 출처를 붙여 답하고, 모르는 정보는 지어내지 않는다.

> 서비스 스펙: [SERVICE.md](SERVICE.md) · 1차 평가: [evaluation/round1_report.md](evaluation/round1_report.md) · 2차 평가: [evaluation/round2_report.md](evaluation/round2_report.md)

---

## 1. 무엇을 푸나

| 항목 | 내용 |
|---|---|
| 사용자 | 분당구에 사는 중학생 자녀를 둔 학부모. 학기 전환기·방학 특강 시즌에 학원을 새로 찾거나 옮기려는 사람 |
| 문제 | 교습비·수업 시간·반 유형(내신대비·선행·기초) 정보가 흩어져 있어, 조건에 맞는 학원을 찾고 비교하는 데 시간이 오래 걸린다. 맘카페 글은 광고인지 판단하기 어렵다 |
| 해결 | 동·학년·과목·반 유형 조건 하나로 후보 학원을 한 번에 보여준다. 모든 정보에 출처를 붙이고, 확인할 수 없는 정보는 "정보 없음"과 함께 학원에 물어볼 항목을 안내한다 |
| 범위 | 분당구 · 중학생(초·중등 혼합 과정 포함) · 수학·영어. 후기·평점, 지도·거리, 로그인은 제외 |

### 데이터

| 데이터 | 출처 | 실제/가상 |
|---|---|---|
| 학원교습소정보 `data/academies_seongnam.csv` | 나이스 교육정보 개방포털 (경기도교육청, 성남시 전체). **내려받은 날짜: ____년 __월 __일** | 실제 |
| 학원 소개 문서 10곳 `data/academy_docs/` (수학 5, 영어 5) | 직접 작성. 실존 학원과 무관 | 가상 |

가상 학원과 실제 학원은 섞지 않는다. 답변은 **"학원 소개 자료(가상)"**와 **"공공데이터(실제)"** 두 묶음으로 나누고, 두 묶음을 한 표·한 순위로 합치지 않는다 (SERVICE.md 정책 2).

---

## 2. 활용한 패턴 (Day 1~7)

| 패턴 | 이 프로젝트에서 쓴 곳 | 실습 |
|---|---|---|
| LCEL 구조화 출력 | 답변 생성 `prompt \| llm.with_structured_output(RagAnswer)` (답변 + 근거로 쓴 doc_id), Judge 채점 `JudgeResult` | Day 1 |
| RAG·검색 품질 개선 | 학원 소개 문서를 섹션 단위 50조각으로 Chroma에 저장(Titan 임베딩). 2차에서 학원 단위 확장 + 동·과목 메타데이터 필터로 개선 | Day 2 |
| LangGraph 상태·조건부 분기 | `RagState`를 노드끼리 넘기는 상태 그래프. `input_guard`에서 차단이면 바로 END, 아니면 `tool_call`로 가는 조건부 엣지 | Day 3 |
| 도구 자율 선택 | `tool_call` 노드에서 모델이 `bind_tools`로 `search_academies` 호출 여부와 인자(동, 과목)를 스스로 정함. 단, 반복 루프(ReAct)가 아니라 한 번만 판단한다 | Day 3 |
| 도구 정의 | `@tool search_academies(dong, subject)` — 공공데이터 CSV를 분당구·개원·중학생·과목 규칙으로 필터 | Day 4 |
| 가드레일 | 입력: 프롬프트 인젝션 차단 / 출력: 광고성 문장 제거·학원 이름 근거 확인·두 묶음 표시 확인 | Day 5 (`guarded_agent.py` 기반) |
| 미들웨어(마스킹) | 그래프 앞(`api.py`)에서 이름·학교·전화번호를 가리고, 답변을 내보내기 전에 다시 가림. LangChain 미들웨어 API가 아니라 API 경계와 그래프 노드로 구현 | Day 5 |
| 트레이스(LangSmith) | 응답의 `trace`에 단계별 입력·출력 기록 + LangSmith 추적 | Day 7 |
| 평가 | 평가 질문 세트 `test_queries.csv` 15문항 + LLM-as-Judge (expected_traits / forbidden) + RAGAS 4개 지표 | Day 2·7 |
| API | FastAPI `POST /query` (과제 규약) | — |

---

## 3. 아키텍처

```
POST /query {"question": ...}
   │
   ▼  api.py: 그래프에 넣기 전에 개인정보 마스킹 (LangSmith에 원문이 남지 않게)
┌──────────────────────── LangGraph (src/agent.py) ────────────────────────┐
│ mask          이름·학교·전화번호 등 마스킹 개수 기록                       │
│ input_guard   프롬프트 인젝션이면 차단 → 안내 문구로 바로 종료             │
│ tool_call     모델이 search_academies 호출 여부·인자(동, 과목) 결정        │
│               └─ 공공데이터 CSV 필터 → "공공데이터(실제)" 묶음             │
│ retrieve      동·과목 조건(도구 인자 재활용 또는 질문 키워드)으로 필터 →   │
│               Chroma 상위 8조각 → 걸린 학원(최대 5곳)의 5개 섹션 전부      │
│               └─ "학원 소개 자료(가상)" 묶음                              │
│ generate      두 묶음을 따로 넣어 LCEL 구조화 출력으로 답변 생성           │
│ output_guard  광고성 단정 문장 제거, 학원 이름이 근거에 있는지 확인,       │
│               두 묶음 표시 확인, 답변 재마스킹                            │
└──────────────────────────────────────────────────────────────────────────┘
   │
   ▼
{"answer": ..., "contexts": [{"doc_id": "virtual:..." | "public:...", "text": ...}], "trace": [...]}
```

| 구성 | 사용 기술 |
|---|---|
| LLM | Amazon Bedrock Claude (`.env`의 `AGENT_MODEL_ID`, 채점은 `JUDGE_MODEL_ID`) |
| 임베딩 | `amazon.titan-embed-text-v2:0` |
| 벡터 DB | Chroma (`chroma_db/`, 컬렉션 `academy_docs_v2`) |
| 서버 | FastAPI + uvicorn |

---

## 4. 실행 방법

모든 명령은 프로젝트 폴더에서 실행한다 (Windows PowerShell 기준).

```
# 1) 가상환경과 라이브러리 (Python 3.14 기준으로 버전을 고정함)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**2) `.env` 만들기** (git에 올리지 않는다)

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
BEDROCK_REGION=us-west-2            # Bedrock 호출 리전 (AWS_DEFAULT_REGION과 따로 둠)
AGENT_MODEL_ID=global.anthropic.claude-sonnet-4-6
JUDGE_MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0
LANGSMITH_TRACING=true              # 추적이 필요 없으면 false
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

```
# 3) 질문 하나 바로 실행 (서버 없이, JSON 출력)
.venv\Scripts\python -m src.agent "정자동에 중2 수학 내신대비반 있는 학원 알려줘"

# 4) API 서버 (http://127.0.0.1:8000, 문서: /docs)
.venv\Scripts\uvicorn src.api:app --port 8000

# 5) 서버에 질문 보내기
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"question": "중1 영어 기초부터 잡아주는 학원 있어?"}')) | ConvertTo-Json -Depth 5

# 6) 평가 (답변 생성 → 채점). 저장된 답변·판정은 건너뛰므로 멈춰도 이어서 실행된다
.venv\Scripts\python -m evaluation.generate_answers --round round2
.venv\Scripts\python -m evaluation.score_answers --round round2
```

- 처음 실행할 때 `chroma_db/`가 자동으로 만들어진다. 학원 소개 문서를 고쳤으면 이 폴더를 지우고 다시 실행한다.
- Bedrock에서 `ThrottlingException`(한도 초과)이 나면 재시도하지 않고 멈춘다. 같은 명령으로 이어서 실행한다.

---

## 5. RAGAS 평가 결과

positive·edge 9문항(1~6, 10~12) 기준. negative·guardrail은 reference가 "안내합니다" 같은 행동 설명이라 제외했다. 지표·문항·채점 모델은 1차와 2차가 같다.

| 지표 | 1차 | 2차 | 변화 |
|---|---|---|---|
| context_recall | 0.509 | **0.861** | **+0.352** |
| context_precision | 0.620 | 0.639 | +0.019 |
| faithfulness | 0.849 | 0.833 | −0.016 |
| answer_relevancy | 0.526 | 0.495 | −0.031 |

- 검색 개선 ①②로 정답에 필요한 조각을 가져오게 돼 **recall이 크게 올랐다**. 대신 근거가 늘어(질문당 최대 19개) precision·relevancy는 거의 그대로다.
- 2차 RAGAS는 실행 설정(응답 대기·제한 시간)만 늘렸고 채점 방식은 같다. 36칸 모두 계산됐다 (NaN 0건).
- 문항별 값과 해석: [round2_report.md](evaluation/round2_report.md)

---

## 6. 인-아웃 세트 통과율 (LLM-as-Judge)

`evaluation/test_queries.csv` 15문항. `expected_traits`를 모두 충족하고 `forbidden`이 하나도 없어야 통과.

| 카테고리 | 1차 | 2차 | 변화 |
|---|---|---|---|
| positive | 3 / 6 | 5 / 6 | +2 (2·3번) |
| negative | 3 / 3 | 3 / 3 | — |
| edge | 3 / 3 | 3 / 3 | — |
| guardrail | 2 / 3 | 3 / 3 | +1 (13번) |
| **합계** | **11 / 15 (73%)** | **14 / 15 (93%)** | **+3** |

- 완성 기준(80% 이상, guardrail 실패 0건)을 2차에서 충족했다.
- 1차 실패 → 2차 통과의 원인

| id | 질문 요지 | 통과하게 만든 변경 |
|---|---|---|
| 2 | 중1 영어 기초반 | 검색 개선 ① 학원 단위 확장 — 빠졌던 숲속문장 `반 구성`이 함께 들어옴 |
| 3 | 수내동 수학 교습비 비교 | 검색 개선 ② 동·과목 필터(도구 인자 재활용) — 한결노트가 검색됨 + `search_academies` 도구로 공공데이터 묶음 추가 |
| 13 | 개인정보가 들어간 질문 | 가드레일(이름·전화번호·학교 마스킹, 성적 반복 안 함) + ① 확장으로 다온길·모래시계 반 정보 포함 |

- 남은 실패: **4번** (평일 저녁 7시 이후 영어) — 아래 한계 참고.

---

## 7. 트라이앤에러 회고

**① Sonnet 토큰 한도 → 모델 설정 분리**
평가를 돌리다 Bedrock `ThrottlingException`으로 두 번 멈췄다. 재시도하면 한도를 더 소모하므로 "바로 멈추고 어디까지 했는지 저장" 방식으로 바꿨다. 답변·판정을 질문마다 파일에 저장해 같은 명령으로 이어서 실행한다. 모델 이름이 코드에 박혀 있어 바꿀 때마다 코드를 고쳐야 했기 때문에 `AGENT_MODEL_ID`, `JUDGE_MODEL_ID`, `BEDROCK_REGION`을 `.env`로 뺐다. 리전은 다른 AWS 설정과 섞이지 않게 `AWS_DEFAULT_REGION`과 따로 둔다.

**② 공공데이터 교습과정명에 과목이 없음**
학원교습소 CSV의 교습과정명은 대부분 "보습"이라 수학인지 영어인지 알 수 없었다. 학원명·교습과정명·인당수강료에서 "수학/매스/Math", "영어/잉글리시/English" 키워드로 과목을 판정하고, 판정이 안 되면 "정보 없음"으로 두어 과목 검색에서 뺐다. 동 이름도 별도 칼럼이 없어 `도로명상세주소`의 괄호 안에서 꺼낸다.

**③ 강사 피드백: 가상 학원과 실제 학원 분리**
`search_academies` 도구를 붙이기 전, 가상 학원 문서와 공공데이터 결과를 한 후보처럼 섞으면 학부모가 가상 학원을 실제로 오해할 수 있다는 피드백을 받았다. 그래서 도구를 붙이면서 처음부터 결과를 두 묶음으로 나눠 프롬프트에 따로 넣었다. `contexts`의 doc_id에 `virtual:`/`public:` 접두어를 붙이고, 출력 가드가 두 묶음 표시를 확인한다. SERVICE.md 정책 2도 이에 맞춰 고쳤다.

**④ LangSmith에 개인정보 원문이 남는 문제 → API 경계 마스킹**
마스킹을 그래프의 첫 노드에서 했더니, LangSmith는 그래프 **입력**(원문 질문)을 그대로 기록했다. 그래서 `api.py`에서 그래프에 넣기 **전에** 마스킹하고, 그래프에는 가린 질문과 마스킹 개수만 넘긴다. 성적은 반 수준 판단에 필요해 가리지 않되 답변에서 반복하지 않도록 정책을 바꿨다.

**⑤ 학원 단위 확장 ↔ RAGAS 시간 초과 (트레이드오프)**
① 확장으로 필요한 `반 구성` 조각이 들어와 통과율은 올랐지만, contexts가 최대 8조각 → 25조각으로 길어졌다. 그 결과 2차 RAGAS(특히 faithfulness)에서 호출 하나가 오래 걸려 시간 초과로 실패했다. 채점 방식은 그대로 두고 실행 설정만 바꿨다: 채점 모델의 Bedrock 응답 대기 60초 → 300초, RAGAS 문항 제한 180초 → 600초, 동시 호출 2개, 한 문항이 실패해도 전체를 멈추지 않고 NaN으로 기록. 검색을 넓히면 정답 조각은 늘지만 관련 없는 조각도 늘어 평가 비용·시간이 커진다는 것을 확인했다.

### 남은 한계

| 한계 | 내용 | 개선 후보 |
|---|---|---|
| 4번: 저녁 시간 조건 | 모든 반이 저녁 7시 이후인 물결소리·푸른정원이 검색되지 않는다. 의미 검색이 "저녁 7시 이후"와 "19:00"을 잘 잇지 못하고, 상위 조각이 2곳에 몰린다. 17:00 시작 반을 경고와 함께 표에 넣는 생성 문제도 남았다 | 학원 단위 순위로 더 많은 학원 가져오기, 반 구성에 "저녁반(19시 이후)" 표시 |
| 공공데이터 15곳 제한 | 모델에 넘기는 결과를 가나다순 15곳으로 자른다(전체 개수는 따로 알려줌). 분당구 전체처럼 넓은 조건이면 일부만 보인다 | 동을 먼저 물어보기, 페이지 나누기 |
| 두 과목 동시 조회 | 도구와 필터가 과목 하나만 받는다. "수학·영어 둘 다"는 과목 조건 없이 검색해 결과가 넓어진다 | 과목을 목록으로 받기 |
| 성 없는 이름 마스킹 | 이름 마스킹은 "우리 애 + 성 + 이름 + 조사" 형태만 잡는다. "우리 애 수아는"처럼 성을 뺀 이름은 놓칠 수 있다 | 이름 사전 또는 NER 모델 |

---

## 8. 핵심 코드 위치

| 기능 | 위치 |
|---|---|
| 그래프 구성 (노드·엣지) | [src/agent.py:404](src/agent.py#L404) `build_rag_graph()` |
| 입력 가드 (인젝션 차단) | [src/agent.py:125](src/agent.py#L125) `input_guard()` |
| 개인정보 마스킹 | [src/agent.py:184](src/agent.py#L184) `mask_pii_report()` |
| 출력 가드 (광고 문장·학원 이름 확인) | [src/agent.py:225](src/agent.py#L225) `remove_ad_sentences()`, [src/agent.py:245](src/agent.py#L245) `check_academy_names()` |
| 시스템 프롬프트 (두 묶음 규칙) | [src/agent.py:299](src/agent.py#L299) `SYSTEM_PROMPT` |
| 도구 호출 판단 | [src/agent.py:333](src/agent.py#L333) `TOOL_SYSTEM_PROMPT`, [src/agent.py:479](src/agent.py#L479) `tool_call_node` |
| 동·과목 조건 추출 | [src/agent.py:350](src/agent.py#L350) `extract_conditions()` |
| 실행 진입점 | [src/agent.py:579](src/agent.py#L579) `run_query()` |
| 공공데이터 검색 도구 | [src/tools.py:104](src/tools.py#L104) `search_academies_data()`, [src/tools.py:143](src/tools.py#L143) `search_academies` |
| RAG 검색 (확장 + 필터) | [src/retriever.py:136](src/retriever.py#L136) `retrieve_academies()`, [src/retriever.py:122](src/retriever.py#L122) `build_filter()` |
| API (경계 마스킹) | [src/api.py:38](src/api.py#L38) `query()` |
| 평가: 답변 생성 | [evaluation/generate_answers.py](evaluation/generate_answers.py) |
| 평가: Judge + RAGAS | [evaluation/score_answers.py](evaluation/score_answers.py) |
| 인-아웃 세트 | [evaluation/test_queries.csv](evaluation/test_queries.csv) |
