# CLAUDE.md

## 프로젝트 요약

분당구 중학생 학부모가 수학·영어 학원을 찾을 때 쓰는 질의응답 에이전트.
서비스 내용, 범위, 정책은 `SERVICE.md`가 기준이다. 충돌하면 SERVICE.md와 아래 과제 규약을 따른다.

이 프로젝트는 AI 에이전트 과정의 미니 프로젝트(Day 8~10)다.
마감이 짧으므로 기능을 넓히기보다 작게 완성하고 하나씩 붙인다.

## 작업 방식 (가장 중요)

- 사용자는 파이썬 초보이고 Java 경험만 있다. 설명은 쉽게, 짧게 한다.
- 한 번에 한 단계만 구현한다. 요청받지 않은 기능, 파일, 라이브러리를 추가하지 않는다.
- 구현이 끝나면 직접 실행해서 결과를 보여주고, 사용자가 따라 할 실행 명령어를 알려준다.
- 에러가 나면 원인을 한두 문장으로 설명하고 고친다.
- 큰 구조를 바꾸거나 파일을 지워야 할 때는 먼저 물어본다.

## 과제 규약 (반드시 지킬 것)

### 폴더 구조

```
mini-pjt_{이름}/
├── src/
│   ├── agent.py          # 메인 에이전트 그래프 (LangGraph)
│   ├── tools.py          # 도메인 도구
│   ├── retriever.py      # RAG 파이프라인
│   └── api.py            # POST /query
├── data/
│   ├── academies_seongnam.csv   # 나이스에서 내려받은 학원교습소정보 (성남시 전체)
│   └── academy_docs/            # 가상 학원 소개 문서 10곳
├── evaluation/
│   ├── test_queries.csv  # 인-아웃 세트 (자체 평가·최종 채점의 기준)
│   ├── round1_report.md  # 1차 자체 평가 결과
│   └── round2_report.md  # 2차 자체 평가 결과 (개선 후)
├── SERVICE.md            # 서비스 스펙
├── CLAUDE.md
├── README.md             # 과제 README 템플릿 준수
└── requirements.txt
```

- Dockerfile, run.sh는 선택 사항이다. 사용자가 요청할 때만 만든다.

### API 스펙

```
POST /query
Body: {"question": "사용자 질의"}

Response:
{
  "answer": "근거 기반 응답",
  "contexts": [{"doc_id": "...", "text": "..."}],
  "trace": [{"step": "retrieve", "input": "...", "output": "..."}]
}
```

- `contexts`에는 답변 근거로 쓴 문서 조각과 도구 결과를 모두 넣는다 (RAGAS 평가에 쓰임).
- `trace`에는 검색, 도구 호출, 가드레일 처리 단계를 순서대로 기록한다.

### test_queries.csv 스키마

칼럼: `id, category, input, expected_traits, forbidden, expected_tools, note`
- category는 positive / negative / edge / guardrail 중 하나
- 필드 안의 여러 항목은 세미콜론(;)으로 구분
- RAGAS context_recall 계산을 위해 `reference`(정답 답변) 칼럼을 추가할 수 있다

### 필수 패턴

LCEL 구조화 출력, RAG, 트레이스, 평가(RAGAS, LLM-as-Judge)는 반드시 들어가야 한다.

## 기술 스택

- Python, LangChain, LangGraph
- 시작 코드: Day 5 실습 SOLUTION `guarded_agent.py`를 `src/agent.py`로 이름을 바꿔 복사한 것이다. 이 파일을 수정해서 쓴다
- LLM과 임베딩 설정은 Day 1~7 실습에서 쓰던 방식을 그대로 따른다
- API 서버: FastAPI
- LLM 키는 `.env`에 두고, `.env`는 git에 올리지 않는다 (`.gitignore`에 추가)

## 데이터 규칙

- 학원교습소정보: `data/academies_seongnam.csv` (나이스 교육정보 개방포털에서 내려받은 실제 데이터). API는 쓰지 않는다
  - 한글이 깨지면 인코딩을 cp949 또는 utf-8-sig로 바꿔 읽는다
  - 이 파일은 수정하지 않는다. 필터링은 코드에서 한다
- `search_academies` 검색 규칙은 SERVICE.md 3장을 따른다. 핵심:
  - 분당구: `도로명주소`에 "분당구" 포함 / 동: `도로명상세주소` 괄호 안
  - 운영 중: `등록상태명` = "개원"
  - 중학생: `인당수강료`·`교습과정목록명`·`교습과정명`에 "중등"/"중학"
  - 과목: 교습과정명은 대부분 "보습"이므로 `학원명`·`교습과정명`·`인당수강료`의 키워드로 판정. 판정 불가면 "정보 없음"
- 학원 소개 문서(`data/academy_docs/`): 직접 만든 가상 학원 10곳(수학 5, 영어 5). 실존 학원 이름을 쓰지 않는다

## 구현 단계

1. RAG만: 학원 소개 문서를 검색해서 답하고 `/query`가 answer, contexts, trace를 반환
2. 1차 평가: test_queries.csv와 RAGAS 실행 → `evaluation/round1_report.md`
3. `search_academies` 도구 추가
4. 가드레일 추가: 개인정보 마스킹, 지어내기 방지, 범위 밖 질문 안내
5. 검색 개선 후 2차 평가 → `evaluation/round2_report.md` (1차 대비 개선폭 명시)
6. README.md 완성

지금 어느 단계인지는 사용자가 요청에서 알려준다. 해당 단계만 작업한다.

## 하지 않을 것

- 로그인, 배포, 후기·평점, 지도·거리 계산 (SERVICE.md 범위 밖)
- MCP, 멀티에이전트, HITL은 사용자가 요청할 때만
- Streamlit 화면은 `/query` API가 완성된 뒤 사용자가 요청할 때만
- test_queries.csv는 1차 평가 이후 수정하지 않는다 (개선폭 비교를 위해)

## 실행 명령어

모든 명령은 프로젝트 폴더(`mini-pjt`)에서 실행한다.

```
# 라이브러리 설치
.venv\Scripts\python -m pip install -r requirements.txt

# 질문 하나를 바로 실행해 JSON 출력 (서버 없이)
.venv\Scripts\python -m src.agent "정자동에 중2 수학 내신대비반 있는 학원 알려줘"

# API 서버 실행 (http://127.0.0.1:8000, 문서: /docs)
.venv\Scripts\uvicorn src.api:app --port 8000

# 서버에 질문 보내기 (PowerShell)
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"question": "중1 영어 기초부터 잡아주는 학원 있어?"}')) | ConvertTo-Json -Depth 5
```

- 학원 소개 문서를 고쳤으면 `chroma_db` 폴더를 지우고 다시 실행한다 (처음 실행할 때 새로 만든다).
- LangSmith 추적은 `.env`의 `LANGSMITH_TRACING=true`면 자동으로 켜진다.
