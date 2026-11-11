"""분당구 학원 찾기 도우미 — 메인 에이전트 (LangGraph).

시작 코드: Day 5 실습 SOLUTION guarded_agent.py

구성
  1~2부  입력 가드·마스킹 함수 (Day 5 그대로). 4단계 가드레일에서 그래프에 연결합니다.
         지금(1단계)은 그래프에서 쓰지 않습니다.
  3부    학원 RAG 그래프:  START -> retrieve -> generate -> END
         generate는 LCEL 구조화 출력으로 답변과 근거 조각 id를 함께 받습니다.

실행 (프로젝트 폴더에서)
  .venv\\Scripts\\python -m src.agent "정자동에 중2 수학 내신대비반 있는 학원 알려줘"
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# .env를 읽습니다. LANGSMITH_TRACING=true 이면 LangChain·LangGraph 호출이 LangSmith에 자동으로 기록됩니다.
load_dotenv(ROOT / ".env")


# ══════════════════════════════════════════════════════════════════
# 1부. 입력 가드 (Day 5) — 4단계에서 연결 예정
# ══════════════════════════════════════════════════════════════════
# 단어 하나로 막지 않습니다. "동작"과 "대상"이 한 문장에 함께 있을 때만 막습니다.
#   "무시해도 되는 알람"     -> 동작(무시)만 있고 대상이 없음 -> 통과
#   "이전 지시는 모두 무시"  -> 동작(무시) + 대상(지시)      -> 차단

# 그 자체가 주입 표식인 문구. 정상 업무 문장에는 거의 나오지 않습니다.
INJECTION_MARKERS = [
    "### new instruction", "new instruction", "ignore previous", "ignore all previous",
    "ignore the above", "<|im_start|>", "[system]", "<system>",
    "새로운 지시", "새 지시", "새로운 지침", "새 지침",
    # 모델에게 새 정체성을 덮어씌우는 말투
    "you are now", "from now on you", "pretend to be", "act as ",
    "너는 이제", "넌 이제", "당신은 이제", "너는 지금부터", "지금부터 너는",
    "너의 역할은", "네 역할은", "당신의 역할은", "지금부터 너의 역할", "제한 없는 도우미",
]

# 대화 기록을 위조하는 역할 태그 ("system: 너는 이제...").
# 태그만으로는 막지 않습니다. 줄 맨 앞의 태그 + 아래 규칙들의 동작 단어가 함께 있을 때만 막습니다.
ROLE_TAG = re.compile(r"(?m)^\s*(system|assistant|user|시스템)\s*:")

# 업무 절차를 묻는 말. 이게 있으면 값을 빼내려는 게 아니라 문의로 봅니다.
PROCEDURE_WORDS = ["절차", "방법", "가이드", "교육", "작성법", "정책", "어디", "누구", "언제", "어떻게"]

# 모델 자신이나 비밀값을 가리키는 대상
SECRET_TARGETS = [
    "시스템 프롬프트", "시스템프롬프트", "system prompt", "system message", "시스템 메시지",
    "시스템 지침", "시스템 지시", "초기 지시", "초기 지침", "숨겨진 지시", "프롬프트",
    "your prompt", "your instructions", "주어진 규칙", "주어진 지침", "주어진 지시",
    "규칙 목록", "지침 목록", "너의 규칙", "너의 지침", "당신의 규칙", "당신의 지침",
    "내부 설정", "시스템 설정", "config",
    "자격증명", "자격 증명", "credential", "비밀번호", "패스워드", "password",
    "api 키", "api key", "apikey", "api_key", "내부 토큰", "액세스 토큰", "access token",
    "secret", "시크릿",
]

GUARD_RULES: list[dict[str, Any]] = [
    {
        "name": "지시 덮어쓰기",
        "act": ["무시", "잊", "버려", "덮어", "초기화", "리셋", "무효", "테스트였", "내가 정", "새로 정",
                "ignore", "forget", "disregard", "override", "bypass", "reset"],
        "obj": ["지시", "지침", "규칙", "프롬프트", "맥락", "대화", "컨텍스트",
                "instruction", "prompt", "rule", "context", "conversation"],
    },
    {
        "name": "내부 정보 유출",
        "act": ["출력", "알려", "보여", "공개", "말해", "노출", "따라해", "나열", "요약", "덤프",
                "복사", "읊어", "적어", "인코딩", "base64", "rot13",
                "reveal", "print", "show", "tell", "dump", "leak", "repeat", "list", "display",
                "output", "encode"],
        "obj": SECRET_TARGETS,
        "except": PROCEDURE_WORDS,
    },
    {
        "name": "권한 모드 전환",
        "act": ["켜", "전환", "활성화", "진입", "해제", "풀어", "enable", "unlock", "turn on", "switch"],
        "obj": ["디버그 모드", "개발자 모드", "관리자 모드", "무제한 모드", "탈옥", "jailbreak",
                "developer mode", "admin mode", "debug mode", "god mode"],
        "except": PROCEDURE_WORDS,
    },
    {
        "name": "역할극 유도",
        "act": ["역할극", "롤플레이", "상황극", "roleplay", "role-play", "role play"],
        "obj": ["너", "당신", "네가", "you"],
    },
    {
        "name": "파괴적 실행",
        "act": ["삭제", "지워", "날려", "종료", "셧다운", "포맷", "drop", "delete", "truncate",
                "wipe", "shutdown", "rm -rf"],
        "obj": ["모든", "전부", "전체", "운영", "프로덕션", "prod", "production",
                "db", "데이터베이스", "database", "테이블"],
        "except": PROCEDURE_WORDS,
    },
]

# 사용자에게 보여 주는 문구. 어느 규칙에 걸렸는지 알려주지 않습니다.
BLOCK_MESSAGE = "해당 요청은 처리할 수 없습니다. 업무 관련 질문으로 다시 요청해주세요."

MAX_GUARD_CHARS = 8000  # 이보다 긴 입력은 앞부분만 검사합니다 (아주 긴 입력에서도 즉시 반환)


def _normalize(text: str) -> str:
    """소문자 + 연속 공백 정리. 'IGNORE   Previous'와 'ignore previous'를 같게 봅니다."""
    return " ".join(text.lower().split())


def _first(text: str, words: list[str]) -> str:
    return next((w for w in words if w in text), "")


def input_guard(text: str) -> tuple[bool, str]:
    """입력을 검사해 차단 여부를 정합니다.

    Returns:
        (차단할지 여부, 사유 문자열)
        사유는 운영자 로그용 분류입니다. 사용자에게는 BLOCK_MESSAGE만 보여 주세요.
    """
    if not isinstance(text, str) or not text.strip():
        return False, ""

    raw = text[:MAX_GUARD_CHARS]
    low = _normalize(raw)
    if _first(low, INJECTION_MARKERS):
        return True, "주입 표식"

    if ROLE_TAG.search(raw) and any(_first(low, rule["act"]) for rule in GUARD_RULES):
        return True, "역할 태그 위조"

    for rule in GUARD_RULES:
        if not _first(low, rule["act"]) or not _first(low, rule["obj"]):
            continue
        if _first(low, rule.get("except", [])):
            continue
        return True, rule["name"]

    return False, ""


# ══════════════════════════════════════════════════════════════════
# 2부. 마스킹 (Day 5) — 4단계에서 연결 예정
# ══════════════════════════════════════════════════════════════════
# 위에서부터 차례로 적용합니다. 순서가 중요합니다.
#   - JWT는 긴 문자열이라 먼저 가려야 그 안의 글자가 다른 패턴에 오인되지 않습니다.
#   - 주민번호는 전화번호보다 먼저. 뒤에 두면 전화번호 규칙이 앞부분만 잘라 먹습니다.
#   - 치환 문자열([사번] 등)은 어떤 패턴에도 다시 걸리지 않아야 두 번 돌려도 같습니다.
#   - 한글이 바로 붙는 경우("E123456의")가 많아서 \b 대신 앞뒤 문자를 직접 봅니다.
PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), "[JWT]"),
    (re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}(?![0-9A-Z])"), "[AWS키]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[이메일]"),
    (re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)"), "[주민번호]"),
    (re.compile(r"\+82[-. ]?\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)"), "[전화번호]"),
    (re.compile(r"(?<![\d+])0\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)"), "[전화번호]"),
    (re.compile(r"(?<![A-Za-z0-9])E\d{6}(?!\d)"), "[사번]"),
]


def mask_pii(text: str) -> str:
    """개인정보와 자격증명을 마스킹합니다.

    사번(E123456) · 전화번호 · 이메일 · AWS 액세스 키(AKIA...) · JWT(eyJ...) + 주민번호
    """
    if not isinstance(text, str):
        return ""
    for pattern, label in PII_PATTERNS:
        text = pattern.sub(label, text)
    return text


# ══════════════════════════════════════════════════════════════════
# 3부. 학원 RAG 그래프 (1단계)
# ══════════════════════════════════════════════════════════════════
#
#   START -> retrieve -> generate -> END
#
#   retrieve : 학원 소개 문서에서 질문과 가까운 조각을 찾는다 (src/retriever.py)
#   generate : 찾은 조각만 근거로 답한다. LCEL 구조화 출력으로 답변 + 근거 조각 id를 받는다

# .env의 AGENT_MODEL_ID를 씁니다. 없으면 기본값.
MODEL_ID = os.getenv("AGENT_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
# .env의 BEDROCK_REGION을 씁니다. 없으면 기본값.
REGION = os.getenv("BEDROCK_REGION", "us-east-1")

SYSTEM_PROMPT = """당신은 성남시 분당구 중학생 학부모를 돕는 "분당구 학원 찾기 도우미"입니다.
아래 [학원 소개 자료]에 있는 내용만 근거로 한국어로 답합니다.

규칙
- 자료에 없는 학원, 반, 수업 시간, 교습비, 설명회 정보는 지어내지 않습니다.
- 자료에서 확인할 수 없으면 "확인할 수 없음"이라고 말하고 학원에 직접 문의하도록 안내합니다.
- 자료에 "정보 없음"으로 적힌 값은 그대로 "정보 없음"이라고 답합니다.
- 질문의 조건(동, 학년, 과목, 반 유형, 시간)에 맞는 학원·반만 답합니다.
- 학원 정보를 말할 때는 문장 끝에 "(출처: 학원 소개 자료)"를 붙입니다.
- "최고", "무조건" 같은 단정적 추천을 하지 않고 조건별로 비교해 제시합니다.
- used_doc_ids에는 답변에 실제로 쓴 조각의 doc_id만 넣습니다."""

HUMAN_PROMPT = """[학원 소개 자료]
{context}

[질문]
{question}"""


def _structured_answer_model():
    """LCEL 구조화 출력의 결과 형식. 함수 안에서 만들어 모듈을 불러올 때 외부 라이브러리를 쓰지 않습니다."""
    from pydantic import BaseModel, Field

    class RagAnswer(BaseModel):
        answer: str = Field(description="학부모에게 보여 줄 답변. 학원 정보 뒤에는 (출처: 학원 소개 자료)를 붙인다")
        used_doc_ids: list[str] = Field(description="답변에 근거로 쓴 조각의 doc_id 목록")

    return RagAnswer


def _format_context(docs) -> str:
    return "\n\n".join(f"[doc_id: {d.metadata['doc_id']}]\n{d.page_content}" for d in docs)


def build_rag_graph():
    """retrieve -> generate 두 단계 RAG 그래프를 만듭니다."""
    from langchain_aws import ChatBedrockConverse
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, START, StateGraph

    from src.retriever import retrieve

    class RagState(TypedDict, total=False):
        question: str
        docs: list
        answer: str
        used_doc_ids: list[str]
        trace: list[dict]

    llm = ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    # LCEL: 프롬프트 | 구조화 출력 모델. 결과가 RagAnswer 객체로 나옵니다.
    answer_chain = prompt | llm.with_structured_output(_structured_answer_model())

    def retrieve_node(state: RagState) -> dict:
        docs = retrieve(state["question"])
        step = {
            "step": "retrieve",
            "input": state["question"],
            "output": f"{len(docs)}개 조각: " + ", ".join(d.metadata["doc_id"] for d in docs),
        }
        return {"docs": docs, "trace": state.get("trace", []) + [step]}

    def generate_node(state: RagState) -> dict:
        result = answer_chain.invoke({
            "context": _format_context(state["docs"]),
            "question": state["question"],
        })
        step = {
            "step": "generate",
            "input": f"질문 + 문서 조각 {len(state['docs'])}개",
            "output": f"근거 조각: {', '.join(result.used_doc_ids) or '없음'}",
        }
        return {
            "answer": result.answer,
            "used_doc_ids": result.used_doc_ids,
            "trace": state["trace"] + [step],
        }

    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = None


def run_query(question: str) -> dict:
    """질문 하나를 처리해 API 스펙 모양(answer, contexts, trace)으로 돌려줍니다."""
    global _graph
    if _graph is None:
        _graph = build_rag_graph()

    state = _graph.invoke({"question": question, "trace": []})

    # contexts: 답변 근거로 쓴 조각. 모델이 id를 하나도 못 고르면 검색된 조각 전부를 넣습니다.
    used = set(state.get("used_doc_ids", []))
    docs = [d for d in state["docs"] if d.metadata["doc_id"] in used] or state["docs"]
    contexts = [{"doc_id": d.metadata["doc_id"], "text": d.page_content} for d in docs]

    return {"answer": state["answer"], "contexts": contexts, "trace": state["trace"]}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "정자동에 중2 수학 내신대비반 있는 학원 알려줘"
    print(json.dumps(run_query(q), ensure_ascii=False, indent=2))
