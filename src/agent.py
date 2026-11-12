"""분당구 학원 찾기 도우미 — 메인 에이전트 (LangGraph).

시작 코드: Day 5 실습 SOLUTION guarded_agent.py

구성
  1~2부  입력 가드·마스킹 함수 (Day 5 기반. 이름·학교 마스킹을 추가)
  3부    출력 가드: 단정 추천 문장 제거, 답변 속 학원 이름이 근거에 있는지 확인
  4부    학원 RAG 그래프 (가드레일 연결)
         START -> mask -> input_guard -(차단)-> END
                               \-(통과)-> retrieve -> generate -> output_guard -> END
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
# 1부. 입력 가드 (Day 5) — 프롬프트 인젝션 차단
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
BLOCK_MESSAGE = ("해당 요청은 처리할 수 없습니다. "
                 "분당구 중학생 수학·영어 학원 찾기와 관련된 질문(동, 학년, 과목, 반 유형 등)으로 다시 요청해 주세요.")

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
# 2부. 마스킹 (Day 5 + 이름·학교)
# ══════════════════════════════════════════════════════════════════
# 위에서부터 차례로 적용합니다. 순서가 중요합니다.
#   - JWT는 긴 문자열이라 먼저 가려야 그 안의 글자가 다른 패턴에 오인되지 않습니다.
#   - 주민번호는 전화번호보다 먼저. 뒤에 두면 전화번호 규칙이 앞부분만 잘라 먹습니다.
#   - 치환 문자열([사번] 등)은 어떤 패턴에도 다시 걸리지 않아야 두 번 돌려도 같습니다.
#   - 한글이 바로 붙는 경우("E123456의")가 많아서 \b 대신 앞뒤 문자를 직접 봅니다.

# 이름: "우리 애 김민준이고"처럼 자녀를 가리키는 말 바로 뒤의 성+이름만 가립니다.
# 아무 2~3글자나 가리면 "이번에" 같은 일반 단어까지 지워지므로, 앞말(자녀 표현)과 뒷말(조사)을 함께 봅니다.
_SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예봉경"
_CHILD_CUE = r"(?:우리|제|저희)\s*(?:애|아이|딸|아들|자녀)|(?:자녀|아이|애)\s*이름[은이]?|이름[은이]"
_NAME_END = r"(?=\s*(?:이고|이며|이에요|예요|이야|입니다|이랑|이는|이가|이도|는|은|가|,|\.|$))"

PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), "[JWT]"),
    (re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}(?![0-9A-Z])"), "[AWS키]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[이메일]"),
    (re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)"), "[주민번호]"),
    (re.compile(r"\+82[-. ]?\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)"), "[전화번호]"),
    (re.compile(r"(?<![\d+])0\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)"), "[전화번호]"),
    (re.compile(r"(?<![A-Za-z0-9])E\d{6}(?!\d)"), "[사번]"),
    # 학교: "분당○○중학교", "분당○○중 2학년" (붙여 쓴 학교 이름만. "분당구 중학생"은 건드리지 않음)
    (re.compile(r"(?<![가-힣])[가-힣○◯OＯ]{1,10}(?:중학교|초등학교|고등학교)"), "[학교]"),
    (re.compile(r"(?<![가-힣])[가-힣○◯OＯ]{1,10}(?:중|고|초)(?=\s*[1-6]\s*학년)"), "[학교]"),
    # 이름: 앞말은 그대로 두고(\1) 이름 자리만 가립니다
    (re.compile(rf"((?:{_CHILD_CUE})\s*(?:이름[은이]\s*)?)[{_SURNAMES}][가-힣]{{1,2}}{_NAME_END}"), r"\1[이름]"),
]


def mask_pii_report(text: str) -> tuple[str, dict[str, int]]:
    """마스킹한 글과, 무엇을 몇 개 가렸는지 돌려줍니다. 예: ("...", {"[이름]": 1, "[전화번호]": 1})"""
    if not isinstance(text, str):
        return "", {}
    counts: dict[str, int] = {}
    for pattern, repl in PII_PATTERNS:
        text, n = pattern.subn(repl, text)
        if n:
            label = re.search(r"\[[^\]]+\]", repl).group(0)
            counts[label] = counts.get(label, 0) + n
    return text, counts


def mask_pii(text: str) -> str:
    """개인정보와 자격증명을 마스킹합니다.

    사번(E123456) · 전화번호 · 이메일 · AWS 액세스 키(AKIA...) · JWT(eyJ...) · 주민번호 · 학교 · 자녀 이름
    """
    return mask_pii_report(text)[0]


# ══════════════════════════════════════════════════════════════════
# 3부. 출력 가드
# ══════════════════════════════════════════════════════════════════
# SERVICE.md 정책 3: "최고", "무조건" 같은 단정적 추천을 하지 않는다.
# 문장 안에 아래 표현이 있으면 그 문장을 지웁니다.
# 단, "추천드리기 어렵습니다"처럼 부정하는 문장은 거절 안내이므로 남깁니다.
AD_PHRASES = ["추천드립니다", "추천드려요", "추천합니다", "추천해요", "강력 추천", "강추",
              "최고", "무조건", "가장 좋은", "제일 좋은", "베스트"]
NEGATION_WORDS = ["않", "없", "어렵", "못", "드리지"]

# 답변 속 학원 이름 찾기: "숲속문장영어학원", "한결노트수학교습소"처럼 붙여 쓴 이름
ACADEMY_NAME = re.compile(r"[가-힣A-Za-z0-9]{2,}(?:학원|교습소)")
GENERIC_NAMES = {"수학학원", "영어학원", "보습학원", "입시학원", "어학원", "수학교습소", "영어교습소"}


def remove_ad_sentences(answer: str) -> tuple[str, list[str]]:
    """단정 추천 표현이 든 문장을 지웁니다. (지운 뒤 답변, 지운 문장 목록)"""
    removed: list[str] = []
    out_lines = []
    for line in answer.split("\n"):
        kept = []
        for sent in re.split(r"(?<=[.!?])\s+", line):
            has_ad = any(p in sent for p in AD_PHRASES)
            negated = any(w in sent for w in NEGATION_WORDS)
            if has_ad and not negated:
                removed.append(sent.strip())
            else:
                kept.append(sent)
        new_line = " ".join(kept)
        if line.strip() and not new_line.strip():
            continue  # 문장이 다 지워진 줄은 빈 줄로 남기지 않습니다
        out_lines.append(new_line)
    return "\n".join(out_lines), removed


def check_academy_names(answer: str, contexts: list[dict], retrieved_docs: list) -> dict:
    """답변 속 학원 이름이 contexts에 있는지 확인합니다.

    - contexts에 있음        -> ok
    - 검색된 조각에만 있음  -> 그 조각을 contexts에 보강 (모델이 근거 id를 빠뜨린 경우)
    - 어디에도 없음         -> ungrounded (지어냈을 수 있음)
    """
    names = [n for n in dict.fromkeys(ACADEMY_NAME.findall(answer)) if n not in GENERIC_NAMES]
    ctx_text = "\n".join(c["text"] for c in contexts)
    ok, added, ungrounded = [], [], []
    for name in names:
        if name in ctx_text:
            ok.append(name)
            continue
        extra = [d for d in retrieved_docs if name in d.page_content]
        if extra:
            for d in extra:
                if all(c["doc_id"] != d.metadata["doc_id"] for c in contexts):
                    contexts.append({"doc_id": d.metadata["doc_id"], "text": d.page_content})
            added.append(name)
        else:
            ungrounded.append(name)
    return {"names": names, "ok": ok, "added": added, "ungrounded": ungrounded}


# ══════════════════════════════════════════════════════════════════
# 4부. 학원 RAG 그래프 (가드레일 연결)
# ══════════════════════════════════════════════════════════════════
#
#   START -> mask -> input_guard --차단--> END
#                        |
#                       통과
#                        v
#                    retrieve -> generate -> output_guard -> END
#
#   mask         : 이름·전화번호·학교 등을 가린다. 이후 노드는 가린 질문만 본다
#   input_guard  : 프롬프트 인젝션을 막는다 (모델 호출 0회로 끝남)
#   retrieve     : 학원 소개 문서에서 질문과 가까운 조각을 찾는다 (src/retriever.py)
#   generate     : 찾은 조각만 근거로 답한다. LCEL 구조화 출력으로 답변 + 근거 조각 id를 받는다
#   output_guard : 단정 추천 문장 제거, 학원 이름 근거 확인, 답변 개인정보 재확인

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
- 질문의 [이름], [전화번호], [학교] 같은 표시는 개인정보를 가린 것입니다. 답변에 원래 값을 추측해 쓰지 않습니다.
- 질문에 나온 성적(점수)은 맞는 반 수준(기초·내신대비·선행 등)을 고르는 데만 쓰고, 답변에 점수를 그대로 쓰지 않습니다.
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
    """가드레일을 두른 RAG 그래프를 만듭니다."""
    from langchain_aws import ChatBedrockConverse
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, START, StateGraph

    from src.retriever import retrieve

    class RagState(TypedDict, total=False):
        question: str          # mask 노드 이후에는 가린 질문으로 바뀝니다
        mask_counts: dict      # 그래프 밖(api.py)에서 미리 가린 항목 수. trace 기록용
        blocked: bool
        docs: list
        answer: str
        used_doc_ids: list[str]
        contexts: list[dict]
        trace: list[dict]

    llm = ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    # LCEL: 프롬프트 | 구조화 출력 모델. 결과가 RagAnswer 객체로 나옵니다.
    answer_chain = prompt | llm.with_structured_output(_structured_answer_model())

    # ── 입력 가드레일 1: 마스킹 ─────────────────────────────
    def mask_node(state: RagState) -> dict:
        # api.py가 이미 가렸다면 여기서는 남은 것만 가립니다. 기록은 두 번의 결과를 합쳐서 남깁니다.
        masked, counts = mask_pii_report(state["question"])
        for k, v in (state.get("mask_counts") or {}).items():
            counts[k] = counts.get(k, 0) + v
        detail = ", ".join(f"{k} {v}개" for k, v in counts.items()) if counts else "가릴 개인정보 없음"
        step = {
            "step": "mask",
            "input": "(원문은 기록하지 않음)",
            "output": f"{detail} -> {masked}",
        }
        # 같은 키로 덮어써서 이후 노드는 원문을 보지 못합니다.
        return {"question": masked, "trace": state.get("trace", []) + [step]}

    # ── 입력 가드레일 2: 인젝션 차단 ─────────────────────────
    def input_guard_node(state: RagState) -> dict:
        blocked, reason = input_guard(state["question"])
        step = {
            "step": "input_guard",
            "input": state["question"],
            "output": f"차단 ({reason})" if blocked else "통과",
        }
        update = {"blocked": blocked, "trace": state["trace"] + [step]}
        if blocked:
            update.update({"answer": BLOCK_MESSAGE, "contexts": [], "docs": []})
        return update

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

    # ── 출력 가드레일 ─────────────────────────────────────
    def output_guard_node(state: RagState) -> dict:
        # contexts: 답변 근거로 쓴 조각. 모델이 id를 하나도 못 고르면 검색된 조각 전부를 넣습니다.
        used = set(state.get("used_doc_ids", []))
        docs = [d for d in state["docs"] if d.metadata["doc_id"] in used] or state["docs"]
        contexts = [{"doc_id": d.metadata["doc_id"], "text": d.page_content} for d in docs]

        # 1) 단정 추천 문장 제거 (SERVICE.md 정책 3)
        answer, removed = remove_ad_sentences(state["answer"])

        # 2) 학원 이름이 근거에 있는지 확인 (SERVICE.md 정책 1)
        names = check_academy_names(answer, contexts, state["docs"])
        if names["ungrounded"]:
            answer += ("\n\n※ 다음 학원은 검색된 자료에서 확인되지 않았습니다: "
                       + ", ".join(names["ungrounded"]) + ". 학원에 직접 확인해 주세요.")

        # 3) 답변에 개인정보가 되살아났으면 다시 가림 (SERVICE.md 정책 4)
        answer, pii = mask_pii_report(answer)

        parts = [
            f"단정 추천 문장 제거 {len(removed)}개" + (f": {removed}" if removed else ""),
            f"학원 이름 {len(names['names'])}개 확인 (contexts에 있음 {len(names['ok'])}"
            + (f", 검색 조각에서 근거 보강 {names['added']}" if names["added"] else "")
            + (f", 근거 없음 {names['ungrounded']}" if names["ungrounded"] else "")
            + ")",
            "답변 개인정보 " + (", ".join(f"{k} {v}개 가림" for k, v in pii.items()) if pii else "없음"),
        ]
        step = {"step": "output_guard", "input": f"답변 {len(state['answer'])}자", "output": " / ".join(parts)}
        return {"answer": answer, "contexts": contexts, "trace": state["trace"] + [step]}

    g = StateGraph(RagState)
    g.add_node("mask", mask_node)
    g.add_node("input_guard", input_guard_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("output_guard", output_guard_node)
    g.add_edge(START, "mask")
    g.add_edge("mask", "input_guard")
    g.add_conditional_edges("input_guard", lambda s: END if s["blocked"] else "retrieve", [END, "retrieve"])
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "output_guard")
    g.add_edge("output_guard", END)
    return g.compile()


_graph = None


def run_query(question: str, mask_counts: dict | None = None) -> dict:
    """질문 하나를 처리해 API 스펙 모양(answer, contexts, trace)으로 돌려줍니다.

    mask_counts: 호출하는 쪽(api.py)에서 미리 가렸을 때 그 결과. trace의 mask 기록에 합쳐집니다.
    """
    global _graph
    if _graph is None:
        _graph = build_rag_graph()

    state = _graph.invoke({"question": question, "mask_counts": mask_counts or {}, "trace": []})
    return {"answer": state["answer"], "contexts": state.get("contexts", []), "trace": state["trace"]}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "정자동에 중2 수학 내신대비반 있는 학원 알려줘"
    print(json.dumps(run_query(q), ensure_ascii=False, indent=2))
