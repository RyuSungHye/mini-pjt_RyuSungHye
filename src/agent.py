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
GENERIC_NAMES = {"수학학원", "영어학원", "보습학원", "입시학원", "어학원", "수학교습소", "영어교습소", "학원교습소"}


def _name_key(text: str) -> str:
    """이름 비교용: 괄호 안 글자와 기호를 뺍니다. '매쓰프레소(Mathpresso)수학교습소' -> '매쓰프레소수학교습소'"""
    return re.sub(r"\([^)]*\)|[^가-힣A-Za-z0-9]", "", text)


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


def check_academy_names(answer: str, contexts: list[dict], pool: list[dict]) -> dict:
    """답변 속 학원 이름이 contexts에 있는지 확인합니다.

    pool: 이번 질문에서 모델에게 준 근거 전체 (가상 문서 조각 + 공공데이터 결과)
    - contexts에 있음   -> ok
    - pool에만 있음     -> 그 근거를 contexts에 보강 (모델이 근거 id를 빠뜨린 경우)
    - 어디에도 없음     -> ungrounded (지어냈을 수 있음)
    """
    names = [n for n in dict.fromkeys(ACADEMY_NAME.findall(answer)) if n not in GENERIC_NAMES]
    ctx_key = _name_key("\n".join(c["text"] for c in contexts))
    ok, added, ungrounded = [], [], []
    for name in names:
        key = _name_key(name)
        if key in ctx_key:
            ok.append(name)
            continue
        extra = [p for p in pool if key in _name_key(p["text"])]
        if extra:
            for p in extra:
                if all(c["doc_id"] != p["doc_id"] for c in contexts):
                    contexts.append(p)
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
#                    tool_call -> retrieve -> generate -> output_guard -> END
#
#   mask         : 이름·전화번호·학교 등을 가린다. 이후 노드는 가린 질문만 본다
#   input_guard  : 프롬프트 인젝션을 막는다 (모델 호출 0회로 끝남)
#   tool_call    : [실제] 모델이 필요하다고 판단하면 search_academies로 공공데이터를 조회한다 (src/tools.py)
#                  여기서 정한 동·과목 조건을 retrieve가 재활용한다
#   retrieve     : [가상] 학원 소개 문서 검색 (src/retriever.py)
#                  ② 동·과목 필터 -> 가까운 조각 -> ① 학원 단위 확장 (반 구성까지 함께)
#   generate     : 두 자료를 따로 받아 답한다. LCEL 구조화 출력으로 답변 + 근거 id를 받는다
#   output_guard : 단정 추천 문장 제거, 학원 이름 근거 확인, 가상/실제 묶음 표시 확인, 답변 개인정보 재확인
#
#   가상 학원과 실제 학원은 섞지 않는다 (SERVICE.md 정책 2).
#   contexts의 doc_id는 virtual:... / public:... 으로 출처를 구분한다.

# .env의 AGENT_MODEL_ID를 씁니다. 없으면 기본값.
MODEL_ID = os.getenv("AGENT_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
# .env의 BEDROCK_REGION을 씁니다. 없으면 기본값.
REGION = os.getenv("BEDROCK_REGION", "us-east-1")

SYSTEM_PROMPT = """당신은 성남시 분당구 중학생 학부모를 돕는 "분당구 학원 찾기 도우미"입니다.
아래 두 자료에 있는 내용만 근거로 한국어로 답합니다.
- [학원 소개 자료 (가상)]: 직접 만든 가상 학원 소개 문서입니다. 실존 학원이 아닙니다. doc_id가 virtual:로 시작합니다.
- [공공데이터 (실제)]: 나이스 학원교습소정보에서 조회한 실제 학원입니다. doc_id가 public:으로 시작합니다.

가상 학원과 실제 학원을 섞지 않기 (가장 중요)
- 학원 정보를 안내할 때는 "## 학원 소개 자료 (가상 학원)"과 "## 공공데이터 (실제 학원)" 두 묶음으로 나눠 씁니다.
- 가상 학원 이름 뒤에는 "(가상)", 실제 학원 이름 뒤에는 "(실제)"를 붙입니다.
- 두 묶음을 한 표나 한 순위로 합치지 않습니다. 교습비 비교도 같은 묶음 안에서만 합니다.
- 한쪽 자료에 맞는 내용이 없으면 그 묶음에 "해당 자료 없음" 또는 "이번 질문에서는 조회하지 않음"이라고 짧게 씁니다.
- 공공데이터 목록은 학원명 가나다순이며 순위가 아닙니다. 전체 개수가 더 많으면 일부만 보여 준다고 알립니다.
- 공공데이터에는 반 이름·수업 시간·설명회 정보가 없습니다. 이런 정보는 "확인할 수 없음"으로 답합니다.

규칙
- 자료에 없는 학원, 반, 수업 시간, 교습비, 설명회 정보는 지어내지 않습니다.
- 자료에서 확인할 수 없으면 "확인할 수 없음"이라고 말하고 학원에 직접 문의하도록 안내합니다.
- 자료에 "정보 없음"으로 적힌 값은 그대로 "정보 없음"이라고 답합니다.
- 질문의 조건(동, 학년, 과목, 반 유형, 시간)에 맞는 학원·반만 답합니다.
- 가상 학원 정보 끝에는 "(출처: 학원 소개 자료·가상)", 실제 학원 정보 끝에는 "(출처: 공공데이터·실제)"를 붙입니다.
- "최고", "무조건" 같은 단정적 추천을 하지 않고 조건별로 비교해 제시합니다.
- 질문의 [이름], [전화번호], [학교] 같은 표시는 개인정보를 가린 것입니다. 답변에 원래 값을 추측해 쓰지 않습니다.
- 질문에 나온 성적(점수)은 맞는 반 수준(기초·내신대비·선행 등)을 고르는 데만 쓰고, 답변에 점수를 그대로 쓰지 않습니다.
- used_doc_ids에는 답변에 실제로 쓴 조각의 doc_id만 넣습니다."""

HUMAN_PROMPT = """[학원 소개 자료 (가상)]
{virtual_context}

[공공데이터 (실제)]
{public_context}

[질문]
{question}"""

# 실제 학원 조회(search_academies)를 할지 정하는 모델의 지시문
TOOL_SYSTEM_PROMPT = """당신은 분당구 학원 찾기 도우미의 도구 선택 담당입니다.
질문을 보고 실제 학원(공공데이터) 조회가 필요하면 search_academies를 한 번 호출합니다.
- 동(지역)이나 과목으로 학원 목록·교습비를 찾는 질문이면 호출합니다. 동이 없으면 dong을 비워 분당구 전체로 조회합니다.
- 초·중등 혼합 과정 여부를 묻는 질문도 호출합니다. 공공데이터에 수강료 항목으로 본 대상(혼합 여부) 정보가 있습니다.
- 반 유형·수업 시간·설명회처럼 공공데이터에 없는 정보만 묻는 질문은 호출하지 않습니다.
  동 없이 반 유형(기초·내신대비·선행 등)·수업 시간·설명회를 찾는 질문도 호출하지 않습니다. 예: "중1 영어 기초부터 잡아주는 학원 있어?"
- 분당구 밖 지역, 후기·평점·순위, 학원 찾기와 상관없는 질문은 호출하지 않습니다.
- "판교 쪽"처럼 넓은 지역은 판교 생활권 동(판교동, 삼평동, 백현동, 운중동)으로 넣습니다.
- 과목은 "수학" 또는 "영어"만 넣습니다."""


# ② 동·과목 필터용 조건. 도구를 부르지 않은 질문은 질문 속 낱말에서 직접 뽑습니다.
BUNDANG_DONGS = ["정자동", "수내동", "서현동", "이매동", "야탑동", "구미동", "금곡동", "대장동", "판교동",
                 "삼평동", "백현동", "운중동", "분당동", "율동", "동원동", "궁내동", "석운동", "하산운동", "도촌동"]
PANGYO_DONGS = ["판교동", "삼평동", "백현동", "운중동"]  # "판교 쪽" = 판교 생활권


def extract_conditions(question: str) -> dict:
    """질문에서 동·과목을 뽑습니다. 과목이 둘 다 나오면(수학이랑 영어) 과목 조건을 걸지 않습니다."""
    q = question.lower()
    dongs = [d for d in BUNDANG_DONGS if d in question]
    if not dongs and "판교" in question:
        dongs = list(PANGYO_DONGS)
    has_math = any(w in q for w in ("수학", "매스", "math"))
    has_eng = any(w in q for w in ("영어", "잉글리시", "english"))
    subject = "수학" if has_math and not has_eng else "영어" if has_eng and not has_math else None
    return {"dong": dongs or None, "subject": subject, "source": "질문 키워드"}


def _structured_answer_model():
    """LCEL 구조화 출력의 결과 형식. 함수 안에서 만들어 모듈을 불러올 때 외부 라이브러리를 쓰지 않습니다."""
    from pydantic import BaseModel, Field

    class RagAnswer(BaseModel):
        answer: str = Field(description="학부모에게 보여 줄 답변. 가상·실제 두 묶음으로 나누고 출처를 붙인다")
        used_doc_ids: list[str] = Field(description="답변에 근거로 쓴 doc_id 목록 (virtual:... 또는 public:...)")

    return RagAnswer


def _virtual_items(docs) -> list[dict]:
    """학원 소개 문서 조각 -> contexts 항목. doc_id에 virtual: 접두어를 붙입니다."""
    return [{"doc_id": f"virtual:{d.metadata['doc_id']}", "text": d.page_content} for d in docs]


def _public_items(public: dict | None) -> list[dict]:
    """search_academies 결과 -> contexts 항목. doc_id에 public: 접두어를 붙입니다."""
    if not public:
        return []
    return [{
        "doc_id": f"public:{r['id']}",
        "text": (f"[공공데이터·실제] 학원명: {r['학원명']} | 동: {r['동']} | 과목: {r['과목']} | "
                 f"교습과정: {r['교습과정']} | 대상: {r['대상']} | 수강료(중등): {r['수강료(중등)']}"),
    } for r in public["결과"]]


def _format_items(items: list[dict]) -> str:
    return "\n\n".join(f"[doc_id: {c['doc_id']}]\n{c['text']}" for c in items)


def _format_public(public: dict | None) -> str:
    if public is None:
        return "(이번 질문에서는 조회하지 않음)"
    cond = public["조건"]
    head = (f"조회 조건: 동={cond['동']}, 과목={cond['과목']} / "
            f"조건에 맞는 실제 학원 전체 {public['전체']}곳 중 {len(public['결과'])}곳 (학원명 가나다순, 순위 아님)")
    if not public["결과"]:
        return head + "\n조건에 맞는 실제 학원 없음"
    return head + "\n\n" + _format_items(_public_items(public))


def build_rag_graph():
    """가드레일을 두른 RAG 그래프를 만듭니다."""
    from langchain_aws import ChatBedrockConverse
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, START, StateGraph

    from src.retriever import retrieve_academies
    from src.tools import normalize_dong, search_academies, search_academies_data

    class RagState(TypedDict, total=False):
        question: str          # mask 노드 이후에는 가린 질문으로 바뀝니다
        mask_counts: dict      # 그래프 밖(api.py)에서 미리 가린 항목 수. trace 기록용
        blocked: bool
        docs: list             # [가상] 학원 소개 문서 조각
        public: dict | None    # [실제] search_academies 결과. 호출하지 않았으면 None
        conditions: dict       # 동·과목 조건 (tool_call 인자 또는 질문 키워드). retrieve 필터에 씀
        answer: str
        used_doc_ids: list[str]
        contexts: list[dict]
        trace: list[dict]

    llm = ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    # LCEL: 프롬프트 | 구조화 출력 모델. 결과가 RagAnswer 객체로 나옵니다.
    answer_chain = prompt | llm.with_structured_output(_structured_answer_model())
    # 도구 호출용 모델: search_academies의 설명·인자 형식을 보고 호출 여부와 인자를 정합니다.
    tool_llm = llm.bind_tools([search_academies])

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

    # ── [가상] 학원 소개 문서 검색 ───────────────────────────
    def retrieve_node(state: RagState) -> dict:
        # ② 동·과목 필터: tool_call이 정한 조건(없으면 질문 키워드)으로 검색 범위를 좁힙니다.
        cond = state.get("conditions") or {}
        # ① 학원 단위 확장: 걸린 조각의 학원은 모든 섹션(반 구성 포함)을 함께 넘깁니다.
        res = retrieve_academies(state["question"], dongs=cond.get("dong"), subject=cond.get("subject"))
        hits = ", ".join(f"virtual:{h.metadata['doc_id']}" for h in res["hits"])
        flt = (f"필터: 동={cond.get('dong') or '전체'}, 과목={cond.get('subject') or '전체'} (조건 출처: {cond.get('source')})"
               if res["filter"] else "필터 없음 (질문에 동·과목 조건 없음)")
        step = {
            "step": "retrieve",
            "input": state["question"],
            "output": (f"[가상] {flt} / 검색 {len(res['hits'])}조각: {hits} "
                       f"-> 학원 단위 확장 {len(res['academies'])}곳 {len(res['docs'])}조각: {', '.join(res['academies'])}"),
        }
        return {"docs": res["docs"], "trace": state["trace"] + [step]}

    # ── [실제] 공공데이터 조회 도구 ──────────────────────────
    def tool_call_node(state: RagState) -> dict:
        ai = tool_llm.invoke([SystemMessage(content=TOOL_SYSTEM_PROMPT), HumanMessage(content=state["question"])])
        calls = [c for c in (getattr(ai, "tool_calls", None) or []) if c["name"] == "search_academies"]
        if not calls:
            step = {"step": "tool_call", "input": "search_academies",
                    "output": "호출 안 함 (공공데이터 조회가 필요 없는 질문으로 판단)"}
            return {"public": None, "conditions": extract_conditions(state["question"]),
                    "trace": state["trace"] + [step]}

        args = calls[0]["args"]  # 한 질문에 한 번만 조회합니다
        public = search_academies_data(args.get("dong"), args.get("subject"))
        # 도구 인자를 가상 학원 검색 필터로 재활용합니다.
        raw_dongs = args.get("dong") or []
        raw_dongs = [raw_dongs] if isinstance(raw_dongs, str) else raw_dongs
        conditions = {"dong": [normalize_dong(d) for d in raw_dongs] or None,
                      "subject": args.get("subject") if args.get("subject") in ("수학", "영어") else None,
                      "source": "tool_call 인자"}
        names = ", ".join(r["학원명"] for r in public["결과"]) or "없음"
        step = {
            "step": "tool_call",
            "input": f"search_academies({json.dumps(args, ensure_ascii=False)})",
            "output": f"[실제] 조건에 맞는 {public['전체']}곳 중 {len(public['결과'])}곳 전달 (가나다순): {names}",
        }
        return {"public": public, "conditions": conditions, "trace": state["trace"] + [step]}

    def generate_node(state: RagState) -> dict:
        result = answer_chain.invoke({
            "virtual_context": _format_items(_virtual_items(state["docs"])) or "(검색된 학원 소개 자료 없음)",
            "public_context": _format_public(state.get("public")),
            "question": state["question"],
        })
        n_public = len((state.get("public") or {}).get("결과", []))
        step = {
            "step": "generate",
            "input": f"질문 + [가상] 조각 {len(state['docs'])}개 + [실제] 학원 {n_public}곳",
            "output": f"근거: {', '.join(result.used_doc_ids) or '없음'}",
        }
        return {
            "answer": result.answer,
            "used_doc_ids": result.used_doc_ids,
            "trace": state["trace"] + [step],
        }

    # ── 출력 가드레일 ─────────────────────────────────────
    def output_guard_node(state: RagState) -> dict:
        # contexts: 답변 근거로 쓴 항목. 모델이 id를 하나도 못 고르면 받은 근거 전부를 넣습니다.
        pool = _virtual_items(state["docs"]) + _public_items(state.get("public"))
        used = set(state.get("used_doc_ids", []))
        contexts = [c for c in pool if c["doc_id"] in used] or list(pool)

        # 1) 단정 추천 문장 제거 (SERVICE.md 정책 3)
        answer, removed = remove_ad_sentences(state["answer"])

        # 2) 학원 이름이 근거에 있는지 확인 (SERVICE.md 정책 1)
        names = check_academy_names(answer, contexts, pool)
        if names["ungrounded"]:
            answer += ("\n\n※ 다음 학원은 검색된 자료에서 확인되지 않았습니다: "
                       + ", ".join(names["ungrounded"]) + ". 학원에 직접 확인해 주세요.")

        # 3) 가상/실제 묶음이 답변에 표시됐는지 확인 (SERVICE.md 정책 2)
        n_virtual = sum(c["doc_id"].startswith("virtual:") for c in contexts)
        n_public = sum(c["doc_id"].startswith("public:") for c in contexts)
        bundle = (f"묶음 표시: 가상 근거 {n_virtual}개·답변 '가상' 표시 {'O' if '가상' in answer else 'X'}, "
                  f"실제 근거 {n_public}개·답변 '실제' 표시 {'O' if '실제' in answer else 'X'}")

        # 4) 답변에 개인정보가 되살아났으면 다시 가림 (SERVICE.md 정책 4)
        answer, pii = mask_pii_report(answer)

        parts = [
            f"단정 추천 문장 제거 {len(removed)}개" + (f": {removed}" if removed else ""),
            f"학원 이름 {len(names['names'])}개 확인 (contexts에 있음 {len(names['ok'])}"
            + (f", 받은 근거에서 보강 {names['added']}" if names["added"] else "")
            + (f", 근거 없음 {names['ungrounded']}" if names["ungrounded"] else "")
            + ")",
            bundle,
            "답변 개인정보 " + (", ".join(f"{k} {v}개 가림" for k, v in pii.items()) if pii else "없음"),
        ]
        step = {"step": "output_guard", "input": f"답변 {len(state['answer'])}자", "output": " / ".join(parts)}
        return {"answer": answer, "contexts": contexts, "trace": state["trace"] + [step]}

    g = StateGraph(RagState)
    g.add_node("mask", mask_node)
    g.add_node("input_guard", input_guard_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("tool_call", tool_call_node)
    g.add_node("generate", generate_node)
    g.add_node("output_guard", output_guard_node)
    g.add_edge(START, "mask")
    g.add_edge("mask", "input_guard")
    g.add_conditional_edges("input_guard", lambda s: END if s["blocked"] else "tool_call", [END, "tool_call"])
    g.add_edge("tool_call", "retrieve")
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
