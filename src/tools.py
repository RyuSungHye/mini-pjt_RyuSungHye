"""도메인 도구: 실제 학원 조회 (공공데이터).

search_academies(동, 과목)
  data/academies_seongnam.csv(나이스 학원교습소정보, 성남시 전체)에서
  SERVICE.md 3장 검색 규칙대로 분당구 중학생 대상 학원을 찾습니다.

  - 지역: 도로명주소에 "분당구" 포함
  - 동: 도로명상세주소 괄호 안의 동 이름. 동을 알 수 없는 학원은 동으로 검색할 때만 제외
  - 운영 상태: 등록상태명 = "개원"
  - 중학생 대상: 인당수강료·교습과정목록명·교습과정명에 "중등" 또는 "중학" (초·중등 혼합 포함)
  - 과목: 학원명·교습과정명·인당수강료의 키워드로 판정. 키워드가 없으면 "정보 없음"(과목 검색 시 제외)
  - 수강료: 인당수강료 중 중등 항목만. 수강료공개여부가 Y가 아니거나 값이 없으면 "정보 없음"

CSV 파일은 수정하지 않고 읽기만 합니다. 결과는 학원명 가나다순이며 순위가 아닙니다.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "academies_seongnam.csv"

SUBJECT_KEYWORDS = {
    "수학": ["수학", "매스", "math"],
    "영어": ["영어", "잉글리시", "english"],
}
MIDDLE_WORDS = ("중등", "중학")
MAX_RESULTS = 15  # 모델에 넘기는 최대 개수 (전체 개수는 따로 알려줌)

DONG_IN_ADDRESS = re.compile(r"\(([가-힣0-9]+동)")


@lru_cache(maxsize=1)
def _load_rows() -> tuple[dict, ...]:
    """CSV를 한 번만 읽어 둡니다. 한글이 깨지면 cp949로 다시 읽습니다."""
    raw = CSV_PATH.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    return tuple(csv.DictReader(text.splitlines()))


def _dong(row: dict) -> str:
    found = DONG_IN_ADDRESS.search(row.get("도로명상세주소") or "")
    return found.group(1) if found else "정보 없음"


def _is_middle(row: dict) -> bool:
    return any(w in (row.get(col) or "")
               for col in ("인당수강료", "교습과정목록명", "교습과정명") for w in MIDDLE_WORDS)


def _subjects(row: dict) -> list[str]:
    text = " ".join(row.get(col) or "" for col in ("학원명", "교습과정명", "인당수강료")).lower()
    return [s for s, words in SUBJECT_KEYWORDS.items() if any(w in text for w in words)]


def _fee_items(row: dict) -> list[tuple[str, str]]:
    """'초등A:190000, 중등A:360000' -> [('초등A', '190000'), ('중등A', '360000')]"""
    items = []
    for part in (row.get("인당수강료") or "").split(","):
        label, _, amount = part.strip().partition(":")
        if label.strip():
            items.append((label.strip(), amount.strip()))
    return items


def _middle_fee(row: dict) -> str:
    if (row.get("수강료공개여부") or "").strip() != "Y":
        return "정보 없음"
    fees = [f"{label} {int(amount):,}원" for label, amount in _fee_items(row)
            if any(w in label for w in MIDDLE_WORDS) and amount.isdigit()]
    return ", ".join(fees) or "정보 없음"


def _target(row: dict) -> str:
    """인당수강료 항목 이름으로 초·중등 혼합 여부를 표시합니다."""
    labels = " ".join(label for label, _ in _fee_items(row))
    has_elem, has_high = "초등" in labels, "고등" in labels
    if has_elem and has_high:
        return "초·중·고 혼합"
    if has_elem:
        return "초·중등 혼합"
    if has_high:
        return "중·고등 혼합"
    return "중등"


def normalize_dong(name: str) -> str:
    """'정자' -> '정자동', '정자1동' -> '정자동' (주소에는 법정동 이름이 들어 있음)"""
    name = re.sub(r"\s+", "", name or "")
    name = re.sub(r"\d+동$", "동", name)
    return name if not name or name.endswith("동") else name + "동"


def search_academies_data(dong: list[str] | str | None = None, subject: str | None = None) -> dict:
    """검색 결과를 dict로 돌려줍니다. 에이전트는 이 함수를 직접 씁니다."""
    dongs = [dong] if isinstance(dong, str) else list(dong or [])
    dongs = [d for d in (normalize_dong(x) for x in dongs) if d]
    subject = subject if subject in SUBJECT_KEYWORDS else None

    matches = []
    for row in _load_rows():
        if "분당구" not in (row.get("도로명주소") or ""):
            continue
        if (row.get("등록상태명") or "").strip() != "개원":
            continue
        if not _is_middle(row):
            continue
        row_dong = _dong(row)
        if dongs and row_dong not in dongs:
            continue  # 동을 알 수 없는 학원("정보 없음")도 여기서 빠집니다
        subjects = _subjects(row)
        if subject and subject not in subjects:
            continue  # 과목 "정보 없음"인 학원도 여기서 빠집니다
        matches.append({
            "id": row.get("학원지정번호") or row.get("학원명"),
            "학원명": row.get("학원명", ""),
            "동": row_dong,
            "과목": "·".join(subjects) or "정보 없음",
            "교습과정": f"{row.get('교습과정명', '')} ({row.get('교습과정목록명', '')})",
            "대상": _target(row),
            "수강료(중등)": _middle_fee(row),
        })

    matches.sort(key=lambda m: m["학원명"])  # 가나다순. 순위가 아닙니다.
    return {
        "조건": {"동": dongs or "분당구 전체", "과목": subject or "전체"},
        "전체": len(matches),
        "결과": matches[:MAX_RESULTS],
    }


@tool
def search_academies(dong: list[str] | None = None, subject: str | None = None) -> str:
    """분당구 실제 학원(공공데이터: 나이스 학원교습소정보)을 동·과목으로 조회한다.
    중학생 대상, 운영 중(개원)인 학원만 돌려준다. 학원명·동·과목·교습과정·중등 수강료를 준다.
    반 이름·수업 시간·설명회 정보는 없다.

    Args:
        dong: 동 이름 목록. 예: ["수내동"]. "판교 쪽"처럼 넓은 지역은 ["판교동", "삼평동", "백현동", "운중동"].
              비우면 분당구 전체.
        subject: "수학" 또는 "영어". 비우면 과목 무관.
    """
    return json.dumps(search_academies_data(dong, subject), ensure_ascii=False)
