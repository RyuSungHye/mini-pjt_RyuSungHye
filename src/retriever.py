"""RAG 파이프라인: 학원 소개 문서를 읽어 조각내고, Chroma에 저장하고, 질문으로 검색합니다.

흐름
  data/academy_docs/*.md  ->  "## 섹션" 단위 조각  ->  Titan 임베딩  ->  Chroma(chroma_db/)
  질문  ->  (동·과목 필터)  ->  가까운 조각 TOP_K개  ->  학원 단위 확장

5단계 검색 개선
  ① 학원 단위 확장: 검색에 걸린 조각의 학원은 그 학원의 모든 섹션(반 구성 포함)을 함께 돌려줍니다.
     조각 하나만 걸려서 "반 구성"이 빠지는 문제를 막습니다.
  ② 동·과목 필터: 질문에 동이나 과목이 있으면 그 조건에 맞는 학원 문서 안에서만 검색합니다.
     의미 검색이 "수내동" 같은 조건을 걸러 주지 못하는 문제를 막습니다.

문서 내용을 바꿨다면 chroma_db 폴더를 지우고 다시 실행하세요. 처음부터 새로 만듭니다.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "academy_docs"
CHROMA_DIR = ROOT / "chroma_db"
# v2: 조각에 동·과목 메타데이터를 추가했습니다. 이름이 바뀌어 처음 실행할 때 새로 만듭니다.
COLLECTION = "academy_docs_v2"

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Day 2·6 실습과 같은 임베딩
# .env의 BEDROCK_REGION을 씁니다. 없으면 기본값. (.env는 src.agent가 먼저 읽습니다)
REGION = os.getenv("BEDROCK_REGION", "us-east-1")
TOP_K = 8          # 질문 하나에 가져올 조각 수 (10곳 x 5섹션 = 50조각 중)
MAX_ACADEMIES = 5  # 학원 단위로 확장할 때 최대 학원 수 (검색 순위가 높은 학원부터)

DONG_IN_LOCATION = re.compile(r"분당구\s+([가-힣0-9]+동)")


def load_chunks() -> list[Document]:
    """학원 문서를 "## 섹션" 단위로 자릅니다.

    조각마다 맨 앞에 학원명·위치를 붙입니다. "반 구성" 조각만 따로 검색돼도
    어느 학원, 어느 동의 반인지 알 수 있게 하려는 것입니다.
    doc_id는 "파일이름#섹션" 형식입니다 (예: math_01_daongil#반 구성).
    메타데이터에 동(dong)과 과목(subject_math, subject_english)을 넣어 필터 검색에 씁니다.
    """
    chunks: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        name = re.search(r"^# (.+)$", text, re.MULTILINE).group(1).strip()
        found = re.search(r"^- 위치: (.+)$", text, re.MULTILINE)
        location = found.group(1).strip() if found else "정보 없음"
        dong = DONG_IN_LOCATION.search(location)
        found = re.search(r"^- 과목: (.+)$", text, re.MULTILINE)
        subjects = found.group(1) if found else ""

        # 첫 조각(제목과 "가상 학원" 안내문)은 버리고 섹션만 씁니다.
        for part in re.split(r"(?m)^## ", text)[1:]:
            section, _, body = part.partition("\n")
            section = section.strip()
            chunks.append(Document(
                page_content=f"학원명: {name}\n위치: {location}\n섹션: {section}\n{body.strip()}",
                metadata={
                    "doc_id": f"{path.stem}#{section}",
                    "academy_id": path.stem,
                    "academy": name,
                    "section": section,
                    "source": "학원 소개 자료",
                    "dong": dong.group(1) if dong else "정보 없음",
                    "subject_math": "수학" in subjects,
                    "subject_english": "영어" in subjects,
                },
            ))
    return chunks


@lru_cache(maxsize=1)
def _chunks_by_academy() -> dict[str, list[Document]]:
    """학원 id -> 그 학원의 모든 조각 (문서 순서: 기본 정보, 반 구성, 설명회, 특장점, 확인 필요)"""
    groups: dict[str, list[Document]] = {}
    for c in load_chunks():
        groups.setdefault(c.metadata["academy_id"], []).append(c)
    return groups


_store: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Chroma 저장소를 엽니다. 비어 있거나 문서 목록이 바뀌었으면 다시 채웁니다."""
    global _store
    if _store is not None:
        return _store

    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    chunks = load_chunks()
    ids = [c.metadata["doc_id"] for c in chunks]
    existing = set(store.get(include=[])["ids"])
    if existing != set(ids):
        if existing:
            store.delete(ids=list(existing))
        store.add_documents(chunks, ids=ids)

    _store = store
    return store


def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    """질문과 가까운 문서 조각 k개를 돌려줍니다. (1단계 방식. 확장·필터 없음)"""
    return get_vectorstore().similarity_search(question, k=k)


def build_filter(dongs: list[str] | None = None, subject: str | None = None) -> dict | None:
    """동·과목 조건을 Chroma 메타데이터 필터로 바꿉니다. 조건이 없으면 None."""
    conds: list[dict] = []
    if dongs:
        conds.append({"dong": {"$in": list(dongs)}})
    if subject == "수학":
        conds.append({"subject_math": True})
    elif subject == "영어":
        conds.append({"subject_english": True})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def retrieve_academies(question: str, dongs: list[str] | None = None, subject: str | None = None,
                       k: int = TOP_K) -> dict:
    """개선된 검색. ② 필터 -> 가까운 조각 k개 -> ① 학원 단위 확장.

    Returns:
        hits: 검색에 걸린 조각 (필터 적용 후)
        academies: 확장한 학원 id 목록 (검색 순위 순, 최대 MAX_ACADEMIES곳)
        docs: 확장된 조각 전체 (모델에 넘길 근거)
        filter: 실제로 쓴 Chroma 필터 (없으면 None)
    """
    flt = build_filter(dongs, subject)
    store = get_vectorstore()
    hits = store.similarity_search(question, k=k, filter=flt) if flt else store.similarity_search(question, k=k)
    academies = list(dict.fromkeys(h.metadata["academy_id"] for h in hits))[:MAX_ACADEMIES]
    docs = [c for a in academies for c in _chunks_by_academy()[a]]
    return {"hits": hits, "academies": academies, "docs": docs, "filter": flt}
