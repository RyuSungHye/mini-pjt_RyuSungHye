"""RAG 파이프라인: 학원 소개 문서를 읽어 조각내고, Chroma에 저장하고, 질문으로 검색합니다.

흐름
  data/academy_docs/*.md  ->  "## 섹션" 단위 조각  ->  Titan 임베딩  ->  Chroma(chroma_db/)
  질문  ->  Titan 임베딩  ->  가까운 조각 TOP_K개

문서 내용을 바꿨다면 chroma_db 폴더를 지우고 다시 실행하세요. 처음부터 새로 만듭니다.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "academy_docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "academy_docs"

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Day 2·6 실습과 같은 임베딩
REGION = "us-east-1"
TOP_K = 8  # 질문 하나에 가져올 조각 수 (10곳 x 5섹션 = 50조각 중)


def load_chunks() -> list[Document]:
    """학원 문서를 "## 섹션" 단위로 자릅니다.

    조각마다 맨 앞에 학원명·위치를 붙입니다. "반 구성" 조각만 따로 검색돼도
    어느 학원, 어느 동의 반인지 알 수 있게 하려는 것입니다.
    doc_id는 "파일이름#섹션" 형식입니다 (예: math_01_daongil#반 구성).
    """
    chunks: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        name = re.search(r"^# (.+)$", text, re.MULTILINE).group(1).strip()
        found = re.search(r"^- 위치: (.+)$", text, re.MULTILINE)
        location = found.group(1).strip() if found else "정보 없음"

        # 첫 조각(제목과 "가상 학원" 안내문)은 버리고 섹션만 씁니다.
        for part in re.split(r"(?m)^## ", text)[1:]:
            section, _, body = part.partition("\n")
            section = section.strip()
            chunks.append(Document(
                page_content=f"학원명: {name}\n위치: {location}\n섹션: {section}\n{body.strip()}",
                metadata={
                    "doc_id": f"{path.stem}#{section}",
                    "academy": name,
                    "section": section,
                    "source": "학원 소개 자료",
                },
            ))
    return chunks


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
    """질문과 가까운 문서 조각 k개를 돌려줍니다."""
    return get_vectorstore().similarity_search(question, k=k)
