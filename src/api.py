"""POST /query API 서버.

실행 (프로젝트 폴더에서)
  .venv\\Scripts\\uvicorn src.api:app --port 8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agent import mask_pii_report, run_query


class QueryRequest(BaseModel):
    question: str


class Context(BaseModel):
    doc_id: str
    text: str


class TraceStep(BaseModel):
    step: str
    input: str
    output: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[Context]
    trace: list[TraceStep]


app = FastAPI(title="분당구 학원 찾기 도우미")

DEMO_PAGE = Path(__file__).resolve().parent / "demo.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    # 시연용 화면: 질문을 입력하면 이 페이지가 POST /query를 불러 결과를 보여줍니다.
    return DEMO_PAGE.read_text(encoding="utf-8")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    # 그래프를 부르기 전에 가립니다. 그래프 입력이 LangSmith에 기록되므로, 여기서 가려야 원문이 남지 않습니다.
    masked, counts = mask_pii_report(req.question)
    return run_query(masked, mask_counts=counts)
