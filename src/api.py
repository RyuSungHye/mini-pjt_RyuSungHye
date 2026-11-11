"""POST /query API 서버.

실행 (프로젝트 폴더에서)
  .venv\\Scripts\\uvicorn src.api:app --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import run_query


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


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    return run_query(req.question)
