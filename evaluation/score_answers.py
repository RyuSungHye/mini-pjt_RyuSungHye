"""평가 ②: 저장된 답변 파일만으로 채점합니다. 에이전트를 다시 호출하지 않습니다.

1. LLM-as-Judge: 질문마다 expected_traits(모두 충족)와 forbidden(하나도 없음)을 판정합니다.
   전부 맞으면 통과. 질문별 결과를 저장하므로 멈춰도 이어서 실행할 수 있습니다.
2. RAGAS: context_recall, context_precision, faithfulness, answer_relevancy.
   reference가 사실 정답인 positive·edge 질문만 계산합니다.
   (negative·guardrail의 reference는 "안내합니다" 같은 행동 설명이라 RAGAS 점수가 의미 없습니다.)

- 채점 모델: Bedrock Claude Haiku 4.5 (에이전트보다 저렴한 모델)
- ThrottlingException이 나면 재시도하지 않고 바로 멈춥니다.
- 결과: evaluation/{round}_scores.json

실행 (프로젝트 폴더에서)
  .venv\\Scripts\\python -m evaluation.score_answers --round round1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from evaluation.generate_answers import is_throttle, save

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
QUERIES = EVAL_DIR / "test_queries.csv"

# .env의 JUDGE_MODEL_ID를 씁니다. 없으면 기본값.
# (.env는 위의 evaluation.generate_answers -> src.agent import 때 이미 읽힙니다)
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
# .env의 BEDROCK_REGION을 씁니다. 없으면 기본값.
REGION = os.getenv("BEDROCK_REGION", "us-east-1")
RAGAS_CATEGORIES = ("positive", "edge")

load_dotenv(ROOT / ".env")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ── 1. LLM-as-Judge (LCEL 구조화 출력) ─────────────────────────

class TraitCheck(BaseModel):
    item: str = Field(description="판정한 기대 특성 문구 그대로")
    met: bool = Field(description="답변이 이 특성을 충족하면 true")
    reason: str = Field(description="판단 근거 한 문장")


class ForbiddenCheck(BaseModel):
    item: str = Field(description="판정한 금지 항목 문구 그대로")
    violated: bool = Field(description="답변에 이 금지 항목이 나타나면 true")
    reason: str = Field(description="판단 근거 한 문장")


class JudgeResult(BaseModel):
    traits: list[TraitCheck]
    forbidden: list[ForbiddenCheck]


JUDGE_SYSTEM = """당신은 학원 검색 챗봇 답변을 채점하는 엄격한 평가자입니다.
[기대 특성]은 하나씩 충족 여부를, [금지 항목]은 하나씩 위반 여부를 판정합니다.
- 목록의 항목 수와 순서를 그대로 지켜 모든 항목을 판정합니다.
- [검색된 근거]와 [정답 예시]는 판단을 돕는 참고 자료입니다. 답변에 나온 학원·반·시간·교습비가
  근거에 없으면 "검색 결과에 없는 학원 이름"이나 "지어내기" 같은 금지 항목 위반으로 봅니다.
- 애매하면 충족하지 않은 것으로 판정합니다."""

JUDGE_HUMAN = """[질문]
{question}

[답변]
{answer}

[검색된 근거]
{contexts}

[정답 예시]
{reference}

[기대 특성]
{traits}

[금지 항목]
{forbidden}"""


def split_items(text: str) -> list[str]:
    return [s.strip() for s in (text or "").split(";") if s.strip()]


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(items, 1)) or "(없음)"


def build_judge():
    from langchain_aws import ChatBedrockConverse
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatBedrockConverse(model=JUDGE_MODEL_ID, region_name=REGION, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])
    return prompt | llm.with_structured_output(JudgeResult)


def judge_one(chain, row: dict, ans: dict) -> dict:
    traits, forbidden = split_items(row["expected_traits"]), split_items(row["forbidden"])
    if ans.get("error"):
        return {"passed": False, "traits": [], "forbidden": [], "note": f"답변 생성 에러: {ans['error']}"}

    contexts = "\n\n".join(f"[{c['doc_id']}]\n{c['text']}" for c in ans["contexts"]) or "(없음)"
    res: JudgeResult = chain.invoke({
        "question": row["input"], "answer": ans["answer"], "contexts": contexts,
        "reference": row.get("reference", ""), "traits": numbered(traits), "forbidden": numbered(forbidden),
    })
    note = ""
    if len(res.traits) != len(traits) or len(res.forbidden) != len(forbidden):
        note = "채점 모델이 항목 수를 다르게 돌려줌 (누락 항목은 실패로 처리)"
    t = [{"item": item, "met": res.traits[i].met if i < len(res.traits) else False,
          "reason": res.traits[i].reason if i < len(res.traits) else "판정 누락"} for i, item in enumerate(traits)]
    f = [{"item": item, "violated": res.forbidden[i].violated if i < len(res.forbidden) else True,
          "reason": res.forbidden[i].reason if i < len(res.forbidden) else "판정 누락"} for i, item in enumerate(forbidden)]
    passed = all(x["met"] for x in t) and not any(x["violated"] for x in f)
    return {"passed": passed, "traits": t, "forbidden": f, "note": note}


# ── 2. RAGAS ───────────────────────────────────────────────────

def run_ragas(rows: list[dict], answers: dict) -> dict:
    from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)

    targets = [r for r in rows if r["category"] in RAGAS_CATEGORIES and not answers[r["id"]].get("error")]
    dataset = EvaluationDataset.from_list([{
        "user_input": r["input"],
        "response": answers[r["id"]]["answer"],
        "retrieved_contexts": [c["text"] for c in answers[r["id"]]["contexts"]],
        "reference": r["reference"],
    } for r in targets])

    llm = LangchainLLMWrapper(ChatBedrockConverse(model=JUDGE_MODEL_ID, region_name=REGION, temperature=0))
    emb = LangchainEmbeddingsWrapper(BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION))
    result = evaluate(
        dataset,
        metrics=[LLMContextRecall(), LLMContextPrecisionWithReference(), Faithfulness(), ResponseRelevancy()],
        llm=llm, embeddings=emb,
        run_config=RunConfig(max_retries=1, max_workers=2, timeout=180),
        raise_exceptions=True,   # 한도 에러를 NaN으로 숨기지 않고 멈춥니다
        show_progress=False,
    )

    df = result.to_pandas()
    names = {"context_recall": "context_recall", "llm_context_precision_with_reference": "context_precision",
             "faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy"}
    per_q = {}
    for r, (_, rec) in zip(targets, df.iterrows()):
        per_q[r["id"]] = {short: (None if rec.get(col) is None or (isinstance(rec.get(col), float)
                                  and math.isnan(rec.get(col))) else round(float(rec.get(col)), 3))
                          for col, short in names.items()}
    means = {}
    for short in names.values():
        vals = [v[short] for v in per_q.values() if v[short] is not None]
        means[short] = round(sum(vals) / len(vals), 3) if vals else None
    return {"question_ids": [r["id"] for r in targets], "per_question": per_q, "mean": means}


# ── 실행 ──────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="round1")
    args = parser.parse_args()

    answers_path = EVAL_DIR / f"{args.round}_answers.json"
    scores_path = EVAL_DIR / f"{args.round}_scores.json"
    answers = json.loads(answers_path.read_text(encoding="utf-8"))["answers"]
    rows = list(csv.DictReader(QUERIES.open(encoding="utf-8-sig")))

    missing = [r["id"] for r in rows if r["id"] not in answers]
    if missing:
        print(f"답변이 아직 없는 질문: {missing}. 먼저 generate_answers를 끝까지 실행하세요.")
        return 1

    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}
    scores.setdefault("judge_model", JUDGE_MODEL_ID)
    judged = scores.setdefault("judge", {})

    chain = build_judge()
    for row in rows:
        qid = row["id"]
        if qid in judged:
            continue
        try:
            judged[qid] = {"category": row["category"], **judge_one(chain, row, answers[qid])}
        except Exception as exc:  # noqa: BLE001
            if is_throttle(exc):
                save(scores_path, scores)
                print(f"멈춤: Judge {qid}번에서 ThrottlingException. 판정 완료 {len(judged)}/{len(rows)}개.")
                return 2
            raise
        save(scores_path, scores)
        print(f"[{qid:>2}] Judge {'통과' if judged[qid]['passed'] else '실패'}")

    if "ragas" not in scores:
        print("RAGAS 계산 중...")
        try:
            scores["ragas"] = run_ragas(rows, answers)
        except Exception as exc:  # noqa: BLE001
            if is_throttle(exc):
                save(scores_path, scores)
                print("멈춤: RAGAS 계산 중 ThrottlingException. Judge 결과는 저장됨. 같은 명령으로 RAGAS만 다시 계산합니다.")
                return 2
            raise
        save(scores_path, scores)

    passed = sum(1 for v in judged.values() if v["passed"])
    print(f"\n통과 {passed}/{len(rows)}")
    for cat in ("positive", "negative", "edge", "guardrail"):
        ids = [r["id"] for r in rows if r["category"] == cat]
        ok = sum(1 for i in ids if judged[i]["passed"])
        print(f"  {cat:<9} {ok}/{len(ids)}")
    print("RAGAS 평균:", scores["ragas"]["mean"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
