"""평가 ①: test_queries.csv 질문마다 에이전트 답변을 한 번 만들어 저장합니다.

- 결과: evaluation/{round}_answers.json (UTF-8)
- 이미 저장된 질문은 건너뜁니다. 토큰 한도로 멈춰도 같은 명령으로 이어서 실행할 수 있습니다.
- ThrottlingException이 나면 재시도하지 않고 바로 멈춥니다.

실행 (프로젝트 폴더에서)
  .venv\\Scripts\\python -m evaluation.generate_answers --round round1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.agent import MODEL_ID, run_query

EVAL_DIR = Path(__file__).resolve().parent
QUERIES = EVAL_DIR / "test_queries.csv"


def is_throttle(exc: BaseException | None) -> bool:
    """예외(와 그 원인 예외)에 ThrottlingException이 들어 있는지 봅니다."""
    while exc is not None:
        if "Throttling" in type(exc).__name__ or "ThrottlingException" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def save(path: Path, data: dict) -> None:
    """임시 파일에 쓴 뒤 바꿔 끼웁니다. 쓰는 도중 멈춰도 기존 파일이 깨지지 않습니다."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="round1")
    args = parser.parse_args()

    out = EVAL_DIR / f"{args.round}_answers.json"
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"model": MODEL_ID, "answers": {}}
    answers = data["answers"]

    rows = list(csv.DictReader(QUERIES.open(encoding="utf-8-sig")))
    for row in rows:
        qid = row["id"]
        if qid in answers:
            print(f"[{qid:>2}] 건너뜀 (이미 저장됨)")
            continue

        try:
            result = run_query(row["input"])
            answers[qid] = {"question": row["input"], "category": row["category"], **result, "error": ""}
            print(f"[{qid:>2}] 저장")
        except Exception as exc:  # noqa: BLE001 - 어떤 에러든 기록하거나 멈춥니다
            if is_throttle(exc):
                save(out, data)
                print(f"\n멈춤: {qid}번 질문에서 ThrottlingException (토큰 한도).")
                print(f"저장된 답변 {len(answers)}/{len(rows)}개. 잠시 뒤 같은 명령으로 이어서 실행하세요.")
                return 2
            # 한도 외 에러는 실패한 답변으로 기록합니다 (채점에서 실패로 처리).
            answers[qid] = {"question": row["input"], "category": row["category"],
                            "answer": "", "contexts": [], "trace": [],
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(f"[{qid:>2}] 에러 기록: {answers[qid]['error']}")

        data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save(out, data)

    print(f"\n완료: {len(answers)}/{len(rows)}개 저장 -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
