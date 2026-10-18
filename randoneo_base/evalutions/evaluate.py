
import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import build_agent
from judge import judge_response

load_dotenv()

HERE = Path(__file__).resolve().parent


def load_cases():
    return json.loads(
        (HERE / "test_cases.json").read_text(
            encoding="utf-8"
        )
    )


def extract_answer(messages):
    for message in reversed(messages):
        if (
            message.type == "ai"
            and isinstance(message.content, str)
            and message.content
        ):
            return message.content

    return ""


def extract_tool_calls(messages):
    calls = []

    for message in messages:
        if message.type == "ai":
            for call in getattr(message, "tool_calls", []):
                calls.append({
                    "name": call["name"],
                    "args": call["args"]
                })

    return calls


async def evaluate_case(agent, case):
    config = {
        "configurable": {
            "thread_id": (
                f"eval-{case['id']}-"
                f"{uuid.uuid4().hex}"
            )
        }
    }

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": case["question"]
                }
            ]
        },
        config=config
    )

    interrupted = bool(result.get("__interrupt__"))
    messages = result.get("messages", [])

    answer = extract_answer(messages)
    calls = extract_tool_calls(messages)

    # Un test ne doit jamais approuver automatiquement
    # une action sensible.
    if interrupted:
        answer += "\n[Action en attente de validation]"

    judgement = await judge_response(
        case["question"],
        answer,
        case["expected"],
        case.get("reference_facts")
    )

    expected_tools = case.get("expected_tools", [])

    # Indicateur de parcours : indépendant du juge.
    tool_check = (
        not expected_tools
        or any(
            c["name"] in expected_tools
            for c in calls
        )
    )

    return {
        "id": case["id"],
        "family": case["family"],
        "question": case["question"],
        "answer": answer,
        "tool_calls": calls,
        "interrupted": interrupted,
        "tool_check": tool_check,
        "judge": judgement,
        "pass": (
            judgement["verdict"] == "pass"
            and tool_check
        )
    }


async def main(mode):
    cases = load_cases()

    agent = await build_agent(
        MemorySaver()
    )

    results = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}")

        try:
            result = await evaluate_case(agent, case)

        except Exception as error:
            result = {
                "id": case["id"],
                "family": case["family"],
                "pass": False,
                "error": str(error)
            }

        results.append(result)

    passed = sum(
        result["pass"] is True
        for result in results
    )

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(
            100 * passed / len(results), 2
        ) if results else 0
    }

    output = {
        "mode": mode,
        "summary": summary,
        "results": results
    }

    path = HERE / (
        "baseline.json" if mode == "before"
        else "after.json"
    )

    path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Enregistré : {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["before", "after"],
        required=True
    )

    args = parser.parse_args()

    asyncio.run(main(args.mode))
