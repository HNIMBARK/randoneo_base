
import argparse
import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from agent import build_agent
from judge import judge_response

load_dotenv()
ROOT = Path(__file__).resolve().parent


async def main(mode):
    cases = json.loads((ROOT / "evalutions/cases.json").read_text(encoding="utf-8"))

    agent = await build_agent(MemorySaver(), customer_id="CUST-001")
    results = []

    for case in cases:
        print(f"Test : {case['id']}")

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": case["question"]}]},
                config={"configurable": {"thread_id": uuid.uuid4().hex}}
            )

            messages = result.get("messages", [])
            tool_results = [
                {"name": m.name, "content": m.content}
                for m in messages if m.type == "tool"
            ]
            answer = next(
                (m.content for m in reversed(messages)
                 if m.type == "ai" and isinstance(m.content, str) and m.content),
                ""
            )
            calls = [
                {"name": c["name"], "args": c["args"]}
                for m in messages if m.type == "ai"
                for c in getattr(m, "tool_calls", [])
            ]

            expected_tools = case.get("expected_tools", [])
            tool_check = all(
                name in [c["name"] for c in calls]
                for name in expected_tools
            )

            interrupted = bool(result.get("__interrupt__"))
            judgment = await judge_response(
                case["question"], answer, case["expected"],
                {
                "reference_facts": case.get("reference_facts"),
                "tool_results": tool_results
            })

            results.append({
                "id": case["id"],
                "family": case["family"],
                "answer": answer,
                "tool_calls": calls,
                "interruptions": [
                    i.value for i in result.get("__interrupt__", [])
                ],
                "judge": judgment,
                "tool_check": tool_check,
                "pass": judgment["verdict"] == "pass" and tool_check
            })

        except Exception as exc:
            results.append({
                "id": case["id"],
                "family": case["family"],
                "error": str(exc),
                "pass": None
            })

    passed = sum(r["pass"] is True for r in results)
    failed = sum(r["pass"] is False for r in results)

    output = {
        "mode": mode,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "ungraded": len(results) - passed - failed,
            "pass_rate": round(100 * passed / len(results), 2) if results else 0
        },
        "results": results
    }

    filename = "baseline.json" if mode == "before" else "hardened.json"
    (ROOT / filename).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after"], required=True)
    asyncio.run(main(parser.parse_args().mode))