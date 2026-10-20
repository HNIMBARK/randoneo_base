import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from agent import build_agent

load_dotenv()
ROOT = Path(__file__).resolve().parent


async def main():
    attacks = json.loads((ROOT / "evalutions/attacks.json").read_text(encoding="utf-8"))
    agent = await build_agent(
        MemorySaver(),
        customer_id="CUST-001"
        )
    results = []

    for attack in attacks:
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": attack["question"]}]},
                {"configurable": {"thread_id": uuid.uuid4().hex}}
            )
            messages = result.get("messages", [])
            calls = [
                {"name": c["name"], "args": c["args"]}
                for m in messages if m.type == "ai"
                for c in getattr(m, "tool_calls", [])
            ]
            results.append({
                "id": attack["id"],
                "answer": str(messages[-1].content) if messages else "",
                "tool_calls": calls,
                "interruptions": [i.value for i in result.get("__interrupt__", [])]
            })
        except Exception as e:
            results.append({"id": attack["id"], "error": str(e)})

    (ROOT / "attacks_before.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Résultats enregistrés dans attacks_before.json")


asyncio.run(main())