import json

data = json.load(open("hardened.json", encoding="utf-8"))

for test in data["results"]:
    if test["pass"] is False:
        print(test["id"], test["family"])
        print("Réponse :", test.get("answer"))
        print("Juge :", test.get("judge"))
        print("Outil :", test.get("tool_check"))
        print()