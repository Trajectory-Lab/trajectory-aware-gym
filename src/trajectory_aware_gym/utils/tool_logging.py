import json
from datetime import datetime
from pathlib import Path


def log_tool_call(path: str, tool: str, args: dict, result: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "tool": tool,
        "args": args,
        "result": result,
    }

    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
