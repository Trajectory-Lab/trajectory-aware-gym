import json
from datetime import UTC, datetime
from pathlib import Path


def log_tool_call(path: str, tool: str, args: dict, result: dict):
    """Log a tool invocation to SQLite (``.db``) or JSONL (legacy)."""
    p = Path(path)
    timestamp = datetime.now(UTC).isoformat()

    if p.suffix == ".db":
        from trajectory_aware_gym.storage import save_tool_call_entry

        save_tool_call_entry(p, timestamp=timestamp, tool=tool, args=args, result=result)
        return

    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": timestamp, "tool": tool, "args": args, "result": result}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
