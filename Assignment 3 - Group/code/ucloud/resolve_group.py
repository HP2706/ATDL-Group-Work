from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
import fire
from video3d.storage import atomic_json

PATH = Path("/work/hp-video/current-sweep.json")


def resolve() -> str:
    head = os.environ["HP_SOURCE_COMMIT"]
    previous = json.loads(PATH.read_text()) if PATH.is_file() else {}
    group = os.environ.get("HP_SWEEP_GROUP")
    if not group and previous.get("git_head") == head and not previous.get("completed", False):
        group = previous["group"]
    if not group:
        group = f"view-coverage-g{head[:12]}-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        raise ValueError("Unsafe run group")
    atomic_json(PATH, {"group": group, "git_head": head, "completed": False})
    return group


def profile_needed(group: str) -> bool:
    path = Path("/work/results/view_coverage/profile.json")
    if not path.is_file():
        return True
    record = json.loads(path.read_text())
    return record["group"] != group or record["git_head"] != os.environ["HP_SOURCE_COMMIT"]


def complete() -> None:
    state = json.loads(PATH.read_text())
    atomic_json(PATH, {**state, "completed": True})


if __name__ == "__main__":
    fire.Fire({"resolve": resolve, "complete": complete, "profile-needed": profile_needed})
