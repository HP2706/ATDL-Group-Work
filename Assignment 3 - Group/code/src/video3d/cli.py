from __future__ import annotations

import json
from pathlib import Path
import chz
import fire
from .config import ExperimentConfig


def resolved_config(payload: dict[str, object]) -> ExperimentConfig:
    flat = {f"{group}.{key}": value for group, fields in payload.items() for key, value in fields.items()}
    return chz.Blueprint(ExperimentConfig).apply(flat).make()


def train_json(path: str) -> None:
    from .train import train
    train(resolved_config(json.loads(Path(path).read_text())))


def sweep(*overrides: str) -> None:
    from .sweep import sweep as run
    run(chz.Blueprint(ExperimentConfig).apply_from_argv(list(overrides)).make())


def prepare(*overrides: str) -> None:
    from .prepare import prepare as run
    run(chz.Blueprint(ExperimentConfig).apply_from_argv(list(overrides)).make())


def profile(*overrides: str) -> None:
    from .profile import profile as run
    run(chz.Blueprint(ExperimentConfig).apply_from_argv(list(overrides)).make())


def inspect(*overrides: str) -> None:
    print(json.dumps(chz.asdict(chz.Blueprint(ExperimentConfig).apply_from_argv(list(overrides)).make()), indent=2))


def main() -> None:
    fire.Fire({"train-json": train_json, "sweep": sweep, "prepare": prepare, "profile": profile, "inspect": inspect})


if __name__ == "__main__":
    main()
