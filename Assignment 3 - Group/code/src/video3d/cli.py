"""One Fire CLI for data generation, training, evaluation and analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import fire
import chz
from video3d.config import apply_process_env, load_config, read_config


def config_for(overrides: tuple[str, ...], condition: str, env: str | None,
               keywords: dict[str, Any]) -> Any:
    extra = [f"{key}={value}" for key, value in keywords.items()]
    cfg = load_config(condition, env, list(overrides) + extra)
    apply_process_env(cfg)
    return cfg


def train(*overrides: str, condition: str = "c3", env: str | None = None,
          benchmark: bool = False, **keywords: Any) -> None:
    from video3d.train import train as run
    run(config_for(overrides, condition, env, keywords), benchmark=benchmark)


def train_json(path: str) -> None:
    from video3d.train import train as run
    cfg = read_config(path)
    apply_process_env(cfg)
    run(cfg)


def evaluate(run: str, *overrides: str, condition: str = "c3", env: str | None = None,
             k: int | None = None, shuffle_frames: bool = False, wrong_video: bool = False,
             repeated_reference: bool = False, split: str | None = None,
             limit: int | None = None, **keywords: Any) -> None:
    from video3d.evaluate import evaluate as execute
    execute(config_for(overrides, condition, env, keywords), run, k=k,
            shuffle=shuffle_frames, wrong_video=wrong_video, repeated_reference=repeated_reference,
            split=split, limit=limit)


def cache_latents(split: str, *overrides: str, condition: str = "c3",
                  env: str | None = None, **keywords: Any) -> None:
    from video3d.models.sam3d import build_sam3d
    from video3d.data.targets import cache_latents as execute
    from video3d.utils import pick_device
    cfg = config_for(overrides, condition, env, keywords)
    device = pick_device(cfg.device)
    execute(cfg, split, build_sam3d(cfg, device), device)


def inspect(*overrides: str, condition: str = "c3", env: str | None = None,
            **keywords: Any) -> None:
    print(json.dumps(chz.asdict(config_for(overrides, condition, env, keywords)), indent=2))


def check_env(*overrides: str, condition: str = "c3", env: str | None = None,
              offline: bool = False, load_models: bool = False, **keywords: Any) -> None:
    from video3d.check_env import check_env as execute
    execute(config_for(overrides, condition, env, keywords), network=not offline, load_models=load_models)


def validate_data(split: str, *overrides: str, condition: str = "c3", env: str | None = None,
                  num_projection_clips: int = 20, report: str | None = None,
                  **keywords: Any) -> None:
    from video3d.data.validate import validate_data as execute
    result = execute(config_for(overrides, condition, env, keywords), split, num_projection_clips)
    if report:
        Path(report).write_text(json.dumps(result, indent=2))
    if result["clips_with_problems"]:
        raise SystemExit(1)


def analyze(results: list[str], *overrides: str, baseline: str | None = None,
            out: str | None = None, condition: str = "c3", env: str | None = None,
            **keywords: Any) -> None:
    from video3d.analyze import analyze as execute
    execute(config_for(overrides, condition, env, keywords), results, baseline=baseline, out_dir=out)


def inspect_sam3d(*overrides: str, condition: str = "c3", env: str | None = None,
                  **keywords: Any) -> None:
    from video3d.inspect_sam3d import inspect_sam3d as execute
    execute(config_for(overrides, condition, env, keywords))


def frame_test(*overrides: str, condition: str = "c3", env: str | None = None,
               num_objects: int = 30, **keywords: Any) -> None:
    from video3d.inspect_sam3d import frame_test as execute
    execute(config_for(overrides, condition, env, keywords), num_objects=num_objects)


def generation(command: str, **options: Any) -> None:
    import importlib
    if command not in {"generate", "export", "assets", "assetfilter", "review", "t6"}:
        raise ValueError("Unknown generation command")
    module = importlib.import_module(f"video3d.data.generation.{command}")
    module.run(**options)


def scanned(command: str, **options: Any) -> None:
    from video3d.data.scan_prepare import prepare_gso, prepare_omniobject, prepare_available_scans
    from video3d.data.scanned import audit
    commands = {'prepare-omni': prepare_omniobject, 'prepare-gso': prepare_gso, 'audit': audit,
                'prepare-available': prepare_available_scans}
    if command not in commands:
        raise ValueError(f'Choose from {list(commands)}')
    print(json.dumps(commands[command](**options), indent=2))


def prepare(*overrides: str, condition: str = "c3", env: str | None = None,
            **keywords: Any) -> None:
    from video3d.prepare import prepare as execute
    execute(config_for(overrides,condition,env,keywords))


def evaluate_json(path: str, run: str, checkpoint: str | None = None,
                  epoch: int | None = None, trigger: str = 'ad_hoc', limit: int | None = None) -> None:
    from video3d.evaluate import evaluate as execute
    cfg = read_config(path)
    apply_process_env(cfg)
    execute(cfg, run, checkpoint=checkpoint, epoch=epoch, trigger=trigger, limit=limit)


def sweep(*overrides: str, condition: str = "c3", env: str | None = None,
          max_seconds: int = 4800, evaluate: bool = True, **keywords: Any) -> None:
    from video3d.sweep import sweep as execute
    execute(config_for(overrides, condition, env, keywords), max_seconds=max_seconds, evaluate=evaluate)


def main(argv: list[str] | None = None) -> None:
    fire.Fire({"train": train, "train-json": train_json, "eval-json": evaluate_json, "sweep": sweep, "prepare": prepare, "eval": evaluate,
               "cache-latents": cache_latents, "inspect": inspect, "check-env": check_env,
               "validate-data": validate_data, "analyze": analyze,
               "inspect-sam3d": inspect_sam3d, "frame-test": frame_test,
               "generation": generation, "scanned": scanned}, command=argv)


if __name__ == "__main__":
    main()
