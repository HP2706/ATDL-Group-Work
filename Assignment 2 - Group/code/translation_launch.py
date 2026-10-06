from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import fire


CODE_DIR = Path(__file__).resolve().parent


def _parse_int_list(value) -> list[int]:
    if isinstance(value, (tuple, list)):
        return [int(part) for part in value]

    return [
        int(part.strip())
        for part in str(value).split(",")
        if part.strip()
    ]

def run(
    data_dir: str,
    output_dir: str,
    results_dir: str,
    embedding_dim: int = 16,
    sample_count: int = 4000,
    embedding_dims: str = "",
    sample_counts: str = "",
    max_steps: int = 10,
    warmup_steps: int = 4000,
    eval_every: int = 10,
    checkpoint_every: int = 10,
    seed: int = 0,
    device: str = "cuda",
    run_root: str = "",
    resume_existing: bool = False,
    plan: bool = False,
) -> None:
    widths = (
        _parse_int_list(embedding_dims)
        if embedding_dims
        else [embedding_dim]
    )

    samples = (
        _parse_int_list(sample_counts)
        if sample_counts
        else [sample_count]
    )

    configurations = [
        (width, samples_n)
        for samples_n in samples
        for width in widths
    ]

    if plan:
        print(
            f"Translation sweep: "
            f"{len(configurations)} configurations, "
            f"widths={widths}, "
            f"samples={samples}, "
            f"steps={max_steps}, "
            f"seed={seed}, "
            f"device={device}"
        )
        return

    for index, (width, samples_n) in enumerate(configurations, start=1):
        print()
        print("=" * 80)
        print(
            f"Translation configuration {index}/{len(configurations)}: "
            f"d={width}, samples={samples_n}, steps={max_steps}"
        )
        print("=" * 80)
        print()

        config_output_dir = (
            Path(output_dir)
            / f"n{samples_n}"
        )

        command = [
            sys.executable,
            str(CODE_DIR / "translation_train.py"),
            f"--data_dir={data_dir}",
            f"--output_dir={config_output_dir}",
            f"--results_dir={results_dir}",
            f"--embedding_dim={width}",
            f"--sample_count={samples_n}",
            f"--max_steps={max_steps}",
            f"--warmup_steps={warmup_steps}",
            f"--eval_every={eval_every}",
            f"--checkpoint_every={checkpoint_every}",
            f"--seed={seed}",
            f"--device={device}",
        ]

        if resume_existing:
            candidates = sorted(
                config_output_dir.glob(
                    f"iwslt14-de-en-d{width}-seed{seed}-*"
                )
            )

            candidates = [
                path
                for path in candidates
                if (path / "config.json").exists()
                and (path / "checkpoint.pt").exists()
            ]

            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one resumable run for "
                    f"n={samples_n}, d={width}, found {len(candidates)}: "
                    f"{candidates}"
                )

            command.append(f"--resume={candidates[0]}")

        subprocess.run(
            command,
            check=True,
            env=os.environ.copy(),
        )


if __name__ == "__main__":
    fire.Fire({"run": run})