"""Run a vision sweep directly or submit it through the configured backend."""

import json
import os
import platform
import posixpath
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import fire
from pydantic import BaseModel, ConfigDict


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CODE_DIR.parent.parent / ".atdl-launch.local.toml"


class UCloudSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    gpu_template_job: str
    cpu_template_job: str
    ssh_key: str = "~/.ssh/id_ed25519"


class LaunchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["ucloud", "direct"]
    ucloud: UCloudSettings | None = None


def load_settings(environment: dict[str, str]) -> LaunchSettings:
    path = Path(environment.get("ATDL_LAUNCH_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Launch config not found: {path}; copy .atdl-launch.example.toml")
    values = tomllib.loads(path.read_text())
    if "ATDL_BACKEND" in environment:
        values["backend"] = environment["ATDL_BACKEND"]
    return LaunchSettings.model_validate(values)


def command_output(command: list[str], environment: dict[str, str], input_text: str | None = None) -> str:
    result = subprocess.run(command, input=input_text, capture_output=True, text=True, env=environment)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{shlex.join(command)} failed: {detail}")
    return result.stdout


def json_output(output: str) -> dict[str, Any]:
    start = output.find("{")
    if start < 0:
        raise ValueError(f"Expected JSON from UCloud: {output[:300]}")
    value = json.loads(output[start:])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from UCloud")
    return value


def shell_environment() -> dict[str, str]:
    if platform.system() != "Darwin":
        return os.environ.copy()
    result = subprocess.run(
        ["zsh", "-lc", "source ~/.zshrc >/dev/null 2>&1; env -0"],
        capture_output=True,
        check=True,
    )
    environment = os.environ.copy()
    for item in result.stdout.split(b"\0"):
        if item:
            name, value = item.decode().split("=", 1)
            environment[name] = value
    return environment


def ssh_connection(job_id: str, environment: dict[str, str]) -> tuple[str, str]:
    output = command_output(["ucloud", "jobs", "ssh", job_id, "--print-only"], environment)
    lines = [line for line in output.splitlines() if line.startswith("ssh ")]
    if len(lines) != 1:
        raise ValueError(f"No SSH endpoint for UCloud job {job_id}")
    parts = shlex.split(lines[0])
    return parts[1], parts[parts.index("-p") + 1]


def ssh_command(host: str, port: str, environment: dict[str, str], command: str, input_text: str | None = None) -> str:
    key = Path(environment.get("UCLOUD_SSH_KEY", "~/.ssh/id_ed25519")).expanduser()
    arguments = [
        "ssh", "-i", str(key), "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", port, host, command,
    ]
    return command_output(arguments, environment, input_text)


def wait_for_ssh(host: str, port: str, environment: dict[str, str]) -> None:
    key = Path(environment.get("UCLOUD_SSH_KEY", "~/.ssh/id_ed25519")).expanduser()
    command = [
        "ssh", "-i", str(key), "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-p", port, host, "true",
    ]
    deadline = time.monotonic() + 90
    while True:
        result = subprocess.run(command, capture_output=True, text=True, env=environment)
        if result.returncode == 0:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Staging job SSH was not ready after 90 seconds: {result.stderr.strip()}")
        time.sleep(3)


def staging_job(mount: str, environment: dict[str, str], settings: UCloudSettings) -> str:
    existing = environment.get("ATDL_STAGE_JOB")
    if existing:
        job = json_output(command_output(
            ["ucloud", "jobs", "status", existing, "--project", settings.project_id, "--output", "json"],
            environment,
        ))
        resources = job["specification"]["resources"]
        if job["status"]["state"] != "RUNNING" or not any(resource["path"] == mount for resource in resources):
            raise ValueError(f"ATDL_STAGE_JOB {existing} must be running with the configured work folder mounted")
        return existing
    name = f"atdl-code-stage-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    response = json_output(command_output(
        ["ucloud", "jobs", "submit", "--from", settings.cpu_template_job,
         "--project", settings.project_id, "--name", name, "--time", "00:10", "--mount", mount,
         "--execute", "--output", "json"],
        environment,
    ))
    job_id = str(response["responses"][0]["id"])
    for _ in range(60):
        status = json_output(command_output(
            ["ucloud", "jobs", "status", job_id, "--project", settings.project_id, "--output", "json"], environment
        ))["status"]["state"]
        if status == "RUNNING":
            print(f"Started 10-minute CPU staging job {job_id}", flush=True)
            return job_id
        if status in {"SUCCESS", "FAILURE", "EXPIRED"}:
            raise RuntimeError(f"CPU staging job {job_id} ended in state {status}")
        time.sleep(2)
    raise TimeoutError(f"CPU staging job {job_id} did not start within two minutes")


def batch_script(remote_root: PurePosixPath, launch_dir: PurePosixPath,
                 hours: int, sweep_args: tuple[str, ...]) -> str:
    if hours < 1:
        raise ValueError("UCloud reservation must be at least one hour")
    command = shlex.join([
        str(remote_root / ".venv" / "bin" / "python"),
        str(launch_dir / "code" / "launch.py"), "run", *sweep_args,
    ])
    return f"""#!/usr/bin/env bash
set -euo pipefail
credential_file={shlex.quote(str(launch_dir / "wandb.env"))}
if [[ ! -f "$credential_file" ]]; then
  echo "Missing one-time W&B credential file" >&2
  exit 1
fi
set -a
source "$credential_file"
set +a
rm -f "$credential_file"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd {shlex.quote(str(launch_dir))}
timeout --signal=INT --kill-after=600s {hours * 3600 - 1200}s {command}
"""


def add_default(arguments: tuple[str, ...], name: str, value: str) -> tuple[str, ...]:
    return arguments if any(arg.startswith(f"{name}=") for arg in arguments) else (*arguments, f"{name}={value}")


def argument_value(arguments: tuple[str, ...], name: str) -> str | None:
    values = [arg.split("=", 1)[1] for arg in arguments if arg.startswith(f"{name}=")]
    return values[-1] if values else None


def run(*sweep_args: str) -> None:
    arguments = tuple(map(str, sweep_args))

    if "task=translation" in arguments:
        arguments = tuple(
            arg for arg in arguments
            if arg != "task=translation"
        )

        fire_arguments = [
            f"--{arg}" if "=" in arg and not arg.startswith("--") else arg
            for arg in arguments
        ]

        os.execv(
            sys.executable,
            [
                sys.executable,
                str(CODE_DIR / "translation_launch.py"),
                "run",
                *fire_arguments,
            ],
        )

    os.execv(
        sys.executable,
        [
            sys.executable,
            str(CODE_DIR / "sweep.py"),
            *arguments,
        ],
    )

    
def submit_ucloud(script: Path, hours: int, sweep_args: tuple[str, ...],
                  settings: UCloudSettings, dry_run: bool, environment: dict[str, str]) -> None:
    environment["UCLOUD_SSH_KEY"] = settings.ssh_key
    hours = int(environment.get("UCLOUD_HOURS", hours))
    template_job = json_output(command_output(
        ["ucloud", "jobs", "status", settings.gpu_template_job,
         "--project", settings.project_id, "--output", "json"], environment
    ))
    resources = template_job["specification"]["resources"]
    if len(resources) != 1 or resources[0]["type"] != "file":
        raise ValueError("GPU template must mount exactly one persistent work folder")
    mount = str(resources[0]["path"])
    remote_root = PurePosixPath("/work") / PurePosixPath(mount).name
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{script.stem.replace('_', '-')}-{stamp}"
    launch_dir = remote_root / "launches" / run_name
    run_root = PurePosixPath(posixpath.normpath(
        argument_value(sweep_args, "run_root") or environment.get("RUN_ROOT")
        or str(remote_root / "runs" / "sweeps" / run_name)
    ))
    if remote_root not in run_root.parents:
        raise ValueError(f"RUN_ROOT must be under {remote_root}")
    sweep_args = add_default(sweep_args, "data_dir", environment.get("DATA_DIR", str(remote_root / "data")))
    sweep_args = add_default(sweep_args, "results_dir", environment.get("RESULTS_DIR", str(remote_root / "our-results-folder")))
    sweep_args = add_default(sweep_args, "run_root", str(run_root))
    stage_id = staging_job(mount, environment, settings)
    host, port = ssh_connection(stage_id, environment)
    wait_for_ssh(host, port, environment)
    ssh_command(host, port, environment, f"mkdir -p {shlex.quote(str(launch_dir))}")
    key = Path(environment.get("UCLOUD_SSH_KEY", "~/.ssh/id_ed25519")).expanduser()
    command_output(
        ["scp", "-r", "-i", str(key), "-o", "StrictHostKeyChecking=accept-new", "-P", port,
         str(CODE_DIR), f"{host}:{launch_dir}/"],
        environment,
    )
    wrapper = batch_script(remote_root, launch_dir, hours, sweep_args)
    ssh_command(host, port, environment,
                f"umask 077; cat > {shlex.quote(str(launch_dir / 'batch.sh'))}; chmod 700 {shlex.quote(str(launch_dir / 'batch.sh'))}",
                wrapper)
    plan = ssh_command(
        host, port, environment,
        f"bash -n {shlex.quote(str(launch_dir / 'batch.sh'))} && "
        f"{shlex.join([str(remote_root / '.venv' / 'bin' / 'python'), str(launch_dir / 'code' / 'launch.py'),
                         'run', *sweep_args, 'plan=true'])}",
    )
    print(plan.splitlines()[0])
    with tempfile.TemporaryDirectory(prefix="atdl-ucloud-") as temporary:
        payload_path = Path(temporary) / "job.json"
        command_output(
            ["ucloud", "jobs", "submit", "--from", settings.gpu_template_job,
             "--project", settings.project_id, "--name", run_name,
             "--time", f"{hours:02d}:00", "--mount", mount, "--write-payload", str(payload_path),
             "--output", "text"],
            environment,
        )
        payload = json.loads(payload_path.read_text())
        payload["items"][0]["parameters"]["batchScript"]["path"] = str(
            PurePosixPath(mount) / "launches" / run_name / "batch.sh"
        )
        payload_path.write_text(json.dumps(payload))
        command_output(
            ["ucloud", "jobs", "submit", "--payload-file", str(payload_path),
             "--project", settings.project_id, "--output", "text"], environment,
        )
        if dry_run:
            print(f"Submission preview: {run_name}, {hours} hours, {mount}")
            print(f"Run root: {run_root}")
            return
        token = environment.get("WANDB_API_KEY")
        if not token:
            raise ValueError("WANDB_API_KEY is not set in the sourced shell environment")
        ssh_command(
            host, port, environment,
            f"umask 077; cat > {shlex.quote(str(launch_dir / 'wandb.env'))}",
            f"WANDB_API_KEY={shlex.quote(token)}\n",
        )
        response = json_output(command_output(
            ["ucloud", "jobs", "submit", "--payload-file", str(payload_path), "--project", settings.project_id,
             "--execute", "--output", "json"],
            environment,
        ))
    job_id = str(response["responses"][0]["id"])
    print(f"Submitted UCloud job {job_id}: {run_name}")
    print(f"Run root: {run_root}")


def submit(script_path: str, hours: int, *sweep_args: str) -> None:
    script = Path(script_path).resolve()
    if script.parent != CODE_DIR / "scripts" or not script.is_file():
        raise ValueError(f"Expected an experiment script under {CODE_DIR / 'scripts'}: {script}")
    arguments = tuple(str(value) for value in sweep_args)
    if "plan=true" in arguments or "plan=True" in arguments:
        run(*arguments)
    dry_run = "submit_plan=true" in arguments
    arguments = tuple(value for value in arguments if value != "submit_plan=true")
    environment = shell_environment()
    settings = load_settings(environment)
    if settings.backend == "direct":
        if dry_run:
            run(*arguments, "plan=true")
        root = environment.get("ATDL_WORK_ROOT")
        if root:
            arguments = add_default(arguments, "data_dir", str(Path(root) / "data"))
            arguments = add_default(arguments, "results_dir", str(Path(root) / "our-results-folder"))
            arguments = add_default(arguments, "run_root", str(Path(root) / "runs" / "sweeps" /
                                                        f"{script.stem}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"))
        os.environ.update(environment)
        run(*arguments)
    if settings.ucloud is None:
        raise ValueError("The ucloud backend requires a [ucloud] section in the launch config")
    submit_ucloud(script, hours, arguments, settings.ucloud, dry_run, environment)


if __name__ == "__main__":
    fire.Fire({"submit": submit, "run": run})
