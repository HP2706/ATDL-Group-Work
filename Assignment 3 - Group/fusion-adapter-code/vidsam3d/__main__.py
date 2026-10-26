"""CLI entrypoint: `python -m vidsam3d <command> [--condition c3] [--env ucloud] [key=value ...]`."""

from __future__ import annotations

import argparse
import sys

from vidsam3d.config import apply_process_env, load_config


def cmd_check_env(cfg, args) -> None:
    """Print GPU, versions, paths and HF access."""
    from vidsam3d.check_env import check_env
    check_env(cfg, network=not args.offline)


def cmd_cache_latents(cfg, args) -> None:
    """Encode GT voxels of a split with the (fake or real) SAM 3D VAE encoder."""
    from vidsam3d.data.targets import cache_latents
    from vidsam3d.models.sam3d import build_sam3d
    from vidsam3d.utils import pick_device, setup_logging
    setup_logging()
    device = pick_device(cfg.device)
    cache_latents(cfg, args.split, build_sam3d(cfg, device), device)


def cmd_train(cfg, args) -> None:
    """Train an adapter, or benchmark 20 steps."""
    from vidsam3d.train import train
    train(cfg, benchmark=args.benchmark)


def not_implemented(phase: str):
    """Return a handler for a command that arrives in a later phase."""
    def handler(cfg, args) -> None:
        sys.exit(f"`{args.command}` is not implemented yet (phase {phase}).")
    return handler


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with one subcommand per CLI command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--condition", help="configs/conditions/<name>.yaml")
    common.add_argument("--env", help="configs/env/<name>.yaml (default: $VIDSAM3D_ENV or local)")
    common.add_argument("--fake-sam3d", action="store_true", help="use FakeSam3D (dry runs)")
    common.add_argument("overrides", nargs="*", help="config overrides, e.g. optim.lr=3e-4")

    parser = argparse.ArgumentParser(prog="python -m vidsam3d")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-env", parents=[common], help="check GPU, paths, HF access")
    p.add_argument("--offline", action="store_true", help="skip network checks")
    p.set_defaults(func=cmd_check_env)

    p = sub.add_parser("inspect-sam3d", parents=[common], help="write docs/sam3d_internals.md")
    p.set_defaults(func=not_implemented("P5"))

    p = sub.add_parser("cache-latents", parents=[common], help="encode GT voxels with the SAM 3D VAE")
    p.add_argument("--split", required=True, help="train | eval | the val split name")
    p.set_defaults(func=cmd_cache_latents)

    p = sub.add_parser("frame-test", parents=[common], help="canonical-frame test T4")
    p.set_defaults(func=not_implemented("P5"))

    p = sub.add_parser("train", parents=[common], help="train an adapter")
    p.add_argument("--benchmark", action="store_true", help="20 steps; report s/step and memory")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", parents=[common], help="write per-sample results.jsonl")
    p.add_argument("--run", required=True, help="<run_dir> | zero-shot | best-frame")
    p.add_argument("--k", type=int, default=None, help="clip length (default: eval.k)")
    p.add_argument("--shuffle-frames", action="store_true")
    p.add_argument("--wrong-video", action="store_true")
    p.set_defaults(func=not_implemented("P6"))

    p = sub.add_parser("analyze", parents=[common], help="tables and plots")
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--baseline", default="c1")
    p.set_defaults(func=not_implemented("P6"))

    p = sub.add_parser("gen-data", parents=[common], help="run one shard of the external generator")
    p.add_argument("--split", required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.set_defaults(func=not_implemented("P7"))

    p = sub.add_parser("merge-manifests", parents=[common], help="merge shard manifests")
    p.add_argument("--split", required=True)
    p.set_defaults(func=not_implemented("P7"))

    p = sub.add_parser("validate-data", parents=[common], help="check files and run the projection check")
    p.add_argument("--split", required=True)
    p.set_defaults(func=not_implemented("P7"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, load the config and run the command."""
    args = build_parser().parse_args(argv)
    overrides = list(args.overrides) + (["fake_sam3d=true"] if args.fake_sam3d else [])
    cfg = load_config(args.condition, args.env, overrides)
    apply_process_env(cfg)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
