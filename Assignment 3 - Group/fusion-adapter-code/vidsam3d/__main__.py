"""CLI entrypoint: `python -m vidsam3d <command> [--condition c3] [--env ucloud] [key=value ...]`."""

from __future__ import annotations

import argparse
import sys

from vidsam3d.config import apply_process_env, load_config


def cmd_check_env(cfg, args) -> None:
    """Print GPU, versions, paths and HF access."""
    from vidsam3d.check_env import check_env
    check_env(cfg, network=not args.offline, load_models=args.load_models)


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


def cmd_inspect_sam3d(cfg, args) -> None:
    """Print SAM 3D runtime internals and write them to docs/sam3d_internals.md (GPU)."""
    from vidsam3d.inspect_sam3d import inspect_sam3d
    inspect_sam3d(cfg)


def cmd_frame_test(cfg, args) -> None:
    """T4: find the canonical-frame permutation between our voxels and SAM 3D (GPU)."""
    from vidsam3d.inspect_sam3d import frame_test
    frame_test(cfg, num_objects=args.num_objects)


def cmd_eval(cfg, args) -> None:
    """Evaluate a run directory, `zero-shot` (C0) or `best-frame` (C0b oracle)."""
    from vidsam3d.evaluate import evaluate
    evaluate(cfg, args.run, k=args.k, shuffle=args.shuffle_frames, wrong_video=args.wrong_video,
             split=args.split, limit=args.limit)


def cmd_analyze(cfg, args) -> None:
    """Tables and plots from results files."""
    from vidsam3d.analyze import analyze
    analyze(cfg, args.results, baseline=args.baseline, out_dir=args.out)


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
    p.add_argument("--load-models", action="store_true", help="also load SAM 3D and V-JEPA (GPU)")
    p.set_defaults(func=cmd_check_env)

    p = sub.add_parser("inspect-sam3d", parents=[common], help="write docs/sam3d_internals.md")
    p.set_defaults(func=cmd_inspect_sam3d)

    p = sub.add_parser("cache-latents", parents=[common], help="encode GT voxels with the SAM 3D VAE")
    p.add_argument("--split", required=True, help="train | eval | the val split name")
    p.set_defaults(func=cmd_cache_latents)

    p = sub.add_parser("frame-test", parents=[common], help="canonical-frame test T4")
    p.add_argument("--num-objects", type=int, default=30)
    p.set_defaults(func=cmd_frame_test)

    p = sub.add_parser("train", parents=[common], help="train an adapter")
    p.add_argument("--benchmark", action="store_true", help="20 steps; report s/step and memory")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", parents=[common], help="write per-sample results.jsonl")
    p.add_argument("--run", required=True, help="run dir or name (e.g. c3_seed0) | zero-shot | best-frame")
    p.add_argument("--k", type=int, default=None, help="clip length (default: eval.k)")
    p.add_argument("--shuffle-frames", action="store_true", help="H3 control")
    p.add_argument("--wrong-video", action="store_true", help="T7 control: video of another clip")
    p.add_argument("--split", default=None, help="default: data.eval_split")
    p.add_argument("--limit", type=int, default=None, help="evaluate at most N new samples")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("analyze", parents=[common], help="tables and plots")
    p.add_argument("--results", nargs="+", required=True, help="results .jsonl files")
    p.add_argument("--baseline", default=None, help="label to compare against (default: first)")
    p.add_argument("--out", default=None, help="default: output_root/analysis")
    p.set_defaults(func=cmd_analyze)

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
