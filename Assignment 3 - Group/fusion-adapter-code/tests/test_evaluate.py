"""Evaluation with FakeSam3D: baselines, runs, test-time flags, resume, identity, analysis."""

import shutil

import pytest
import torch

from tests.fixtures.fake_config import fake_train_cfg
from vidsam3d.analyze import analyze
from vidsam3d.data import reader
from vidsam3d.data.samples import ClipDataset
from vidsam3d.evaluate import evaluate
from vidsam3d.models.adapter import build_adapter
from vidsam3d.train import train
from vidsam3d.utils import read_jsonl

pytest.importorskip("pytorch3d")
METRIC_KEYS = {"viou", "viou_aligned", "chamfer", "fscore@0.02", "fscore@0.05"}


@pytest.fixture(scope="module")
def trained(fake_root, tmp_path_factory):
    """A 3-step c3 run on the fake dataset; returns (cfg, run_dir)."""
    tmp = tmp_path_factory.mktemp("eval")
    cfg = fake_train_cfg(fake_root, tmp, **{"train.max_steps": 3, "data.val_split": "eval"})
    train(cfg)
    return cfg, tmp / "out" / "c3_seed0"


def test_eval_run_writes_one_line_per_sample_and_resumes(trained):
    cfg, run_dir = trained
    n = len(ClipDataset(cfg, "eval", train=False))
    path = evaluate(cfg, str(run_dir))
    records = read_jsonl(path)
    assert path.name == "c3_seed0.jsonl" and len(records) == n
    assert METRIC_KEYS <= set(records[0]) and records[0]["condition"] == "c3"
    assert {"visible_frac", "amodal_px", "arc_span_deg", "num_objects", "asset_id", "source"} <= set(records[0])
    evaluate(cfg, str(run_dir))                              # resume: nothing new
    assert len(read_jsonl(path)) == n


def test_eval_limit_then_resume(trained):
    cfg, run_dir = trained
    path = evaluate(cfg, str(run_dir), k=4, limit=2)
    assert path.name == "c3_seed0_k4.jsonl" and len(read_jsonl(path)) == 2
    evaluate(cfg, str(run_dir), k=4)
    assert len({r["sample_id"] for r in read_jsonl(path)}) == len(read_jsonl(path)) == 5


def test_baselines(trained):
    cfg, _ = trained
    c0 = read_jsonl(evaluate(cfg, "zero-shot"))
    c0b = read_jsonl(evaluate(cfg, "best-frame"))
    assert c0[0]["label"] == c0[0]["condition"] == "c0"
    assert c0b[0]["label"] == c0b[0]["condition"] == "c0b_oracle"
    for r in c0b:                                            # oracle frame = max visible_frac over S
        clip = reader.load_clip(cfg.paths.data_root, {"clip_id": r["clip_id"], "split": "eval",
                                                       "path": f"clips/eval/{r['clip_id']}"})
        vis = reader.visibility(clip)[:, r["k"] - 1]
        assert vis[r["oracle_frame"]] == max(vis[t] for t in range(0, 32, 2))


def test_untrained_adapter_equals_zero_shot(trained, tmp_path):
    """T1 on CPU: zero gates -> the adapted pipeline reproduces SAM 3D sample by sample."""
    cfg, run_dir = trained
    fresh = tmp_path / "fresh_run"
    shutil.copytree(run_dir, fresh)
    cfg_text = (fresh / "config.yaml").read_text().replace("run_name: c3_seed0", "run_name: fresh")
    (fresh / "config.yaml").write_text(cfg_text)
    torch.save({"ema": build_adapter(cfg.adapter).state_dict()}, fresh / "best.pt")
    ours = {r["sample_id"]: r for r in read_jsonl(evaluate(cfg, str(fresh)))}
    base = {r["sample_id"]: r for r in read_jsonl(evaluate(cfg, "zero-shot"))}
    assert ours.keys() == base.keys()
    assert all(ours[s]["viou"] == base[s]["viou"] for s in ours)


def test_shuffle_and_wrong_video_controls(trained):
    cfg, run_dir = trained
    shuffled = read_jsonl(evaluate(cfg, str(run_dir), shuffle=True))
    wrong = read_jsonl(evaluate(cfg, str(run_dir), wrong_video=True))
    assert shuffled[0]["label"] == "c3_seed0_shuffle" and shuffled[0]["shuffle_frames"]
    assert wrong[0]["label"] == "c3_seed0_wrongvideo"
    assert all(r["video_from"].split("/")[0] != r["clip_id"] for r in wrong)


def test_analyze_writes_tables_and_plots(trained, tmp_path):
    cfg, run_dir = trained
    paths = [evaluate(cfg, "zero-shot"), evaluate(cfg, str(run_dir)), evaluate(cfg, str(run_dir), k=4)]
    out = analyze(cfg, [str(p) for p in paths], baseline="c0", out_dir=tmp_path / "analysis")
    assert (out / "summary.md").is_file() and (out / "delta_vs_c0.csv").is_file()
    assert (out / "viou_vs_visible_frac.png").is_file() and (out / "viou_vs_k.png").is_file()
    rows = (out / "delta_vs_c0.csv").read_text().splitlines()
    assert rows[0] == "label,vs,metric,n,mean_delta,ci_lo,ci_hi,wilcoxon_p" and len(rows) > 1
