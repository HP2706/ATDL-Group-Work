"""Config merge order, struct mode, env interpolation, saving, and the check-env command."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue

from vidsam3d.__main__ import main
from vidsam3d.config import CONFIG_DIR, load_config, save_config


def test_base_only_defaults():
    cfg = load_config(env="local")
    assert cfg.condition == "base" and cfg.env == "local"
    assert cfg.adapter.n_blocks == 2 and cfg.encoder.kv_source == "vjepa"


@pytest.mark.parametrize("name, kv_source, obj_emb, pair_avg", [
    ("c1", "dinov2_ref", True, False),
    ("c2", "dinov2_clip", True, True),
    ("c3", "vjepa", True, False),
    ("c3_nomask", "vjepa", False, False),
])
def test_conditions(name, kv_source, obj_emb, pair_avg):
    cfg = load_config(name, "local")
    assert cfg.encoder.kv_source == kv_source
    assert cfg.adapter.use_object_embedding is obj_emb
    assert cfg.encoder.dinov2.pair_average is pair_avg
    assert cfg.run_name == f"{name}_seed0"


def test_merge_order(tmp_path):
    """base <- condition <- env <- CLI: each later layer wins."""
    for sub in ("conditions", "env"):
        (tmp_path / sub).mkdir()
    (tmp_path / "base.yaml").write_text((CONFIG_DIR / "base.yaml").read_text())
    (tmp_path / "conditions" / "x.yaml").write_text("seed: 1\noptim:\n  lr: 0.5\n")
    (tmp_path / "env" / "e.yaml").write_text(
        (CONFIG_DIR / "env" / "local.yaml").read_text() + "optim:\n  lr: 0.25\n")
    cfg = load_config("x", "e", config_dir=tmp_path)
    assert cfg.seed == 1 and cfg.optim.lr == 0.25          # env beats condition
    cfg = load_config("x", "e", ["optim.lr=0.125"], config_dir=tmp_path)
    assert cfg.optim.lr == 0.125                           # CLI beats env


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigKeyError):
        load_config("c3", "local", ["adapter.nblocks=3"])


def test_missing_condition_file():
    with pytest.raises(FileNotFoundError, match="available"):
        load_config("c9", "local")


def test_mandatory_values_fail_loudly():
    cfg = load_config("c3", "local")
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.adapter.query_sets


def test_env_from_environment_and_path_interpolation(monkeypatch):
    monkeypatch.setenv("VIDSAM3D_ENV", "ucloud")
    monkeypatch.setenv("PROJECT_DIR", "/work/myproj")
    monkeypatch.setenv("DATA_ROOT", "/work/elsewhere/dataset")
    cfg = load_config("c3")
    assert cfg.env == "ucloud"
    assert cfg.paths.data_root == "/work/elsewhere/dataset"
    assert cfg.paths.output_root == "/work/myproj/outputs"


@pytest.mark.parametrize("env", ["local", "ucloud", "gcp"])
def test_every_env_sets_every_path(env):
    cfg = load_config(env=env)
    for key in cfg.paths:
        assert isinstance(cfg.paths[key], str) and cfg.paths[key]


def test_save_config_roundtrip(tmp_path):
    cfg = load_config("c2", "local", ["seed=3"])
    path = save_config(cfg, tmp_path / "run")
    loaded = OmegaConf.load(path)
    assert loaded.seed == 3 and loaded.run_name == "c2_seed3"
    assert loaded.paths.data_root == cfg.paths.data_root   # interpolations resolved


def test_check_env_cli_runs(capsys):
    main(["check-env", "--env", "local", "--offline"])
    out = capsys.readouterr().out
    assert "paths.data_root" in out and "failed" in out


def test_gen_data_needs_a_shard(monkeypatch):
    monkeypatch.delenv("SHARD_ID", raising=False)
    with pytest.raises(SystemExit, match="shard"):
        main(["gen-data", "--env", "local", "--split", "train", "--num-shards", "2"])


def test_torch_kernel_compatibility_rule():
    from vidsam3d.check_env import torch_supports
    cu121 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert torch_supports((9, 0), cu121) and torch_supports((8, 9), cu121)   # H100, L40S
    assert not torch_supports((10, 0), cu121)                               # B200: no kernels
    assert torch_supports((10, 0), cu121 + ["sm_100"])
    assert torch_supports((10, 0), cu121 + ["compute_90"])                  # PTX JIT
