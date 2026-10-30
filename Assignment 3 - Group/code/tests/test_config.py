"""Typed presets, unknown-key rejection, immutable updates and JSON persistence."""
import typing
import json
from pathlib import Path
import chz
import pytest
from video3d.cli import main
from video3d.config import load_config, read_config, replace_config, save_config

@pytest.mark.parametrize("name,source", [("c1","dinov2_ref"),("dino_pooler","dinov2_frames"),("c3","vjepa")])
def test_presets(name: str, source: str) -> None:
    cfg = load_config(name)
    assert cfg.encoder.kv_source == source
    assert cfg.run_name.startswith(f"{name}_seed0-")

def test_config_unknown_key_rejected() -> None:
    with pytest.raises(chz.blueprint.ExtraneousBlueprintArg):
        load_config("c3", overrides=["adapter.nblocks=3"])

def test_json_roundtrip(tmp_path: typing.Any) -> None:
    cfg = load_config("c3", overrides=["seed=3", "optim.lr=0.02"])
    restored = read_config(save_config(cfg, tmp_path))
    assert chz.asdict(restored) == chz.asdict(cfg)
    other = replace_config(cfg, **{"adapter.strategy":"pooler"})
    assert other.adapter.strategy == "pooler" and cfg.adapter.strategy == "fusion"


def test_historical_nonaveraging_run_config_remains_readable(tmp_path: Path) -> None:
    cfg = load_config("dino_pooler")
    path = save_config(cfg, tmp_path)
    values = json.loads(path.read_text())
    values["encoder"]["dinov2"]["pair_average"] = False
    path.write_text(json.dumps(values))
    assert chz.asdict(read_config(path)) == chz.asdict(cfg)
    values["encoder"]["dinov2"]["pair_average"] = True
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="historical run"):
        read_config(path)

def test_environment(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("VIDEO3D_ENV","ucloud")
    monkeypatch.setenv("DATA_ROOT","/work/test")
    cfg = load_config()
    assert cfg.env == "ucloud" and cfg.paths.data_root == "/work/test"

def test_inspect_cli(capsys: typing.Any) -> None:
    main(["inspect", "--condition=dino_pooler", "--optim.lr=0.25"])
    assert '"lr": 0.25' in capsys.readouterr().out

def test_check_env_cli(capsys: typing.Any) -> None:
    main(["check-env", "--offline"])
    assert "paths.data_root" in capsys.readouterr().out

def test_torch_kernel_compatibility_rule() -> None:
    from video3d.check_env import torch_supports
    assert torch_supports((10,0), ["compute_90"])
    assert not torch_supports((10,0), ["sm_90"])
