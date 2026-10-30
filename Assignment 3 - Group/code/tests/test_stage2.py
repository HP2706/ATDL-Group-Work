from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import chz
import pytest
import torch

from video3d.config import ExperimentConfig, TrainConfig, Stage2Config, Sam3DConfig
from video3d.data.stage2 import StructuredTarget, audit_targets, file_hash, load_target, validate_target
from video3d.stage2 import training_loss, validate_training


def target_fixture(root: Path) -> ExperimentConfig:
    cfg = ExperimentConfig(stage2=Stage2Config(target_root=str(root)))
    path = root / 'asset.pt'
    torch.save({'asset_id': 'asset', 'coords': torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int32),
                'features': torch.ones(2, 8), 'source_mesh_sha256': 'mesh'}, path)
    (root / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'target_kind': 'source_mesh_slat_encoder_mean',
        'sam_revision': cfg.sam3d.checkpoint_revision, 'coordinates': 'canonical-surface64-xyz',
        'features': 'unnormalized-slat-encoder-mean', 'assets': {
            'asset': {'file': path.name, 'sha256': file_hash(path), 'source_mesh_sha256': 'mesh'}}}))
    return cfg


def test_sparse_target_integrity_and_missing_asset(tmp_path: Path) -> None:
    cfg = target_fixture(tmp_path)
    audit_targets(cfg, ['asset'])
    assert load_target(cfg, 'asset').features.shape == (2, 8)
    with pytest.raises(FileNotFoundError, match='Missing 1'):
        audit_targets(cfg, ['other'])
    with (tmp_path / 'asset.pt').open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit_targets(cfg, ['asset'])


@pytest.mark.parametrize('coords,features', [
    (torch.tensor([[64, 0, 0]]), torch.ones(1, 8)),
    (torch.tensor([[1., 2., 3.]]), torch.ones(1, 8)),
    (torch.tensor([[1, 2, 3], [1, 2, 3]]), torch.ones(2, 8)),
    (torch.tensor([[1, 2, 3]]), torch.full((1, 8), float('nan'))),
])
def test_invalid_sparse_targets_fail(coords: torch.Tensor, features: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        validate_target(coords, features)


def test_teacher_predictions_are_not_ground_truth(tmp_path: Path) -> None:
    cfg = target_fixture(tmp_path)
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['target_kind'] = 'sam3d_prediction'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incompatible'):
        load_target(cfg, 'asset')


class SparseVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.1))
        self.pipeline = SimpleNamespace(slat_mean=torch.zeros(1, 8), slat_std=torch.ones(1, 8) * 2)

    def velocity(self, x: torch.Tensor, tau: torch.Tensor, cond: dict[str, torch.Tensor],
                 layout: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert x.shape[0] == 1 and layout['coords'].shape[1] == 4
        return self.gain * x + cond['image'].mean(), {}


def test_sparse_batch_loss_equals_mean_of_separate_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    targets = {'small': StructuredTarget(coords=torch.tensor([[1, 2, 3]]), features=torch.ones(1, 8)),
               'large': StructuredTarget(coords=torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]]),
                                         features=torch.ones(3, 8) * 3)}
    def load(cfg: ExperimentConfig, asset: str) -> StructuredTarget:
        return targets[asset]
    monkeypatch.setattr('video3d.stage2.load_target', load)
    model = SparseVelocity()
    cfg = ExperimentConfig()
    batch = [{'info': {'asset_id': name}} for name in targets]
    cond = {'image': torch.tensor([[[0.2]], [[0.7]]], requires_grad=True)}
    tau = torch.tensor([0.2, 0.8])
    generator = torch.Generator().manual_seed(7)
    loss = training_loss(model, cond, batch, cfg, torch.device('cpu'), generator, tau)
    generator = torch.Generator().manual_seed(7)
    individual = [training_loss(model, {'image': cond['image'][i:i+1]}, [batch[i]], cfg,
                                torch.device('cpu'), generator, tau[i:i+1]) for i in range(2)]
    torch.testing.assert_close(loss, torch.stack(individual).mean())
    loss.backward()
    assert model.gain.grad is not None and model.gain.grad.abs() > 0
    assert cond['image'].grad is not None and bool(cond['image'].grad.abs().sum() > 0)


def test_stage2_configuration_does_not_change_stage1_default() -> None:
    assert ExperimentConfig().train.stage == 'stage1'
    cfg = ExperimentConfig(train=TrainConfig(stage='stage2'), sam3d=Sam3DConfig(lora=True))
    with pytest.raises(ValueError, match='target_root'):
        validate_training(cfg)
    cfg = chz.replace(cfg, stage2=Stage2Config(target_root='targets'))
    validate_training(cfg)
    cfg = chz.replace(cfg, val=chz.replace(cfg.val, final_full_validation=True))
    validate_training(cfg)
    cfg = chz.replace(cfg, val=chz.replace(cfg.val, epoch_media_samples=1))
    with pytest.raises(ValueError, match='coarse'):
        validate_training(cfg)


def test_evaluation_restores_stage_and_target_identity(tmp_path: Path) -> None:
    from video3d.config import save_config, AdapterConfig
    from video3d.evaluate import load_run
    from video3d.models.adapter import build_adapter
    cfg = ExperimentConfig(train=TrainConfig(stage='stage2'),
                           stage2=Stage2Config(target_root='pinned-targets'),
                           adapter=AdapterConfig(strategy='native'))
    save_config(cfg, tmp_path)
    torch.save({'ema': build_adapter(cfg.adapter).state_dict()}, tmp_path / 'epoch_01.pt')
    resolved, _, _ = load_run(ExperimentConfig(), str(tmp_path), torch.device('cpu'),
                              str(tmp_path / 'epoch_01.pt'))
    assert resolved.train.stage == 'stage2'
    assert resolved.stage2.target_root == 'pinned-targets'


def test_native_pipeline_generator_is_registered_and_only_lora_trains() -> None:
    from video3d.stage2 import initialize_stage2
    model = torch.nn.Module()
    backbone = torch.nn.Module()
    block = torch.nn.Module()
    block.cross_attn = torch.nn.Module()
    for name in ['to_q', 'to_kv', 'to_out']:
        block.cross_attn.add_module(name, torch.nn.Linear(4, 4))
    backbone.blocks = torch.nn.ModuleList([block])
    guidance = torch.nn.Module()
    guidance.backbone = backbone
    generator = torch.nn.Module()
    generator.reverse_fn = guidance
    model.pipeline = SimpleNamespace(models={'slat_generator': generator},
                                     condition_embedders={'slat_condition_embedder': torch.nn.Linear(4, 4)})
    cfg = ExperimentConfig(train=TrainConfig(stage='stage2'), sam3d=Sam3DConfig(lora=True))
    initialize_stage2(model, cfg, torch.device('cpu'))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 6
    assert all(name.startswith('slat_generator.') and name.endswith(('.a', '.b')) for name in trainable)
    assert not generator.reverse_fn.backbone.blocks[0].cross_attn.to_q.base.weight.requires_grad
    assert not model.pipeline.condition_embedders['slat_condition_embedder'].weight.requires_grad


@pytest.mark.parametrize('shortcut', [False, True])
def test_refinement_restores_plain_and_shortcut_sampler(shortcut: bool) -> None:
    from video3d.stage2 import refinement_sampler
    generator = SimpleNamespace(inference_steps=12,
                                reverse_fn=SimpleNamespace(strength=0.0,
                                                           backbone=SimpleNamespace(is_shortcut_model=shortcut)))
    if shortcut:
        generator.no_shortcut = False
    with pytest.raises(RuntimeError, match='failed'):
        with refinement_sampler(generator, 25, 1.0):
            assert generator.inference_steps == 25 and generator.reverse_fn.strength == 1.0
            if shortcut:
                assert generator.no_shortcut
            raise RuntimeError('failed')
    assert generator.inference_steps == 12 and generator.reverse_fn.strength == 0.0
    if shortcut:
        assert not generator.no_shortcut


def test_independent_stage2_lora_configuration() -> None:
    from video3d.stage2 import lora_settings
    cfg = ExperimentConfig(sam3d=Sam3DConfig(lora=False, lora_rank=4),
                           stage2=Stage2Config(lora=True, lora_rank=16, lora_alpha=32))
    assert lora_settings(cfg) == (True, 16, 32)
    assert not cfg.sam3d.lora


def test_projection_uses_canonical_centers() -> None:
    from video3d.data.stage2_prepare import project_features
    pose = torch.eye(4)
    pose[2, 3] = 2
    cameras = {'T_object_camera': [pose.tolist()], 'K': [[50,0,49.5],[0,50,49.5],[0,0,1]],
               'resolution':100}
    features = torch.ones(1, 3, 10, 10)
    coords = torch.tensor([[31,31,31],[32,32,32]], dtype=torch.int32)
    torch.testing.assert_close(project_features(features, coords, cameras), torch.ones(2,3))


def test_synthetic_target_audit_resolves_asset_from_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    import video3d.stage2 as stage2
    seen: list[str] = []
    class Dataset:
        index = [{'clip_id':'c', 'k':1}]
        def clip(self, clip_id: str) -> SimpleNamespace:
            return SimpleNamespace(info={'objects':[{'asset_id':'original'}]})
    monkeypatch.setattr(stage2, 'audit_targets', lambda cfg, ids: seen.extend(ids))
    stage2.audit_training_targets(ExperimentConfig(), Dataset(), None)
    assert seen == ['original']


def test_final_stage2_validation_requests_refined_meshes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    from video3d.evaluate import evaluate_full_validation
    from video3d.config import read_config
    calls: list[object] = []
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: calls.append(args))
    cfg = ExperimentConfig(train=TrainConfig(stage='stage2'))
    evaluate_full_validation(cfg, tmp_path, 5)
    selected = read_config(tmp_path/'full-validation-epoch-05.json')
    assert selected.eval.refined_mesh and selected.data.eval_split == 'val'
    assert selected.eval.evaluation_id.endswith('-refined')


def test_stage2_defaults_inherit_existing_lora_settings() -> None:
    from video3d.stage2 import lora_settings
    for enabled in (True, False):
        cfg = ExperimentConfig(sam3d=Sam3DConfig(lora=enabled, lora_rank=12, lora_alpha=24))
        assert lora_settings(cfg) == (enabled, 12, 24)


def test_shared_mesh_ground_truth_preserves_original_geometry(tmp_path: Path) -> None:
    import trimesh
    from video3d.data.stage2 import file_hash
    from video3d.mesh_analysis import prepare_ground_truth
    cfg = target_fixture(tmp_path)
    mesh = tmp_path/'original.ply'
    trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False).export(mesh)
    path = tmp_path/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['assets']['asset'].update(mesh_path=str(mesh),mesh_sha256=file_hash(mesh))
    path.write_text(json.dumps(manifest))
    class Dataset:
        def clip(self, clip_id: str) -> SimpleNamespace:
            return SimpleNamespace(info={'objects':[{'asset_id':'asset'}]})
    out = tmp_path/'evaluation'
    out.mkdir()
    protocol = prepare_ground_truth(cfg,Dataset(),0,{'asset_id':'asset','clip_id':'clip'},out)
    recovered = trimesh.load(out/'ground-truth.glb',force='mesh',process=False)
    assert len(recovered.faces) == 1
    assert protocol['ground_truth_mesh_sha256'] == file_hash(mesh)
    assert protocol['visibility_cache'] is None
