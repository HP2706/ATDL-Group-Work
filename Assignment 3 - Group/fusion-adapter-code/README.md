# Video SAM 3D

Trains a small gated cross-attention **fusion adapter** (~18M parameters) that lets the frozen SAM 3D Objects Geometry model use a video clip through frozen V-JEPA 2.1 features. Evaluation is on the coarse 64³ shape only.

Docs: [data format v0.1](docs/data_format.md) · [research plan](docs/research_plan.md) · [adapter plan](docs/fusion_adapter_plan.md) · [SAM 3D internals](docs/sam3d_internals.md) (written by `inspect-sam3d`, P5)

> Stub (P1). The full README comes in P8.

## Quickstart (local, CPU)

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch -r requirements.txt -e .
.venv/bin/python -m vidsam3d check-env --env local
.venv/bin/python -m pytest
```

## Config

`configs/base.yaml` ← `configs/conditions/<name>.yaml` ← `configs/env/<env>.yaml` ← `key=value` overrides.
Select with `--condition c3 --env ucloud` (env also from `$VIDSAM3D_ENV`, default `local`). Every run saves its resolved `config.yaml`.
