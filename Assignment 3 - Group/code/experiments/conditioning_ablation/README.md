# Does replacing reference RGB conditioning improve video-to-3D learning?

Six trainable configurations share the existing `video3d.train` and
`video3d.evaluate` paths. A seventh configuration runs frozen native SAM 3D.
No DINO video encoder and no model-specific training loop are introduced.

`variants.json` is the complete matrix: fusion or V-JEPA token replacement pooling,
each with connector-only, SAM LoRA, or SAM + V-JEPA LoRA. The native baseline has
no LoRA. Both strategies inherit identical preprocessing, dimensions, clip length
and sampler settings from the base configuration, overriding only the three model
switches. Pooling replaces the selected native RGB token sets; SAM mask/pointmap
conditioning remains native.

Run `configure.py` with the existing prepared smoke config as the base, the FULL
audited Omni root, and the already frozen 250-object GSO root. The default requires
at least 6,000 audited Omni records and a nonempty object-disjoint train/val split.
The script rejects the current nine-object pilot or 100-object smoke dataset.
No models or datasets are downloaded by configuration generation.

```bash
python experiments/conditioning_ablation/configure.py \
  --base_config=/persistent/smoke/train-config/config.json \
  --output=/persistent/conditioning-<timestamp> \
  --omni_root=/persistent/omniobject3d --gso_root=/persistent/gso-250
```

The output holds immutable copied manifests and references to shared
assets, seven chz-resolved training configurations, seven GSO evaluation
configurations, and a matrix manifest. Training is ten epochs, seed 42, effective
batch 32, initial physical batch 8. The preparation GPU runs `profile.py` for each configuration at batches 8/16/32
before finalizing physical batch; accumulation preserves effective batch. Optimizer is
AdamW, LR 1e-4 with 5% warmup, EMA decay .99, BF16. Video inputs and image sizes
are inherited unchanged from the base config so the comparison isolates models.
LoRA rank/alpha and connector dimensions are likewise inherited.

The existing trainer evaluates fixed validation loss and canonical IoU panels
(up to 50 Omni validation objects) each epoch. It already saves `epoch_NN.pt`
before those evaluations, with a resumable recovery checkpoint every 50 updates
and at completion. Epoch reconstruction previews stay disabled because that
helper currently targets synthetic clips. No changes to train.py/evaluate.py
are needed to train or evaluate these six model combinations.

Resolved configurations are finalized before training after profiling; candidate
configs and measurements remain under `profiles/`. The effective batch stays 32.

Use one foreground UCloud batch invocation of `run.sh ROOT prepare-targets PYTHON`
to cache native GT latents once for train/val/GSO, followed by
`profile.py ROOT` in that same allocation before fan-out. Then launch seven separate
foreground batch invocations of `run.sh ROOT VARIANT PYTHON`; the six training
jobs require the completed targets marker and run training followed by final
GSO geometry evaluation. Provider allocation exit must release each GPU. Use a
provider wall limit exceeding the configured ten-hour training fallback by the
bounded evaluation allowance. Logs and exit JSON persist outside the checkout.

W&B is enabled in the resolved configs using the base config's existing project
and entity. Credentials are inherited at launch, never embedded in configs.
No Hugging Face publication is configured. Record the source commit and exact
resolved configs with the job payloads. For a full research run, finalize batching
and record measured trainable parameter counts in the same allocation first.

The existing evaluator prefers `best.pt` (validation canonical IoU), falling back
to `latest.json`; this final evaluator invocation follows that established
behavior. Epoch checkpoints are all retained for separately evaluating exact
epochs. This recipe does not silently substitute an epoch-10 checkpoint for the
selected checkpoint, add per-epoch full GSO exports, or claim mesh refinement:
its existing trained-model output is Stage-1 latent/voxel geometry.

Thin objects with no token above the configured foreground threshold retain their
most-covered positive-coverage token(s). Background and padding stay excluded;
a genuinely empty foreground still fails. Profiling can resume completed variants
with `--resume=True --attempt=recovery1`, preserving prior attempt logs.

## Full synthetic training export

The same resolver accepts `--synthetic_root=/persistent/snapshot/full-q98`
instead of `--omni_root`. It requires 10,000 clips by default and reuses
`ClipDataset` and its existing asset-disjoint split. By default, 3% of assets
are selected by the existing stable hash for validation; every clip containing
one of those assets is excluded from training. Only held-out assets are targets
in validation clips. Set `--synthetic_val_asset_frac` to change this fraction.
The original export and shards remain intact and are linked into the run.

`data/synthetic/split.json` records the exact asset lists, clip/target assignments,
and actual sample counts. Epoch steps use training clip-object pairs, matching
the trainer, rather than counting clips or unique assets. Validation sample
counts include the existing deterministic reference-frame choices. The synthetic
recipe fixes clip length at 16 frames to match this experiment's model comparison;
all other image/video preprocessing is inherited from the base configuration.
The frozen 250-object GSO evaluation source and seven-model matrix are retained.
`--evaluation_objects=200` freezes its first 200 manifest entries in the run,
so every model and resumed evaluation uses exactly the same bounded selection.
The source manifest remains unchanged.

```bash
python experiments/conditioning_ablation/configure.py \
  --base_config=/persistent/prior/configs/fusion-frozen.json \
  --output=/persistent/conditioning-full-q98-<timestamp> \
  --synthetic_root=/persistent/snapshot/full-q98 \
  --gso_root=/persistent/gso-250 --seed=42 --epochs=5 --evaluation_objects=200
```

Run fresh target preparation and GPU profiling for the new dataset before
submitting training. Transfer checksum validation and sample decoding are not a
full audit of the author's export; retain that distinction in run provenance.
