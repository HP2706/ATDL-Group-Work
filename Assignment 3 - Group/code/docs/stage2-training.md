# Optional Stage-2 training

Stage 2 uses the existing trainer, checkpoint machinery and evaluator. The source-mesh
preparation path is implemented; bounded native GPU verification covers real mesh-derived
targets rather than generated teacher latents. Full-dataset target preparation and
scientific Stage-2 training are separate launches, not implied by a smoke check.

## Networks and LoRA

`train.stage=stage1` remains the default. `train.stage=stage2` trains the sparse
structured-latent flow network (`slat_generator`) and, when configured, a separate
video-conditioning adapter. The pinned Stage-1 flow/adapter/LoRA, image/video feature
encoders and mesh/Gaussian decoders remain frozen. Stage-2 adapters initialize afresh;
Stage-1 conditioning weights are not silently reused for different token sets.

**The default LoRA settings are the existing `sam3d.lora`, `sam3d.lora_rank` and
`sam3d.lora_alpha`.** Thus an existing rank-8/alpha-8 run keeps those settings for
Stage 2. Optional `stage2.lora`, `stage2.lora_rank` and `stage2.lora_alpha` default to
null and only change behavior when explicitly supplied. Stage-2 LoRA attaches to
`blocks.<n>.cross_attn.to_q`, `to_kv` and `to_out` in the sparse flow backbone.
The tested rank-8 setup has 1,376,256 trainable LoRA parameters. The trained mesh and
Gaussian decoders already exist in the pretrained model; fine-tuning the Stage-2 flow
can use them without retraining the decoders.

Fusion, attentive pooling and appended video/mask tokens use the same conditioning
interface for both stages. `stage2-model.json` records resolved LoRA settings, exact
attachment paths, frozen components and parameter counts beside the run configuration.

## Original meshes and target preparation

All 4,515 original GLBs are downloaded to:
`/work/MathiasElmegaardNielsen#6181/sam3dvid-original-assets-2026-10-30/glbs/`.
They total 39,118,830,396 bytes and are SHA256 verified. Source attribution and the
maintained downloader's index/completion receipts remain beside them.

`prepare-stage2` reuses the existing Blender normalizer and renderer. It checks original
asset hashes, recovers the original canonical surface voxels, and requires at least
0.98 IoU against the exported dataset before constructing targets. The two native
verification assets, including the held-out dinosaur, both matched exactly (IoU 1.0).
No generated SAM prediction is accepted as ground truth.

The feature recipe uses 32 deterministic Fibonacci-sphere views, transparent 518px
renders and frozen DINOv2 ViT-L/14-register prenorm patch features. Bilinear projection
onto canonical 64³ surface voxel centers and averaging across all views follows the
[official feature-extraction implementation](https://raw.githubusercontent.com/microsoft/TRELLIS/main/dataset_toolkits/extract_feature.py).
This is a documented custom rendering recipe, not a claim to reproduce unpublished
SAM training data. Render and normalizer source hashes, frozen feature weight hash,
source mesh hashes and normalization hashes are recorded.

Run these commands remotely, using the existing configuration/CLI overrides:

```bash
python -m video3d prepare-stage2 \
  --raw_root=/work/.../sam3dvid-original-assets-2026-10-30 \
  --output=/work/.../stage2-features \
  --asset_ids='["train_uid","heldout_uid"]'
python -m video3d cache-stage2 /work/.../stage2-features \
  stage2.target_root=/work/.../stage2-targets
```

Preparation saves canonical meshes, normalization metadata, view cameras/renders and
N×1024 source features. The frozen native SLAT encoder produces N×8 posterior means;
training applies native `slat_mean`/`slat_std` normalization. Source and target manifests
journal completed assets and pin file hashes. Existing incompatible caches are rejected
rather than overwritten. A changed recipe or source manifest requires a new target
cache identity. Held-out assets may be encoded, but the existing asset-disjoint sampler
keeps them outside optimization batches.

## Training and evaluation configuration

Use the existing run configuration with these partial overrides:

```json
{
  "train": {"stage": "stage2"},
  "stage2": {
    "target_root": "/work/.../stage2-targets",
    "stage1_run": "/work/.../corresponding-stage1-run",
    "stage1_checkpoint": "/work/.../corresponding-stage1-run/epoch_05.pt"
  },
  "sam3d": {"lora": true, "lora_rank": 8, "lora_alpha": 8.0},
  "val": {
    "epoch_media_samples": 0,
    "heavy_epochs": [1, -1],
    "heavy_data_root": "/work/.../audited-gso200",
    "final_full_validation": true
  },
  "eval": {"refined_mesh": true, "refinement_steps": 25},
  "logging": {"wandb": true}
}
```

This is not a complete production launch specification. Profile physical batch size
before a full run, retain the intended effective batch and pin the corresponding
Stage-1 checkpoint. An empty `stage1_run` uses pretrained Stage 1. A nonempty one requires
an immutable checkpoint; its EMA adapter/LoRA is restored and frozen. Starting Stage 2
creates a new optimizer/run. Resuming Stage 2 restores its optimizer, scheduler, RNG,
adapter/LoRA, EMA and progress. Target manifest and Stage-1 checkpoint/config hashes are
part of data identity, so incompatible resumes fail.

The sparse backend processes one object at a time; the microbatch loss averages objects
equally, independent of occupied voxel count. Existing accumulation preserves effective
batch size. Best checkpoint selection uses held-out Stage-2 flow loss. Stage-1-only
coarse preview metrics cannot measure Stage-2 improvement and are disabled/rejected.

Heavy GSO evaluation runs at `val.heavy_epochs` (with `-1` meaning the final epoch),
retaining the same manifest, references, frames, seeds and exact GSO mesh ground truth.
Final in-distribution evaluation retains the existing **full** synthetic held-out split
and now runs refined meshes for Stage 2. Its ground truth is the recovered canonical
original mesh, not a mesh made from 64³ occupancy. Frozen SAM3D, MV-SAM3D and Stream3D
can use the same ground-truth preparation and surface analyzer with `full_mesh=True`
and the same config's `stage2.target_root`. This does not change or restart old jobs.

Each evaluation saves the coarse latent/layout/occupancy, exact refined mesh, source
mesh, input identity, decoder SLAT features, alignment and sampled surface errors,
mesh validity diagnostics, timing and peak CUDA memory. Surface metrics and voxel
metrics retain their separate normalization protocols; they must not be pooled into
one leaderboard. Synthetic export cameras are preserved; high-resolution external
occlusion partitions are not yet derived from them. Existing voxel occlusion diagnostics
remain available. Stage-2 flow-loss summaries use their own SLAT protocol label.

## Maintained downloader

`scripts/generation/download_objaverse.py` retains seeded TRELLIS selection and also
accepts `--manifest /path/to/attribution.json` plus optional
`--object_paths /path/to/object-paths.json.gz`. Exact-manifest mode preserves order,
skips aesthetic filtering and treats `--limit` as a prefix. Existing GLBs are hash
verified, corrupt copies and prior completion records are preserved, partial transfers
resume, and failures exit nonzero. `index.json` lists only verified successful files;
`status.json` and `complete.json` record progress, selected identity and bytes.
Native CPU job 12440569 verified all 4,515 assets in 10.8309 seconds with zero downloads
and exit 0. The research log retains the download and earlier backend smoke receipts.

## Verification receipt

Native B200 job **12440583** passed with explicit exit 0 and provider SUCCESS;
its allocation released automatically. It verified two real-target optimizer updates,
checkpoint/resume through the shared trainer, inherited rank-8/alpha-8 LoRA with
72 projection attachments, and a refined held-out prediction. Fusion, pooling and
appended video/mask conditioning each received finite nonzero gradients while the
video encoder remained frozen. Peak allocated memory for the recorded native
training/evaluation portion was 18,772,855,808 bytes; this is a smoke measurement,
not a production batch-size profile.

The held-out dinosaur source encoder/decoder round trip produced 203,390 vertices.
Canonical closest-triangle Chamfer L1 sum was 0.00420447 with longest GT side 1;
F-score at 0.01 was 0.96429. These measure the source target round trip, not a
trained model's reconstruction quality. All 61 selected remote regression checks
passed, including Stage-1 behavior, checkpoint recovery, default LoRA inheritance,
shared original-mesh ground truth and the original downloader.

Evidence root: `/work/MathiasElmegaardNielsen#6181/stage2-real-targets-2026-10-30/`.
See `ready-integrated.json`, `preflight-integrated-exit.json`,
`run-specification-integrated.json`, `final-regression-complete.log` and
`final-regression-exit.json`. Runtime source archive SHA256:
`4972224a36cc66a878cfb7efacc7893e2591a5c2183971524a3c5e59887bc7fa`.
