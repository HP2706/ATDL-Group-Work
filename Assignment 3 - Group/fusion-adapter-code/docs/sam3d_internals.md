# SAM 3D internals

What the wrapper (`vidsam3d/models/sam3d.py`) relies on. The facts come from a static reading of `facebookresearch/sam-3d-objects` @ `f91db41` and `facebookresearch/vjepa2` @ `204698b`, the commits `scripts/fetch_upstream.sh` pins. File paths are relative to `sam3d_objects/`. Values that live in the gated checkpoint configs (`pipeline.yaml`, `ss_generator.yaml`) are marked **RUNTIME**: `python -m vidsam3d inspect-sam3d` reads them on a GPU and fills in the section at the end of this file.

## Findings

Item numbers are referenced from `configs/base.yaml`.

1. **Conditioning tokens.** `EmbedderFuser` (`model/backbone/dit/embedder/embedder_fuser.py`) runs each embedder, projects every token set to a shared width (`LayerNorm` + `FeedForward`), adds a learned per-group modality embedding (`idx_emb`) and **concatenates the sets along the token axis**. Names, order and lengths are RUNTIME; an earlier pilot measured 4 × 1370 DINOv2 tokens (518 px: CLS + 37²) plus pointmap tokens.
   → The wrapper splits the fused tensor into named sets with forward hooks. `adapter.query_sets` = the two RGB sets, named from the runtime output. It asserts `compression_projection_multiplier == 0`.
2. **Time dependence.** The pipeline calls the condition embedder once (`InferencePipeline.get_condition_input`) and passes the tokens to `ss_generator(latent_shape, device, cond)`.
   → The adapter runs once per sample; c' is reused for every ODE step.
3. **CFG null branch.** `ClassifierFreeGuidance` with `unconditional_handling = "add_flag"`: the unconditional call gets the same tokens plus `cfg=True`, and `SparseStructureFlowTdfyWrapper.forward` replaces them with `cond * 0` if `force_zeros_cond`. Guidance is `(1 + s)·cond − s·uncond` (default s = 7) for `t·1000 ∈ cfg_interval` (default `[0, 500]`). In `train()` mode the module drops the condition at random.
   → The null branch is zeros and never sees the adapter; `force_zeros_cond` is asserted at runtime (otherwise CFG would be a no-op). `velocity()` calls the backbone directly in eval mode, so there is no condition dropout in training.
4. **Shortcut and sampler.** The generator is a `ShortCut` model (`generator/shortcut/model.py`) with a step-size input d. Pipeline default: `no_shortcut=True` (d = 0) with CFG, `ss_inference_steps = 25`, `rescale_t = 3`; `use_distillation=True` gives d = 1/steps without CFG. Actual values are RUNTIME.
   → Training uses d = 0. The sampler is `sampler.mode` (question A).
5. **τ distribution.** `FlowMatching.training_time_sampler_fn` defaults to logit-normal(0, 1); the yaml may override it (RUNTIME). Convention: `x_t = (1 − t) x0 + t x1`, target `x1 − x0`, t multiplied by `time_scale = 1000` before the backbone.
   → Matches our loss; `velocity()` applies `time_scale`.
6. **Latent normalisation.** None: `sample_sparse_structure` decodes the generator output directly. The shape latent is `(B, 4096, 8)` tokens ↔ `(B, 8, 16, 16, 16)` via `permute(0, 2, 1).view(...)`.
   → `cache-latents` stores the encoder mean as `(4096, 8)` without scaling.
7. **Shape ↔ layout attention.** In `MOTMultiHeadSelfAttention`, `protect_modality_list = ["shape"]`: shape queries attend **only to shape keys**; layout queries attend to layout and detached shape keys. Both streams cross-attend to the same conditioning tokens.
   → The shape velocity does not depend on the layout input, so Phase A feeds layout noise (`sam3d.phase_a_layout: noise`) and needs no zero-shot layout cache. A GPU test checks this numerically.
8. **VAE encoder input.** `SparseStructureEncoderTdfyWrapper(x)` takes `x: (B, 1, 64, 64, 64)` float and returns `{"z", "mean", "logvar"}` with mean `(B, 8, 16, 16, 16)`. The official demo (`notebook/demo_ss_encoder.ipynb`) fills the interior (solid occupancy); the pipeline thresholds decoder logits at `> 0`.
   → Our `voxels.npy` is surface occupancy, so `target.voxel_fill: surface | solid` is set from T5 (question B).
9. **Stage 1 only.** `InferencePipelinePointMap.run(stage1_only=True)` stops after `sample_sparse_structure` and `pose_decoder`. Path: `compute_pointmap` (MoGe) → `preprocess_image` → `ss_condition_embedder` → `ss_generator` → `ss_decoder`.
   → The wrapper instantiates the pipeline from `pipeline.yaml`, keeps these components and deletes the stage-2 models.
10. **Frames and layout.** The encoder demo maps glTF Y-up to Z-up with `[[1,0,0],[0,0,-1],[0,1,0]]` (our `CANONICAL_FROM_GLTF`) and scales the longest side to 1, as data format v0.1 does. The layout is a local→camera quaternion in **PyTorch3D camera axes** (x and y flipped vs. OpenCV); the pose convention (`module.pose_target_convention`) and latent names (`latent_mapping`) are RUNTIME.
    → Expect P = identity in T4. Phase B needs `ours_to_sam3d` to handle the axis flip and the pose convention.

## Loading (no source changes)

- Set `LIDRA_SKIP_INIT=1` before importing `sam3d_objects` (as `notebook/inference.py` does) and put the repo on `sys.path`.
- Load `<checkpoint_dir>/pipeline.yaml`, set `workspace_dir`, `rendering_engine = "pytorch3d"` and `compile_model = False`, then `hydra.utils.instantiate`. The VAE encoder comes from `ss_encoder.yaml` / `ss_encoder.ckpt` if the pipeline does not load it.
- DINOv2 comes through `torch.hub` (`facebookresearch/dinov2`) at construction, so the first run needs network access; it is cached in `TORCH_HOME` afterwards.
- Attention backend: `flash_attn` only on A100/H100/H200; elsewhere (B200) `sdpa`, so flash_attn and xformers are not needed.
- Activation checkpointing: `block.use_checkpoint = True` on the generator blocks (non-reentrant, so gradients reach the conditioning tokens).
- Environment: torch 2.7.1+cu128 from `scripts/setup_env.sh`, because SAM 3D's pinned torch 2.5.1+cu121 has no B200 (sm_100) kernels. Importing the pipeline also imports stage-2 modules, so pytorch3d and kaolin (rebuilt), spconv, gsplat (built without CUDA, never run) and open3d (needs `libGL`) must import. The list is in `requirements.gpu.txt`.

## V-JEPA 2.1

- The only ViT-L checkpoint is `vjepa2_1_vit_large_384` (`src/hub/backbones.py`): ViT-L/16 distilled from ViT-G at 384 px, with interpolated RoPE, so 256 px input works. The pinned code points `VJEPA_BASE_URL` at `localhost`; we set it to `https://dl.fbaipublicfiles.com/vjepa2` (a module attribute, not a source edit).
- Input `(B, 3, T, H, W)`. T = 1 uses the image patch embedding and image modality embedding; otherwise tubelets of 2 frames. Output `(B, N, 1024)`; `out_layers = [5, 11, 17, 23]` gives the 4 hierarchical levels.
- The patch embedding is `Conv3d(...).flatten(2).transpose(1, 2)`, so the token order is (t, h, w); T3a checks it.
- ImageNet mean/std on [0, 1] input.

## Open questions

- **A. Sampler for all conditions.** Shortcut, 4 steps, no CFG (≈ 4 network evaluations per sample; the default) or flow, 25 steps, CFG 7 (≈ 50).
- **B. Surface vs. solid voxels.** T5 decides. If solid wins, GT voxels are filled (`scipy.ndimage.binary_fill_holes`) before encoding and in the metrics.
- **C. Cost of the conditioning path.** MoGe plus the DINOv2 passes run online for every sample and step. If the benchmark shows they dominate, cache the pointmap per (clip, frame); it does not depend on the object.

<!-- RUNTIME:BEGIN (written by `python -m vidsam3d inspect-sam3d`; do not edit by hand) -->
## Runtime values

Not run yet (needs a GPU machine with the SAM 3D environment and checkpoints).
<!-- RUNTIME:END -->
