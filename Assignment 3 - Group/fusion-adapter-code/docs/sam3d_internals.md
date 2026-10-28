# SAM 3D internals (what the wrapper relies on)

**Sources.** Static reading of `facebookresearch/sam-3d-objects` @ `f91db41` and `facebookresearch/vjepa2` @ `204698b` (the commits the group's earlier pilot pinned), cloned to `third_party/` (gitignored, never modified). File references are relative to `sam3d_objects/`.
**Status.** Everything below is read from code. Values that live in the gated checkpoint configs (`pipeline.yaml`, `ss_generator.yaml`) are marked **RUNTIME**: `python -m vidsam3d inspect-sam3d` prints them on a GPU machine and replaces the section at the end of this file.

## Summary of decisions

| # | Question | Answer from code | Consequence for us |
|---|---|---|---|
| 1 | Conditioning tokens | `EmbedderFuser` (`model/backbone/dit/embedder/embedder_fuser.py`) runs each `(embedder, [(kwarg, pos_group)])` of `embedder_list`, projects every set with a per-embedder `LayerNorm + FeedForward` to a shared width, adds a learned per-group embedding `idx_emb` (the modality embedding), and **concatenates along tokens** into one tensor. Names/order/lengths: **RUNTIME**. The earlier pilot measured 4 × 1370 tokens (DINOv2 at 518 px: CLS + 37² patches) + pointmap tokens. | The wrapper splits the fused tensor into named sets with forward hooks (no copied upstream code). `adapter.query_sets` = the two RGB sets; set from the runtime names. Assert `compression_projection_multiplier == 0`. |
| 2 | Time dependence | The pipeline calls the condition embedder **once** (`InferencePipeline.get_condition_input`), then passes the tokens to `ss_generator(latent_shape, device, cond)`. | Adapter runs once per sample; c' reused for every ODE step. |
| 3 | CFG null branch | `ClassifierFreeGuidance` with `unconditional_handling = "add_flag"` (set by the pipeline): the unconditional call gets the same tokens plus `cfg=True`; `SparseStructureFlowTdfyWrapper.forward` then uses `cond * 0` **if `force_zeros_cond`**. Formula `(1 + s)·cond − s·uncond`, applied only for `t·1000 ∈ cfg_interval` (default `[0, 500]`), `s = 7` default. In `train()` mode the module randomly drops the condition (`p_unconditional`). | Null branch = zeros, so it is untouched by the adapter. **RUNTIME assert** `force_zeros_cond is True` (else CFG is a no-op). Our `velocity()` calls the backbone directly in eval mode, so there is no CFG dropout during training. |
| 4 | Shortcut / sampler | The generator is a `ShortCut` model (`generator/shortcut/model.py`, step-size input `d`, `d_embedder`). Pipeline default: `no_shortcut=True` (d = 0) + CFG; `use_distillation=True` gives d = 1/steps and **no CFG**. Default `ss_inference_steps = 25`, `rescale_t = 3`; actual values **RUNTIME**. | Training uses d = 0. Sampler mode is a config choice (`sampler.mode`); see open question A. |
| 5 | τ distribution | `FlowMatching.training_time_sampler_fn` defaults to `lognorm_sampler(mean=0, std=1)` = logit-normal(0, 1); the yaml may override it (**RUNTIME**). Convention `x_t = (1 − t) x0 + t x1`, target `x1 − x0`, `t` scaled by `time_scale = 1000` before the backbone. | Matches our loss. `velocity()` multiplies τ by `time_scale`. |
| 6 | Latent normalisation | None. `sample_sparse_structure` decodes the generator output directly; `Latent.to_input` is a plain Linear + position embedding. The shape latent is `(B, 4096, 8)` tokens ↔ `(B, 8, 16, 16, 16)` via `permute(0, 2, 1).view(B, 8, 16, 16, 16)`. | Cache the encoder **mean** as `(4096, 8)`; no scaling. |
| 7 | Shape ↔ layout attention | `MOTMultiHeadSelfAttention.mm_scale_dot_product_attention`: `protect_modality_list = ["shape"]` (hard-wired default). Shape queries attend **only to shape keys**; layout queries attend to layout + detached shape keys. Each stream cross-attends to the same conditioning tokens. | **v_shape does not depend on the layout input.** `phase_a_layout: noise`; no zero-shot layout cache is needed in Phase A. GPU test checks it numerically. |
| 8 | VAE encoder input | `SparseStructureEncoderTdfyWrapper(x)` with `x: (B, 1, 64, 64, 64)` float, returns `{"z", "mean", "logvar"}`; mean `(B, 8, 16, 16, 16)`. The official demo `notebook/demo_ss_encoder.ipynb` voxelises a mesh and **fills the interior** (`fill(method="holes")`), i.e. solid occupancy. Decoder logits thresholded at `> 0` in the pipeline. | Our `voxels.npy` is **surface** occupancy. Config `target.voxel_fill: surface | solid`; T5 measures both and the better one is kept (open question B). |
| 9 | Stage 1 only | `InferencePipelinePointMap.run(stage1_only=True)` returns after `sample_sparse_structure` + `pose_decoder`. Components: `compute_pointmap` (MoGe depth model) → `preprocess_image(ss_preprocessor, pointmap)` → `ss_condition_embedder` → `ss_generator` → `ss_decoder`. | The wrapper instantiates the official pipeline from `pipeline.yaml`, keeps these components and deletes the stage-2 models. |
| 10 | Frames and layout | Canonical frame: the encoder demo maps glTF Y-up to Z-up with `[[1,0,0],[0,0,-1],[0,1,0]]` (= our `CANONICAL_FROM_GLTF`) and normalises the longest side to 1, as data format v0.1. Layout: quaternion local→camera in **PyTorch3D camera axes** (`camera_to_pytorch3d_camera`: x and y flipped vs. OpenCV), target convention from `module.pose_target_convention` (e.g. `ScaleShiftInvariant`, normalised translation), latent names in `latent_mapping` (**RUNTIME**). | Expect P = identity in T4 (to be confirmed). Phase B needs `ours_to_sam3d` to handle the axis flip and the pose convention. |

## Loading (no source changes)

- `os.environ["LIDRA_SKIP_INIT"] = "1"` before importing `sam3d_objects` (as `notebook/inference.py` does), `sys.path.insert(0, sam3d_repo)`.
- `OmegaConf.load(<checkpoint_dir>/pipeline.yaml)`, set `workspace_dir`, `rendering_engine = "pytorch3d"`, `compile_model = False`, `hydra.utils.instantiate`. The VAE encoder comes from `ss_encoder.yaml` / `ss_encoder.ckpt` in the same directory if the pipeline does not load it.
- DINOv2 is loaded by the `Dino` embedder through `torch.hub` (`facebookresearch/dinov2`, GitHub) at construction, so the first run needs network access (cached in `TORCH_HOME` afterwards).
- Attention backend: `inference_pipeline.py` sets `ATTN_BACKEND=flash_attn` at import only on A100/H100/H200; elsewhere (B200) the default `sdpa` is used, so flash_attn and xformers are not needed.
- Activation checkpointing: set `block.use_checkpoint = True` on the generator blocks (non-reentrant `torch.utils.checkpoint`, so gradients reach the conditioning tokens).
- Environment: torch 2.7.1+cu128 (`scripts/setup_env.sh`), not SAM 3D's pinned torch 2.5.1+cu121, which has no B200 (sm_100) kernels. Importing the pipeline also imports stage-2 modules, so their dependencies must import: pytorch3d and kaolin (rebuilt for the GPU), spconv (CUDA-12 wheel), gsplat (installed without its CUDA build, never run), open3d (needs `libGL`). The list is in `requirements.gpu.txt`. We do not patch SAM 3D (the earlier pilot patched `tdfy_dit/models/__init__.py` to skip these imports).

## V-JEPA 2.1

- Only ViT-L checkpoint: `vjepa2_1_vit_large_384` (`src/hub/backbones.py`), ViT-L/16 distilled from ViT-G, trained at 384 px, RoPE with `interpolate_rope=True`, so 256 px input works. Weights `vjepa2_1_vitl_dist_vitG_384.pt`; the pinned code points `VJEPA_BASE_URL` at `http://localhost:8300`, so we set it to `https://dl.fbaipublicfiles.com/vjepa2` (module attribute, not a source edit).
- Input `(B, 3, T, H, W)`; `T == img_temporal_dim_size (1)` uses the **image patch embedding** + image modality embedding, otherwise tubelets of 2 + video modality embedding. Output `(B, N, 1024)` after the last `norms_block`; `encoder.out_layers = [5, 11, 17, 23]` returns the 4 hierarchical levels.
- Patch embedding is `Conv3d(...).flatten(2).transpose(1, 2)`, so token order is (t, h, w); T3a checks it.
- Normalisation: ImageNet mean/std on [0, 1] input (`evals/video_classification_frozen/utils.py`).
- Dependencies `timm`, `einops` (in `requirements.gpu.txt`).

## Open questions

- **A. Sampler for all conditions.** (i) shortcut / distilled, 4 steps, no CFG: about 4 network evaluations per sample; (ii) flow, 25 steps, CFG 7: about 50 evaluations. Default in `base.yaml`: (i), for compute.
- **B. Surface vs. solid voxels.** T5 decides; if solid wins, all GT voxels are filled (`scipy.ndimage.binary_fill_holes`) before encoding and in the metrics.
- **C. Cost of the conditioning path.** MoGe (pointmap) + 4 DINOv2 passes per sample and step, online. If the benchmark shows it dominates, cache the preprocessed pointmap per (clip, frame); it does not depend on the object.

<!-- RUNTIME:BEGIN (written by `python -m vidsam3d inspect-sam3d`; do not edit by hand) -->
## Runtime values

Not run yet (needs a GPU machine with the SAM 3D environment and checkpoints).
<!-- RUNTIME:END -->
