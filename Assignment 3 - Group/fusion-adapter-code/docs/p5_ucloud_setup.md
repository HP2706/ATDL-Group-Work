# P5 setup on UCloud Terminal

This guide prepares the P5 GPU diagnostics in an interactive UCloud Terminal Ubuntu
job. It downloads source repositories and model files from their upstream services
directly to UCloud storage; you do not need to upload the large third-party repos
or SAM 3D weights from your laptop.

The commands below assume the project's UCloud config defaults to `/work/vidsam3d`.
If your attached persistent storage is mounted elsewhere, set `PROJECT_DIR` to that
path instead. UCloud's `/work` storage persists across jobs; files elsewhere in the
container may not.

## 1. Start an interactive GPU Terminal

1. Open UCloud's [Terminal Ubuntu app](https://docs.cloud.sdu.dk/Apps/terminal.html).
2. Select a GPU machine with at least **32 GB VRAM** and enough job time and project
   storage. The SAM 3D generator checkpoint alone is about 6.7 GB; leave room for
   the environment, source checkouts, and downloaded V-JEPA weights.
3. Start an interactive terminal session. Do not use **Batch processing** for this
   walkthrough: UCloud terminates that job after its script exits.
4. In the terminal, set up persistent paths:

   ```bash
   export PROJECT_DIR=/work/vidsam3d
   mkdir -p "$PROJECT_DIR"
   export HF_HOME="$PROJECT_DIR/hf_home"
   export TORCH_HOME="$PROJECT_DIR/torch_home"
   export CONDA_PKGS_DIRS="$PROJECT_DIR/conda/pkgs"
   mkdir -p "$HF_HOME" "$TORCH_HOME" "$CONDA_PKGS_DIRS"
   ```

   If the app exposes persistent storage at a different path, change the first line.

## 2. Get the project and pinned upstream code

If the P5 project commit is pushed to the repository's default branch, a sparse
checkout avoids downloading unrelated reports and outputs:

```bash
git clone --filter=blob:none --sparse \
  https://github.com/HP2706/ATDL-Group-Work.git "$PROJECT_DIR/source"
cd "$PROJECT_DIR/source"
git sparse-checkout set "Assignment 3 - Group/fusion-adapter-code"
export CODE_DIR="$PROJECT_DIR/source/Assignment 3 - Group/fusion-adapter-code"
```

If your P5 changes are on another pushed branch, check out that branch before
running the sparse-checkout command. If they have not been pushed, transfer only
`fusion-adapter-code` to UCloud (excluding `.venv`, `third_party`, and checkpoints)
or push the branch first.

The checked-in P5 internals document records these upstream pins:

```bash
export SAM3D_REPO="$PROJECT_DIR/sam-3d-objects"
export VJEPA2_REPO="$PROJECT_DIR/vjepa2"

git clone https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_REPO"
git -C "$SAM3D_REPO" checkout f91db411c50efee93d8db7aeb323885650f6f722

git clone https://github.com/facebookresearch/vjepa2.git "$VJEPA2_REPO"
git -C "$VJEPA2_REPO" checkout 204698b45b3712590f06245fbfba32d3be539812
```

Verify that those are still the intended project pins before using a later P5
revision. The V-JEPA checkout contains code; its pretrained weights are downloaded
when the encoder is first loaded and are cached under `TORCH_HOME`.

## 3. Create the SAM 3D Python environment

SAM 3D's [upstream setup guide](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
requires Linux and a CUDA GPU. Use the environment instructions for the pinned
SAM 3D checkout:

```bash
cd "$SAM3D_REPO"
mamba env create --prefix "$PROJECT_DIR/envs/sam3d-objects" \
  -f environments/default.yml
eval "$(mamba shell hook --shell bash)"
mamba activate "$PROJECT_DIR/envs/sam3d-objects"

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra
```

Then install the adapter package and its additional dependencies:

```bash
cd "$CODE_DIR"
pip install -r requirements.txt -e .
```

Building PyTorch3D and Kaolin can depend on the active CUDA/GPU environment. If
these installation steps fail, follow the error and the upstream setup guide for
the selected UCloud machine rather than installing a different PyTorch/CUDA build
at random.

## 4. Authenticate to Hugging Face and download only the needed checkpoints

First, request and accept access to
[`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects) on
Hugging Face. A read token with access to the gated repository is required.

**SSH is not required to authenticate.** The UCloud terminal runs in the browser;
the HF token is used by HTTPS requests from that machine. UCloud's optional SSH
server is just another way to open a shell, and would require adding your SSH
public key to UCloud.

For a session-only token, enter it at a hidden prompt instead of putting it in a
command, file, or shell history:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
printf '\n'
export HF_TOKEN
```

This leaves the token in the current shell environment for the commands below;
the project's `check-env` command reports whether a token exists but does not print
it. Avoid `export HF_TOKEN=hf_...`, committing a token, or saving it in a shared
project file. At the end, run `unset HF_TOKEN`.

Download the files directly from Hugging Face. The repository stores them inside
its `checkpoints/` folder; setting `SAM3D_CHECKPOINT_DIR` to that folder makes the
paths match what this project expects. The YAML files are needed alongside their
corresponding weights:

```bash
export SAM3D_CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$SAM3D_CHECKPOINT_DIR"

hf download --repo-type model \
  --local-dir "$PROJECT_DIR" \
  --max-workers 1 \
  facebook/sam-3d-objects \
  checkpoints/pipeline.yaml \
  checkpoints/ss_generator.yaml checkpoints/ss_generator.ckpt \
  checkpoints/ss_decoder.yaml checkpoints/ss_decoder.ckpt \
  checkpoints/ss_encoder.yaml checkpoints/ss_encoder.ckpt
```

The selected model files use roughly 7 GB. Do not clone the Hugging Face model
repository with Git LFS or download the whole repository unless you need its other
models; this project uses the listed `ss_*` geometry files.

## 5. Check configuration, data, and run P5

Confirm the required files and repositories are present:

```bash
test -f "$SAM3D_CHECKPOINT_DIR/pipeline.yaml"
test -f "$SAM3D_CHECKPOINT_DIR/ss_generator.ckpt"
test -f "$SAM3D_CHECKPOINT_DIR/ss_decoder.ckpt"
test -f "$SAM3D_CHECKPOINT_DIR/ss_encoder.ckpt"
test -f "$VJEPA2_REPO/src/hub/backbones.py"
```

Run the diagnostics from the adapter source directory, inside the activated SAM 3D
environment:

```bash
cd "$CODE_DIR"
python -m vidsam3d check-env --env ucloud --load-models
python -m vidsam3d inspect-sam3d --env ucloud
VIDSAM3D_ENV=ucloud pytest -m gpu tests/gpu -s
python -m vidsam3d frame-test --env ucloud
unset HF_TOKEN
```

Read `check-env` output, not just its exit status: resolve any failed model loads,
missing files, or denied HF access before proceeding. The GPU tests and
`frame-test` also require the evaluation dataset, including
`$DATA_ROOT/manifest.jsonl`; by default the UCloud config looks under
`/work/vidsam3d/dataset`. A test summary showing skipped tests is not a successful
end-to-end GPU validation. Set `DATA_ROOT` if the dataset is mounted elsewhere.

`inspect-sam3d` updates `docs/sam3d_internals.md`. `frame-test` writes
`frame_test.json` under the configured output directory and reports the canonical
voxel permutation to use.

## Can setup be automated further?

Yes: code and checkpoints can be cloned/downloaded from inside the UCloud terminal,
as above, instead of manually uploading them. At the P5 revision documented here,
the P7 setup script mentioned in the project notes is not present in
`fusion-adapter-code`; environment creation and these downloads are therefore
manual steps for now. A future P7 script can automate this sequence, but should
still obtain the HF token interactively or through an approved secret mechanism,
not store it in the repository.
