# Translation model setup

The paper's Appendix B.1 specifies a six-layer encoder–decoder Transformer, eight attention heads per layer, embedding width `d`, feed-forward width `4d`, and no dropout. Appendix B.3 specifies BPE preprocessing and the Vaswani warmup followed by inverse-square-root learning-rate schedule. Section 4 and Figure 8 say 80,000 updates; Appendix B.1 says “80 gradient steps,” which conflicts with those statements. Appendix A lists Adam for translation, while the authors' repository README says SGD. The paper also specifies 10% label smoothing.

The authors' [GitLab repository](https://gitlab.com/harvard-machine-learning/double-descent) does not include the translation code. The relevant files from [Fairseq v0.9.0](vendor/fairseq-v0.9.0/README.md), released in 2019, are copied unchanged under `vendor/` for reference. [`src/text_experiment/model.py`](src/text_experiment/model.py) is a standalone adaptation of their training path. It uses **only PyTorch** and runs in the parent project's Python 3.12 environment. The executable adaptation retains Fairseq's separate attention projections, initialization, sinusoidal positions, post-normalization, and encoder–decoder structure. The [upstream MIT license](LICENSE.fairseq) is included.

From the `ATDL-Group-Work` directory, a small model can be constructed with:

```bash
PYTHONPATH='Assignment 2 - Group/code' .venv/bin/python -c 'from src.text_experiment.model import make_transformer; print(make_transformer(16, 19, 1, 1, 8))'
```

The model was checked under Python 3.12: six encoder and decoder layers, eight self-attention heads, eight decoder cross-attention heads, `d_ff = 4d`, and finite output logits. For a stricter comparison, matching weights from Fairseq 0.12.2 were loaded into this adaptation and its teacher-forced output differed by at most `7.2e-7` on the same padded input. This verifies the implemented forward path against Fairseq, not the unpublished original experiment configuration.

The factory takes source/target vocabulary sizes and padding indices explicitly. The paper does not identify its Fairseq commit, vocabulary sizes, embedding-sharing choice, exact BPE files, batching, warmup duration, or seeds. The model currently uses separate embeddings and post-normalization, matching Fairseq's base defaults. Incremental generation, checkpoint migration, and unrelated Fairseq options were deliberately omitted; the assignment's perplexity experiments use teacher forcing. Record these assumptions and any training deviations in the report.

The A100 timing pilot and the reduced 400-epoch CIFAR-10 ResNet sweep have completed. Training and timing belong on remote compute, with checkpoints during each run.

## Training entry points

Run these from `Assignment 2 - Group/code/` in the project `.venv`. The source tree now has two runners:

| Script | Supported tasks | Key controls |
| --- | --- | --- |
| `vision_train.py` | ResNet18 or five-layer CNN on CIFAR-10 or CIFAR-100 | Width, fixed label noise, deterministic sample subset, augmentation, Adam/SGD, learning-rate schedule, momentum, weight decay, seed |
| `translation_train.py` | IWSLT'14 German–English only | Embedding width, sample subset, BPE text files, token budget, Adam warmup schedule, label smoothing, seed |

The image runner uses the authors' image model definitions, with one implementation for all four architecture/dataset combinations. It defaults to Adam at `1e-4` and 4,000 epochs for ResNet, SGD at `0.1` with inverse-square-root decay and 500,000 steps for CIFAR-10 CNN, and one million steps for CIFAR-100 CNN. These are paper protocols, not suitable values for a local smoke run. Both runners save periodic metrics and atomic checkpoints. `SIGTERM` or `SIGINT` checkpoints after the current update; pass `--resume=<run-directory>` with the same config to continue. Different seeds create independent runs and independent image corruption masks. `test_predictions.pt` contains final image predictions for ensemble analysis.

The two runners inherit shared data, output, sample-count, evaluation, checkpoint, seed, and device fields from `src/training/common.py:TrainingConfig`. Result rows are validated as `VisionResult` or `TranslationResult` Pydantic models, serialized with `model_dump()`, and written to Parquet with the authors' common column types.

### Vision sweep launcher

The scripts in `code/scripts/` define experiment arguments and each call [launch.py](launch.py) once. `plan=true` lists conditions locally. The launch backend comes from the repo-local `.atdl-launch.local.toml`, copied from [the example](../../.atdl-launch.example.toml); `ATDL_LAUNCH_CONFIG` can select another file and `ATDL_BACKEND` can override `backend`. The local config is ignored by Git. Use `backend = "ucloud"` to submit from the Mac or `backend = "direct"` to execute on a Linux GPU host. Relative `results_dir` and `extend_from` arguments are placed on the persistent work drive (`/work/<drive>` on UCloud, `ATDL_WORK_ROOT` for `direct`). The UCloud project ID may be shared by collaborators with access to that project, while template job IDs and the SSH key can differ by user.

For UCloud, the launcher starts a ten-minute CPU staging job from the configured CPU template, waits for its SSH endpoint, copies an isolated code snapshot to the persistent drive, and submits a GPU batch job from the configured GPU template. Set `ATDL_STAGE_JOB` to an already-running staging job ID to reuse it after a failed attempt; the launcher checks that it mounts the expected work folder. The batch wrapper calls `launch.py run` directly with the experiment arguments; it never calls the experiment shell script again. `submit_plan=true` stages and previews the UCloud request without launching a GPU job, but it still starts the short CPU staging job. A one-time W&B credential file is removed when the batch starts. With `WANDB_MODE=disabled` in the submitting shell, no key is needed and W&B logging is switched off for the whole sweep.

`code/scripts/cifar10_resnet.sh` defines the reduced ResNet18/CIFAR-10 sweep: widths `{2,3,4,6,8,12,16,24,32,64}` × noise `{0,10,20}%`, one seed, augmentation, Adam at constant LR `1e-4`, batch size 128, and 400 epochs. `code/scripts/cifar100_resnet.sh` uses the same grid on CIFAR-100. The chz-based Python sweep derives steps per epoch from the selected dataset and sample count, resumes incomplete conditions, runs four conditions per visible GPU, and displays one global progress bar in optimizer steps. Measurements and resumable checkpoints are saved every ten epochs. W&B receives metrics; comparison-compatible Parquet and checkpoints stay under persistent `/work` storage.

`code/scripts/figure_6.sh` currently defines the Adam arm of the reduced clean, non-augmented CIFAR-10 CNN comparison: 11 widths at 400 epochs. The separate `cifar10_cnn_sgd_no_data_aug.sh` supplies the matching clean SGD arm at 50,000 steps.

`code/scripts/figure_11a.sh` adds augmented CIFAR-10 CNN runs at 12,500 and 25,000 training samples, 10% and 20% label noise, and widths `{1,2,3,4,6,8,12,16,24,32,64}`. These 44 conditions use SGD for 50,000 steps and one seed, matching the reduced horizon of our completed 50,000-sample sweep. This is a sparse, single-seed subset of the paper's Figure 11(a).

The `x1_*.sh` to `x4_*.sh` scripts define the targeted ablations; see [ABLATIONS.md](../ABLATIONS.md).

```bash
# From this project directory on the Mac:
bash code/scripts/cifar10_resnet.sh plan=true
bash code/scripts/cifar10_resnet.sh submit_plan=true
bash code/scripts/cifar10_resnet.sh  # uses the configured backend

# A narrower one-off grid uses the same submission path:
bash code/scripts/cifar100_resnet.sh widths=2,3 noise_rates=0.0
```

Edit the argument array in a sweep script to change its default grid. Each UCloud submission gets a timestamped `run_root` under the mounted `/work` drive and a separate source snapshot; set `RUN_ROOT` to the printed earlier path to resume that run. Dataset and results paths share the persistent drive. The script's `8` or `10` launcher argument is the reservation in hours; `UCLOUD_HOURS=...` overrides it for one submission. The batch wrapper sends `SIGINT` 20 minutes before that limit and allows ten minutes for checkpointing. Completed batch jobs exit when the sweep exits; the reservation is a ceiling. UCloud CLI authentication, an SSH key, the project drive and Python environment, and `WANDB_API_KEY` from the local shell configuration are required.

The launcher downloads and verifies the CIFAR dataset **on the GPU host** under `/work`. At normal completion or a handled interruption, the trainer writes a Parquet file under `RESULTS_DIR/data/vision/`. Config, metadata, metric history, checkpoints, and predictions remain under the persistent `/work` run directory; W&B receives metric logs only. All 30 reduced ResNet sweep Parquet files were downloaded locally; the completed job's SSH endpoint is gone.

`code/scripts/cifar10_cnn_sgd.sh` defines the exploratory augmented five-layer CNN sweep. It uses SGD without momentum, initial LR `0.1` with inverse-square-root decay every 512 updates, batch size 128, and **50,000 steps** per condition. It evaluates and checkpoints every 1,250 updates. The shared `sweep.py` supports either epoch-based or step-based horizons and shows one global progress bar in optimizer steps. UCloud job `12403687` completed an earlier 16-width, three-noise-rate grid of 48 runs; results and checkpoints remain on the persistent drive. The current script defines its own grid, visible with `plan=true`.

```bash
# Inspect the grid without training:
bash code/scripts/cifar10_cnn_sgd.sh plan=true

# Preview a new UCloud submission without launching another job:
bash code/scripts/cifar10_cnn_sgd.sh submit_plan=true
```

The CNN script creates a timestamped `RUN_ROOT`. To resume after interruption, set `RUN_ROOT` to that exact previous directory. The run files and Parquet results stay under `/work`; W&B receives only metrics. The 50,000-step runs are a shorter comparison, not completed 500,000-step paper reproductions. To continue completed runs to a longer horizon, pass `extend_from=<completed run root>`: the sweep copies each matching run into the new run root, changes only its output paths and horizon, and resumes it. The original runs are left unchanged.

### CIFAR setup

The vision loader uses `torchvision.datasets.CIFAR10` or `CIFAR100`. `--download=True` fetches the selected dataset to `--data_dir`; perform data preparation and training on the remote machine. From `ATDL-Group-Work/Assignment 2 - Group/code/`:

```bash
../../.venv/bin/python vision_train.py --architecture=resnet --dataset=cifar10 --width=8 --label_noise=0.15 --download=True --data_dir=data/cifar --output_dir=runs/vision
../../.venv/bin/python vision_train.py --architecture=cnn --dataset=cifar100 --width=8 --augmentation=False --download=True --data_dir=data/cifar --output_dir=runs/vision
```

Changing `--architecture` and `--dataset` covers the four combinations. For a **small local functional check only**, override `--max_steps=1 --batch_size=2 --eval_every=1 --checkpoint_every=1 --device=cpu` with a tiny fixture. Full training and timing belong on remote compute. The image test split is always clean; training label corruption is sampled once per original image and always changes the class. A subset is selected before training by a recorded seed; a sample count equal to the split size keeps the full split in its original order. Augmentation is a padded random crop and horizontal flip. The paper does not specify channel normalization, so the runner scales pixels to `[0,1]` without mean/std normalization.

For Figures 16–18 use `--optimizer=adam|sgd`, `--schedule=constant|inverse_sqrt|dynamic_drop`, `--learning_rate=...`, and `--momentum=0.9` as appropriate. Figure 21/22 weight decay is `--weight_decay=...`. Sample-wise image experiments use `--sample_count=...`. The dynamic-drop defaults are implementation choices because the paper does not fully specify its patience/drop settings; record selected values in the report. The runner does not implement Figure 26 adversarial training or the Fashion-MNIST random-feature appendix, which were excluded from the agreed scope.

### IWSLT'14 setup

The translation loader expects six **aligned, already tokenized/BPE-processed** UTF-8 files in one data directory: `train.de`, `train.en`, `valid.de`, `valid.en`, `test.de`, and `test.en`. Each line is one sentence; corresponding German and English files must have equal line counts. The [official Fairseq IWSLT preparation script](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt14.sh) produces these files under `iwslt14.tokenized.de-en/`. Run that preparation on the remote machine, pin the Fairseq script revision, and copy the six files to `data/iwslt14/`. The paper does not identify the exact original BPE revision or split hashes, so save those in the report. The runner records SHA-256 hashes of all six files and refuses a resume if they changed.

```bash
../../.venv/bin/python translation_train.py --data_dir=data/iwslt14 --embedding_dim=64 --sample_count=4000 --output_dir=runs/translation
```

Omit `--sample_count` for the full IWSLT experiment. Sweep dimensions in multiples of eight for Figure 3, use `--sample_count=4000` and `18000` for its two curves, or fix `--embedding_dim=64`/`80` and sweep sample counts for Figure 11(b). The runner builds source and target vocabularies from the full training split, then takes a deterministic subset. It teacher-forces the target, trains with 10% label smoothing, and reports **unsmoothed** token NLL and perplexity on train, validation, and test data. It defaults to Adam `(0.9, 0.98)`, epsilon `1e-9`, a 4,000-step warmup, inverse-square-root decay, no dropout, and 80,000 updates. Token budget and optimizer settings are documented assumptions: the authors did not release their exact translation command, and their README conflicts with the paper on Adam versus SGD.

All outputs under `data/` and `runs/` are ignored by Git. Do not treat a successful local one-step smoke check as evidence of reproducing a paper figure.

## Comparable result files

Each run keeps `metrics.jsonl`, `config.json`, `run_metadata.json`, and its checkpoint in `runs/`. On a normal finish or a handled interruption, the trainer exports a Parquet comparison artifact under `../our-results-folder/data/`. If a process is killed before export, rebuild it with `../../.venv/bin/python export_results.py <run-directory>`.

Image run IDs are `{dataset}-{architecture}-k{width}-seed{seed}-{timestamp}`. When the optimizer is not the architecture's default (Adam for ResNet, SGD for CNN), it follows the architecture, as in `cifar10-resnet-sgd-k12-…`. This keeps the two optimizer variants of a sweep apart when they start in the same second; older runs keep their names. The export refuses to replace a Parquet file written by a run with other settings. Pass `--name=<unique run ID>` to `export_results.py` to write it under another file name and run ID instead.

- Vision: `data/vision/<run-id>.parquet` has the **same first 11 columns and Arrow types** as the authors' vision Parquet conversion: source, trial, width, sample size, parameter count, measurement index, train/test error, train/test loss, and robust error. Our error values are fractions in `[0,1]`; cross-entropy losses are mean nats. Extra columns give the actual seed, noise, optimizer, update, and epoch. `trial_index` is null because the original `Mlist` position cannot be inferred from our run. Our `measurement_index` is the zero-based position in our own evaluation history, never an inferred original epoch.
- Translation: `data/translation/<run-id>.parquet` is **one endpoint row** with the same first 10 columns and Arrow types as the authors' CSV conversion. `data/translation_history/<run-id>.parquet` preserves each measured point with explicit update and epoch. Our `train_loss`/`test_loss` are unsmoothed, teacher-forced per-token NLL in nats. Our error fields are non-padding token error percentages. The authors' error columns and loss units remain unresolved, so do not overlay their curves with ours until a shared definition is established. Perplexity is stored separately as `exp(NLL)`.

The [assignment-local AGENTS.md](../AGENTS.md) records the full result-format convention for future work. The authors' files under `their-results/` are never changed by the trainers.
