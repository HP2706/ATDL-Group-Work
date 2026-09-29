---
pretty_name: Deep Double Descent published results
configs:
- config_name: vision
  data_files:
  - split: published
    path: data/vision/*.parquet
- config_name: translation
  data_files:
  - split: published
    path: data/translation/*.parquet
---

# Deep Double Descent published results

This is a tabular conversion of the public result files linked from the [authors' repository](https://gitlab.com/harvard-machine-learning/double-descent), pinned to commit `0d653296f870e6bf7ee82dc71b4f7b49d53fa85d`. It contains published experiment measurements, **not** CIFAR/translation training examples, trained weights, or results from our group. The public object names, generations, sizes, and MD5 hashes are in [`../manifest.json`](../manifest.json); conversion output hashes and row counts are in [`conversion-manifest.json`](conversion-manifest.json).

## Configs

- `vision`: 32 Parquet files, one per source experiment family. Each row describes one recorded time point for one model width, trial, and (where available) training-set size. Columns are `source_experiment`, `trial_index`, `model_width`, `sample_size`, `parameter_count`, `measurement_index`, `train_error`, `test_error`, `train_loss`, `test_loss`, and `robust_test_error`.
- `translation`: 296 published CSV rows combined in one Parquet file. Columns preserve `source_file` and `source_row`, with `task`, `sweep`, `model_width`, `sample_size`, and the four published metrics. Duplicate source rows remain separate.

`measurement_index` is the array position, starting at zero. The public pickle files do not provide a reliable optimizer step or elapsed epoch for every position. Missing `sample_size`, `parameter_count`, and `robust_test_error` values are null. `trial_index` is the position in the original `Mlist`, not a recovered random seed.

The original translation files and notebooks have unresolved metric-label and optimizer discrepancies; see the [experiment audit](../../report/main.tex). Do not compare these values as though every loss has a verified common unit or every row is an independent run. The split name `published` is a storage convention, **not** a model-training split.

## Load with Hugging Face Datasets

```python
from datasets import load_dataset

vision = load_dataset("parquet", data_files="data/vision/*.parquet", split="train", streaming=True)
translation = load_dataset("parquet", data_files="data/translation/*.parquet", split="train")
```

Run these commands from this directory, or use absolute paths. The local `parquet` loader calls its result `train` by default; this does not represent a model-training partition. When this folder is used as a Hugging Face dataset repository, its two named configs expose the files under the `published` split.

## Conversion

The original object snapshot was downloaded and converted on a CPU Modal Sprite with [`../../our-plots/download_results.py`](../../our-plots/download_results.py) and [`../convert_results.py`](../convert_results.py). The raw files remain available from the authors' public bucket. The conversion preserves float64 values and all recorded rows; it does not interpolate, aggregate trials, or infer missing metadata.

The authors' GitLab repository does not state a license for redistribution. This directory is prepared as a local Hugging Face dataset artifact; publishing it to the Hub requires a separate decision about destination and distribution rights.
