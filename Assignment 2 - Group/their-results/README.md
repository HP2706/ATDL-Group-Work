# Authors' published results

`manifest.json` pins the 74 public objects in the authors' `hml-public/dd/` bucket by generation, size, and MD5. `hf_dataset/` is their tabular Parquet conversion for Hugging Face Datasets. The raw `dd/` files are preserved locally if downloaded, but ignored by Git; the converted Parquet files are the versioned result artifact.

The 33 converted Parquet files are the only Assignment 2 files configured for
Git LFS. A normal clone downloads their contents. To get the code and plots
first and fetch only the result sources needed for a specific plot, clone with
`GIT_LFS_SKIP_SMUDGE=1` and then run `git lfs pull --include=<path>` from the
group-work repository root. For example:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/HP2706/ATDL-Group-Work.git
cd ATDL-Group-Work
git lfs pull --include="Assignment 2 - Group/their-results/hf_dataset/data/vision/cifar10-resnet18k-50k-adam.parquet"
```

All 229 of our completed-run Parquet files total about 3.6 MB and are ordinary
Git files, so they arrive with the clone. Pilot result exports and the raw
authors' objects stay out of Git. A plot that reads a skipped LFS file needs
that file fetched first.

The [conversion scripts](convert_results.py) run on a CPU Modal Sprite or another Linux host. They keep the authors' values in float64, retain duplicate translation rows, and preserve source identifiers. See the [dataset card](hf_dataset/README.md) for schema and limitations.

The [provenance archive](provenance/README.md) preserves the public CSV and notebook snapshots and grid/count verification records used for the historical compute audit. These are source records, not submission content. They were moved here from `report/` to keep the manuscript directory focused on report files.
