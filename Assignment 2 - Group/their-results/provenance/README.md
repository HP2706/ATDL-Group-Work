# Verification record

This folder is working provenance for the historical compute audit. It is not required report content and is not included by `../../report/main.tex`. See the [report guide](../../report/README.md) for the manuscript structure.

Checked 2026-09-25. This is a source/count audit, not an executed ML replication or timing benchmark.

## Primary sources

- Local paper: `../../../../papers/deep-double-descent/source.pdf` (2021 journal version, 32 pages, Figures 1–29). DOI: https://doi.org/10.1088/1742-5468/ac3a74
- Upstream commit: `0d653296f870e6bf7ee82dc71b4f7b49d53fa85d` at https://gitlab.com/harvard-machine-learning/double-descent
- Notebook source snapshots contain only code/Markdown cells, not embedded outputs. Notebooks were read, not executed.
- Public object metadata: https://storage.googleapis.com/storage/v1/b/hml-public/o?prefix=dd/&delimiter=/

- `nlp-model-dd.csv`: https://storage.googleapis.com/hml-public/dd/nlp-model-dd/nlp-model-dd.csv
- `dd-translation-model-french.csv`: https://storage.googleapis.com/hml-public/dd/nlp-model-dd/dd-translation-model-french.csv
- `df4k.csv`: https://storage.googleapis.com/hml-public/dd/nlp-small-datasets/df4k.csv
- `df18k.csv`: https://storage.googleapis.com/hml-public/dd/nlp-small-datasets/df18k.csv
- `dd-translation-samples-aug26-small.csv`: https://storage.googleapis.com/hml-public/dd/nlp-sample-dd/dd-translation-samples-aug26-small.csv
- `dd-translation-samples-aug27-size80.csv`: https://storage.googleapis.com/hml-public/dd/nlp-sample-dd/dd-translation-samples-aug27-size80.csv

## Numeric grid provenance

`verified-grids.json` contains numeric values extracted with `pickletools.genops`, without unpickling/executing the public payloads. Each key except the sample-count key is the directory name in `https://storage.googleapis.com/hml-public/dd/<key>/ks`. Sample counts come from `dd/dd_grid_p20/ns`. Width-list presence does not prove all runs or seed combinations completed.

## Checks performed

- Counted CSV rows and distinct coordinates: 296 rows, 294 distinct within-file coordinates.
- Confirmed duplicate dimensions 56 and 112 in `df18k.csv`. Four possible cross-file overlaps at sample counts 4000/18000 and dimensions 64/80 require seed/subset provenance before reuse.
- Verified 64 ResNet widths and 42 × 9 image sample-grid coordinates.
- Read all figure captions, Appendix A and protocol text; visually inspected PDF pages 22, 23 and 26 for learning-rate/decay legends.
- No public result CSV has a runtime/GPU-count field. No timing measurements or training were performed.
- Large Mlist arrays and checkpoints were not downloaded. Model source definitions were not executed.
- Source conflicts and metric-column problems are documented in ../../report/experiment-cost-audit.tex.
- Pricing is an inherited $3.49/hour scenario, not a newly verified market quote.

## Local file hashes

- `SimpleNLP-sampledd-notebook-source.txt`: `42a120080392ffcc8f7e8113d8414aa204d7a7660ea6b57388484e55b5145caf`
- `dd-translation-model-french.csv`: `539435d69d6f00d672a1f3c9fa3afa7d8b3581d77badda43d61e1e4dd8039c3c`
- `dd-translation-samples-aug26-small.csv`: `afb0978c4e8bd0f57ed1a4805f227270cc4e6fdedad996b3252c2e689414c180`
- `dd-translation-samples-aug27-size80.csv`: `2eac9e99eb09d04cbd04db6810934412ddae44fc2e2c543caeb79f8b215f74ca`
- `df18k.csv`: `11f2129a1a1076b780c00a60b198199eac434348c7d0534e2dbebb611c38f6ff`
- `df4k.csv`: `c2d14a00cb1c614fecd5cf839d83f487235ee4170b457b3b03a8f896e0bcf39a`
- `intro_ocean_dynamics-notebook-source.txt`: `c0c89b3fd70ad2414142e209f2f8d50419916a6c662912e2f5f930b21c359552`
- `intro_ocean_plot-notebook-source.txt`: `498619b15c31f2ee272789794d7ea38691fc0ec23a616c92137689f3057d059f`
- `intro_resnet_plot-notebook-source.txt`: `c874941f8f9ca026f372008140a69f9535441867b0088cc3fc04ded6ba209286`
- `nlp-model-dd.csv`: `c07b98b900259ee25999838d6dafecf199798994990bb92d745306da9ca3b236`
- `nlp-model-double-descent-notebook-source.txt`: `cd7969ffa4297f9b2094ac858f899dc26e52eb4df297708a29b63608495a0d55`
- `source-manifest.json`: `c112877f88152362d9e923dae0298e4e603de59bc56b7d7cd1f2fe8e63abdb69`
- `upstream-README.md`: `5c9b9e7f8920e775d44ee195d971f5d1839f6933cf3b5f647c683f508223bdd5`
- `verified-csv-counts.json`: `faa376be05955f1fcf8c491c71a91b9a9f61b2f728eb16c489badbc5de3f81a6`
- `verified-grids.json`: `7b6550dc75bfbe0fd9b2142926bfd8389ae6ca0bc8749216cfae76b8739412f3`
