# Assignment 2: Reproducibility study

Selected paper: **Deep Double Descent**, replication track. The [report guide](report/README.md) contains the official requirements, suggested narrative outline, and figure/source conventions. The [report](report/main.tex) now separates the reproducibility narrative from the complete figure appendix. The historical [experiment inventory and compute audit](report/experiment-cost-audit.tex) is preserved as working material rather than included in the submission.

For a compact figure-by-figure table, code availability, and setup status, see [the experiment map](EXPERIMENTS.md). The targeted ablations X1–X4 and how to run them are in [ABLATIONS.md](ABLATIONS.md).

The authors' public result snapshot is converted to [a Parquet dataset for Hugging Face Datasets](their-results/hf_dataset/README.md), with the original files pinned in [the source manifest](their-results/manifest.json). Curated images are organized in [plots](plots/README.md): paired comparisons, our measurements alone, and reconstructions from the authors' measurements alone. The [published plot coverage table](their_plots/README.md) identifies figures the release cannot reconstruct. Our measurements belong in [our-results-folder](our-results-folder/README.md). Use the [plotting CLI](our-plots/README.md) to render the authors' curve and heatmap layouts without notebooks.

The earlier [paper-selection cost comparison](report/paper-selection-cost-report.tex) is preserved separately, with its [original PDF](report/paper-selection-cost-report.pdf). Its runtime allowances are historical, unbenchmarked estimates. Public data snapshots and the verification record are in [their-results/provenance/](their-results/provenance/).

Deadline: October 11, 2026 at 23:59 Europe/Copenhagen.

Choose one curated paper from Topics 1-4 and reproduce a meaningful subset of its experiments. The goal is to assess whether the paper's findings can be reproduced and whether its conclusions are supported by your results.

Tracks:

- Baselines track: reproduce and evaluate the baselines reported in the paper, including relevant ablations or hyperparameter tuning.
- Replication track: reproduce the paper's main results using the released codebase when available, or re-implement the method from the paper and support the report with targeted ablations.

Each group member is expected to reproduce one experiment from the selected paper.

Requirements:

- Use the ISBI LaTeX template in `../ISBI Template/`.
- Maximum 4 pages, excluding references, figures, and tables.
- Submit a PDF through Absalon.
- State the group number, all members, and selected track on the front page.
- Describe each member's concrete contribution. Do not write only "equal contribution".
- Include a link to the assignment's GitHub repository.
- Explicitly disclose smaller or modified models and discuss the implications.

Additional notes from the published Canvas specification:

- Submission is a PDF upload through Absalon.
- Public pretrained weights and smaller model variants are acceptable when needed for feasibility.
- Large pretrained models may be used for inference. If necessary, smaller models may be trained from scratch or distilled from a larger model.
- If you deviate from the original model scale or setup, explain how that may affect reproducibility and conclusions.

Source: [Canvas assignment specification](https://absalon.ku.dk/courses/91642/assignments/268755), checked October 3, 2026. No official starter repository or additional setup archive is linked.
