# Assignment 2: Reproducibility study

[Compare all currently published curated papers, feasible experiments, and GPU costs](paper-reproduction-costs.md).

The blank ISBI report scaffold is [report/main.tex](report/main.tex). From the `report/` directory, build it with `latexmk -pdf main.tex`.

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

Source: [Canvas assignment specification](https://absalon.ku.dk/courses/91642/assignments/268755), checked September 13, 2026. No official starter repository or additional setup archive is linked.
