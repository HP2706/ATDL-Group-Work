# Assignment 2 report

## Official report description

Source: [Group Assignment 2 (Topics 1–2–3–4), Absalon](https://absalon.ku.dk/courses/91642/assignments/268755), checked directly on 3 October 2026.

Conduct a reproducibility study of a **subset of the main experiments** of one curated paper. Assess whether its experimental findings can be reproduced and whether its conclusions are supported by your results. Identify findings that could not be reproduced and give potential explanations. Each group member is expected to reproduce one experiment.

For the replication track, use released code or reimplement the paper's methods; support the report with targeted ablations and discuss their effects. There is no requirement to reproduce every figure or include a compute-cost catalogue.

- Use the ISBI template. Do not change font size, line spacing, or margins.
- Maximum **four pages of text**, excluding references, figures, and tables.
- Front page: group number, every member's name, selected track.
- State each member's concrete contribution; “equal contribution” is insufficient.
- Link the assignment GitHub repository.
- Explicitly describe changed model scale or experimental settings and their implications.
- Use precise academic language and mathematical notation where appropriate.
- Review the course information page's exam format, AI policy, and evaluation criteria.
- Submit a PDF through Absalon by **11 October 2026, 23:59 Copenhagen time**.

## Suggested narrative outline

Organize around the findings being tested rather than numbered figures or historical compute planning. Target four pages of text, with approximately the following allocation:

1. **Abstract and introduction (0.5 page):** identify the paper, explain double descent, and state the three questions: behavior versus model size, training time, and sample size. Show the authors' released 10% data beside our 10% adaptation; retain the original 15% opening figure separately in the appendix.
2. **Methodology and deviations (1 page):** describe datasets, architectures, width grids, fixed label-noise masks, augmentation, optimizers/schedules, evaluation, seeds, and horizons. Include a compact original-versus-ours protocol table. Explain how 229 training cells provide multiple figure views. State which code was reused or reimplemented, with its source.
3. **Results (1.5 pages):** group by model-wise, epoch-wise, and sample-wise behavior. For each, state the paper's claim, show or reference a comparison, quantify our observation, and explain whether it supports that claim under the reduced protocol. Use Figures 4–7 for noise/augmentation/optimizer checks, Figures 9–10 for trajectories, and Figures 11(a)–12 for sample size. Ten charts are not ten independent experiments.
4. **Discussion (0.6 page):** assess the effect of shorter training, sparse width grids, one seed, different noise rates, metric transformations, and uncertain authors' time coordinates. Distinguish inconclusive evidence from a contradiction. Do not claim that unrun translation experiments were reproduced.
5. **Conclusion and individual contributions (0.4 page):** answer the replication questions at the measured scope; give each member's concrete work and the repository link.
6. **References**, followed by the **figure appendix**. These are excluded from the four-page text allowance.

The current `main.tex` is a short factual working draft under this structure, not a submission-ready manuscript. Before submission, supply the actual group/member metadata and contribution statements, expand the implementation details and interpretation, and finish interpreting each comparison. The 229-run count, 9,160 recorded rows, and quantitative examples in this draft were rechecked against the local Parquet data on 3 October 2026.

## Figures and source distinctions

- Main Figure 1: noise-matched released-data reconstruction versus ours, both at 10% noise, ten widths, and 400 displayed epochs.
- Appendix decision table (`appendix-protocol.tex`): one row per core model, data, or training setting, with exact journal pages, appendix subsections, released filenames, and factual differences. The methodology references this table.
- Appendix A: the full settings comparison with source pages and released filenames.
- Appendix B: one gallery of ten paired replication charts, each comparing ours with the authors. No standalone or duplicate side-by-side galleries are included.

The journal's opening figure uses **15% noise and 4,000 epochs**; our adaptation uses **10% noise and 400 epochs**. The existing matched Figure 1 chart uses the authors' released **10%** data, not their original opening figure. Keep those distinctions explicit.

`figures/paper-figure-01.png` is a crop rendered directly from physical page 4 of `../../../papers/deep-double-descent/source.pdf`; it includes the complete two-panel figure without its caption. All other existing plot assets remain in `../plots/`, referenced directly rather than duplicated.

## Source provenance lives outside the report

[`../their-results/provenance/`](../their-results/provenance/README.md) contains public CSVs, extracted notebook source, grid/count checks, hashes, and provenance metadata from the September compute audit. It is **working provenance**, not report prose or a required appendix. It was moved out of `report/` on 3 October 2026. The manuscript does not import or need these files to compile. Preserve them to retain the verification history behind `EXPERIMENTS.md`; do not submit them with the PDF.

## File roles

- `main.tex`: report source, opened in the native LaTeX editor.
- `references.bib`: paper citation; use `\cite{nakkiran2021deepdouble}`.
- `appendix-figures.tex`: the ten paired replication charts, included by `main.tex`; standalone galleries are excluded.
- `experiment-cost-audit.tex`: the old cost/inventory sections removed from the active manuscript.
- `draft-before-reorganization.tex`: exact snapshot of the edited draft before this reorganization, including its unfinished abstract.
- `paper-selection-cost-report.tex`: earlier historical paper-selection comparison.
- `../their-results/provenance/`: historical source provenance, outside the manuscript directory.

The source is opened in the native editor. Its compiler currently cannot resolve this multi-file project (the local ISBI style, bibliography, figure assets, and appendix). The saved PDF is compiled successfully with the existing TeX installation: run `latexmk -pdf main.tex` from this directory. Open `main.pdf` to inspect that verified export.

## Interpretation of targeted ablations

The assignment asks for well-targeted ablations and discussion of their effect on model behavior. It does not explicitly require ablations absent from the original paper. A controlled reproduction of a paper's augmentation, noise, optimizer, or sample-size comparison can serve that purpose if the report explains the question, what is held fixed, and the observed effect. Merely including many plots is insufficient. This is an interpretation of the specification, not an additional rule stated by the teaching staff.

For our reduced reproduction, a useful additional sensitivity experiment would extend the training horizon at a few representative widths while keeping data, noise, optimizer, and seed fixed. This tests whether shorter training accounts for a missing peak or second descent. Widths should span the small-model region, a candidate critical region, and a larger-model region. Check the candidate critical region against training error rather than assuming it from a test-error peak. This is ablation X1 in [ABLATIONS.md](../ABLATIONS.md), alongside seed, weight-decay, and optimizer ablations (X2–X4).

Repeating the translation experiments on a different dataset would usually be a generalization or robustness extension rather than a conventional ablation of the original experiment. It changes the data distribution, vocabulary, sequence lengths, and possibly evaluation difficulty. Establishing an original-dataset reference first would strengthen its interpretation. There is no clear instruction to add another dataset; our current noise, augmentation, optimizer, and sample-size comparisons already offer targeted comparisons to discuss. No new experiments are launched by this guide.
