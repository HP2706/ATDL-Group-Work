# Assignment 2 report

## Measure text length

From this directory, run `../../.venv/bin/python measure_text_pages.py`
with `latexmk` and `pdfinfo` on the shell's PATH. The script rebuilds the
report, then creates two text-only PDFs in the gitignored `.text-length/`
directory without editing `main.tex`:

- `main-text.pdf`: title, abstract and main sections through the conclusion.
- `all-text.pdf`: the same material plus appendix prose, contributions and headings.

Both exclude figures, tables, captions and references, retain the ISBI font,
spacing, margins and two-column layout, and preserve original citation and
figure numbers. Appendix page breaks and the one-column gallery layout are
removed for the all-text measurement. Equations and headings count as text.
The page counts are written to `.text-length/summary.txt`. These are rendered
page counts, so a partially filled final page still counts as a page. This is
a reproducible layout check, rather than an official interpretation of which
additional material the course may exempt.

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

## Current narrative structure

The introduction and background are preserved. Methodology gives the essential
scope and deviations, with detailed settings in appendix tables. Results retain
quantitative comparisons and direct plot references; discussion prioritizes
interpretation and uncertainty. The conclusion distinguishes central findings
(C1, C3, C4) from supporting findings (C2, C5).

Individual contributions and the repository link begin the appendix. Moving
prose to the appendix does not automatically exempt it from the text limit,
so the all-text measurement includes it. The current rendered checks give
four pages for main text and five including appendix prose. The latter
exceeds the four-page allowance if appendix prose is counted.

## Figures and source distinctions

- Main Figure 1: noise-matched released-data reconstruction versus ours, both at 10% noise, ten widths, and 400 displayed epochs.
- Appendix decision table (in `main.tex`): one row per core model, data, or training setting, with exact journal pages, appendix subsections, released filenames, and factual differences. The methodology references this table.
- Appendix A: reproducibility information and individual contributions.
- Appendix B: vision sweeps, translation coverage, and a figure-by-figure table with internal references to the appendix plots.
- Appendix C: vision and translation settings comparisons, including source references and unresolved protocol details.
- Appendix D: mathematical definition of the text-length measurement.
- Appendix E: twelve comparison figures, including the IWSLT'14 experiments for paper Figures 3 and 11(b).

The journal's opening figure uses **15% noise and 4,000 epochs**; our adaptation uses **10% noise and 400 epochs**. The existing matched Figure 1 chart uses the authors' released **10%** data, not their original opening figure. Keep those distinctions explicit.

`figures/paper-figure-01.png` is a crop rendered directly from physical page 4 of `../../../papers/deep-double-descent/source.pdf`; it includes the complete two-panel figure without its caption. All other existing plot assets remain in `../plots/`, referenced directly rather than duplicated.

## Source provenance lives outside the report

[`../their-results/provenance/`](../their-results/provenance/README.md) contains public CSVs, extracted notebook source, grid/count checks, hashes, and provenance metadata from the September compute audit. It is **working provenance**, not report prose or a required appendix. It was moved out of `report/` on 3 October 2026. The manuscript does not import or need these files to compile. Preserve them to retain the verification history behind the report's appendix tables; do not submit them with the PDF.

## File roles

- `main.tex`: complete report source, including all appendix tables and figures, opened in the native LaTeX editor.
- `references.bib`: paper citation; use `\cite{nakkiran2021deepdouble}`.
- `experiment-cost-audit.tex`: the old cost/inventory sections removed from the active manuscript.
- `draft-before-reorganization.tex`: exact snapshot of the edited draft before this reorganization, including its unfinished abstract.
- `paper-selection-cost-report.tex`: earlier historical paper-selection comparison.
- `../their-results/provenance/`: historical source provenance, outside the manuscript directory.

The source is opened in the native editor. Its compiler currently cannot resolve this multi-file project (the local ISBI style, bibliography, and figure assets). The saved PDF is compiled successfully with the existing TeX installation: run `latexmk -pdf main.tex` from this directory. Open `main.pdf` to inspect that verified export.

## Interpretation of targeted ablations

The assignment asks for well-targeted ablations and discussion of their effect on model behavior. It does not explicitly require ablations absent from the original paper. A controlled reproduction of a paper's augmentation, noise, optimizer, or sample-size comparison can serve that purpose if the report explains the question, what is held fixed, and the observed effect. Merely including many plots is insufficient. This is an interpretation of the specification, not an additional rule stated by the teaching staff.

For our reduced reproduction, a useful additional sensitivity experiment would extend the training horizon at a few representative widths while keeping data, noise, optimizer, and seed fixed. This tests whether shorter training accounts for a missing peak or second descent. Widths should span the small-model region, a candidate critical region, and a larger-model region. Check the candidate critical region against training error rather than assuming it from a test-error peak.

Repeating the translation experiments on a different dataset would usually be a generalization or robustness extension rather than a conventional ablation of the original experiment. It changes the data distribution, vocabulary, sequence lengths, and possibly evaluation difficulty. Establishing an original-dataset reference first would strengthen its interpretation. There is no clear instruction to add another dataset; our current noise, augmentation, optimizer, and sample-size comparisons already offer targeted comparisons to discuss. No new experiments are launched by this guide.
