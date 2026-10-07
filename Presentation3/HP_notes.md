# HP presentation manuscript: nnActive, part 3

Slides 9–12. Target time: approximately 3:45 at a steady speaking pace.
Bracketed directions are not spoken.

## Slide 9: A benchmark across tasks and budgets (~55 seconds)

> Now that we have seen the acquisition methods, the question is how to compare
> them fairly. The authors test four datasets: cardiac structures, fifteen
> abdominal organs, kidney structures, and the hippocampus. These cover different
> anatomies and different numbers of classes.
>
> Each dataset has a low, medium, and high annotation budget. The numbers in the
> table count query patches, rather than whole scans. Patch dimensions differ
> between datasets, so equal patch counts do not mean equal amounts of annotation
> across datasets.
>
> They compare eight methods using four acquisition seeds and the same held-out
> test split. Within each setting, the methods share the initial labels. They
> start at twenty percent of the budget and acquire more patches in four rounds.
>
> This lets us ask whether a method works across tasks and budgets, rather than
> only in one favorable experiment.

## Slide 10: Foreground-aware random sampling (~55 seconds)

> The random baseline is especially important here. A medical scan can contain
> a lot of background, so uniformly sampling patches may spend much of the
> budget on regions without the organs or tumors we want to segment.
>
> Random sixty-six percent foreground is a stronger baseline. About one-third
> of its queries are completely random. The remaining two-thirds target
> foreground. Half of those are centered on a randomly selected foreground
> class, and half on a foreground-class boundary.
>
> [Point to “Random 66% FG.”]
>
> Sixty-six percent describes how often selection targets foreground. It does
> not mean that sixty-six percent of every patch contains foreground.
>
> The benchmark uses ground truth to simulate finding these structures. The
> idea is to approximate rough human localization before detailed annotation,
> although the screening time is not measured. The question becomes: can
> uncertainty selection beat this anatomy-aware random baseline?

## Slide 11: The baseline changes the verdict (~45 seconds)

> [Point first to the uniform Random comparison, then to Random 66% FG.]
>
> This figure shows how much the answer depends on the baseline. Against uniform
> Random, the active methods have many more significant wins than losses.
>
> Against foreground-aware Random, the picture changes. Only Predictive Entropy
> has a positive overall win-loss balance against Random sixty-six percent
> foreground, and even that does not mean it wins consistently across settings.
>
> These bars count significant pairwise comparisons. They do not show the size
> of the Dice improvements.
>
> The main point is that beating uniform Random alone gives incomplete evidence
> for the value of active learning. Part of the apparent benefit can come from
> selecting useful anatomy instead of easy background.

## Slide 12: The dataset can reverse the winner (~70 seconds)

> These two examples make the dataset dependence concrete. Dice measures overlap
> between the prediction and the reference segmentation; higher is better.
> The scores here are multiplied by one hundred.
>
> [Point to the AMOS row.]
>
> On low-budget AMOS, foreground-aware Random reaches about seventy-one, while
> uniform Random and Predictive Entropy reach only about thirty-six and
> thirty-nine. AMOS has fifteen organs. The authors suggest that explicitly
> sampling classes helps cover small structures, while uncertainty selection
> can repeatedly concentrate on a subset of classes.
>
> [Point to the KiTS row.]
>
> On medium-budget KiTS, the ordering reverses. Predictive Entropy reaches about
> sixty-five, compared with fifty-two for foreground-aware Random. Here, the
> authors suggest uncertainty helps find difficult background regions where
> the model makes false-positive predictions.
>
> We should compare methods within each row, because these are different tasks
> and budgets. These explanations are the authors' interpretations of the
> results. The broader conclusion is that there is no consistent winner. The
> next question is whether these accuracy gains also mean less annotation work.

## Quick reference for questions

- **Eight methods:** Predictive Entropy, BALD, PowerPE, PowerBALD,
  SoftrankBALD, uniform Random, Random 33% FG, and Random 66% FG.
- **Four seeds versus five models:** four seeded acquisition experiments;
  each uses a five-model ensemble trained through five-fold cross-validation.
  These are different sources of repetition. Model training itself is not seeded.
- **Budget points:** 20%, 40%, 60%, 80%, and 100% of the chosen patch budget,
  not percentages of the full dataset. Low-budget AMOS ends at 200 patches;
  medium-budget KiTS ends at 1,000 patches.
- **Dice:** $2|P\cap G|/(|P|+|G|)$ for predicted foreground $P$ and reference
  foreground $G$. The study reports mean Dice per 3D image.
- **Final Dice versus AUBC:** Final Dice measures the endpoint; area under the
  budget curve also rewards good performance early in acquisition. Random 66%
  FG has the best aggregate mean AUBC rank in the main study, while Predictive
  Entropy is the strongest active method overall.
- **Exact slide values:** AMOS low: Random 36.36, Random 66% FG 71.11,
  Entropy 39.19. KiTS medium: Random 48.41, Random 66% FG 51.67,
  Entropy 65.39. These are reported means, not guarantees for a new run.
- **Scope:** the results do not show that all active learning is ineffective.
  They concern the tested methods, datasets, budgets, and training setup.

Sources: local course paper, sections 4–5.2, Figure 3, Appendix E, and Table 6.
See [the detailed paper summary](../../papers/nnactive/summary.md).
