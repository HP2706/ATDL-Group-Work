# Presentation manuscript: nnActive, part 4

Slides 14–17. Target time: approximately 3:45 at a steady speaking pace.
Bracketed directions are not spoken.

## Slide 14: Measuring annotation effort (~65 seconds)

> We have compared accuracy at fixed patch budgets. But two patches of the same
> size can take very different amounts of time to annotate. An empty background
> patch is usually easier than one containing complicated organ boundaries.
>
> The authors therefore introduce Foreground Efficiency. They plot performance
> against the amount of foreground annotated and fit how quickly the gap to
> a fully labeled reference model closes. A larger gamma means faster closure
> per annotated foreground fraction.
>
> [Point to the KiTS example.]
>
> On medium-budget KiTS, PowerPE has higher foreground efficiency than Predictive
> Entropy: about nine-point-seven versus six-point-two. Yet Entropy has higher
> final Dice: about sixty-five versus fifty-nine.
>
> So efficient use of foreground labels and the best final accuracy can favor
> different methods. Foreground count is still only a proxy for human effort.
> It does not measure minutes, and gamma should only be compared within matched
> experimental settings.

## Slide 15: What changes the acquisition outcome? (~65 seconds)

> The authors also test whether their findings depend on implementation choices.
>
> First, smaller acquisition batches generally help because the model gets
> feedback more often and can adjust what it selects next. The cost is more
> rounds of training and inference. These are annotation batches, not training
> minibatches.
>
> Second, longer training helps both the final predictions and the quality of
> patch selection. They separate these effects by also training longer on
> patches selected by shorter-trained models.
>
> Third, adding selection noise often helps at small budgets by broadening
> selection. Less noise can help later, but there is no universally best setting.
>
> Finally, changing query-patch dimensions can change the method rankings.
> Halving each dimension gives eight times fewer voxels per patch. That does
> not establish equal accuracy with eight times less annotation work.
>
> Together, these experiments show that acquisition quality depends on the
> surrounding training and annotation procedure.

## Slide 16: How far should we trust the conclusion? (~55 seconds)

> The study has several strengths: a strong segmentation pipeline, four diverse
> tasks, multiple budgets and seeds, and more demanding baselines. It also
> considers annotation effort instead of treating every patch as equally costly.
>
> But there are important limits. Foreground localization is simulated using
> ground truth, so its real screening cost remains unknown. Counting foreground
> voxels also misses differences in boundary complexity, annotator skill, and
> annotation tools.
>
> The study tests five uncertainty-based methods. It cannot establish that all
> forms of active learning behave the same way.
>
> Finally, the starting labels already cover every foreground class. These
> experiments examine how to expand a structured initial dataset. They do not
> test how to choose the first labels from a completely unlabeled collection.

## Slide 17: Does active learning save annotation effort? (~40 seconds)

> The paper's answer is conditional. Active learning usually beats uniform
> random patch sampling. But none of the tested active methods consistently
> dominates foreground-aware random sampling across tasks and budgets.
>
> Predictive Entropy is the strongest active method overall in the main setup,
> although its final accuracy does not capture early performance or the amount
> of foreground that had to be annotated.
>
> The practical lesson is to demonstrate gains over a strong baseline using a
> defensible measure of annotation cost. A higher Dice score at the same patch
> count does not, by itself, establish that a method saves human work.

## Quick reference for questions

- **What does gamma measure?** A fitted exponential rate at which the gap to
  a matched fully labeled reference closes as annotated foreground increases.
  It is neither Dice nor an accuracy gain per minute.
- **Why compare within matched setups?** The dataset, label regime, starting
  annotations, and training setup affect the curve and its reference point.
  Gamma values across unrelated tasks do not form a useful leaderboard.
- **Exact KiTS medium values:** PowerPE: FG-Eff 9.69, Final Dice 58.67.
  Predictive Entropy: FG-Eff 6.20, Final Dice 65.39.
- **Longer-training control:** compare 200-epoch acquisition, 500-epoch
  acquisition, and 500-epoch training on query trajectories precomputed by
  200-epoch models. The last condition helps separate better fitting from
  better selection.
- **What would strengthen the annotation-cost claim?** A timed annotator study
  that includes screening, delineation, correction, and interaction overhead,
  alongside final accuracy and compute requirements.
- **Does noise guarantee diversity?** No. Randomized selection can broaden
  acquisition without guaranteeing coverage of different semantic classes.
- **What is left open?** Other acquisition families, other clinical domains,
  cold-start selection, and realistic end-to-end human annotation costs.

Sources: local course paper, sections 5.2–6, Appendix D.4, Figure 7, Table 6,
and Appendix G. See [the detailed paper summary](../../papers/nnactive/summary.md).
