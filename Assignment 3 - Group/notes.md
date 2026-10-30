Jeg ved ikke hvor lang tid det ville tage at køre data pipelinen igen og hvor svært det ville være at køre igen, men jeg tror det kunne være fedt at have nogle videor hvor objekterne er mere occluded, f.eks i sam3d paper gør de netop det bevidst så modellen er incentiveret til at forudse objektet ud fra en intern model den bygger af objektet og ikke bare information på billedet. Det samme for vores model skal vi incentivere den til at samle information fra flere billeder hvilket den jo ikke rigtigt behøver hvis det meste af objektet kan ses på alle billederne. igen vi kan også bare nævne det som en weakness, men hvis vi kan nå at have det med kunne det være meget fedt.

### GSO evaluation: the same 200 objects, coarse geometry

All six trained epoch-1 checkpoints completed evaluation on exactly the same 200 GSO object IDs. Older baseline results below are recomputed from their saved per-object scores on that exact subset of the original 250 objects. Loss was not computed in these geometry evaluations.

Protocol A: older baselines, 25-step flow sampling, 20,000 metric points and 100 ICP iterations. Protocol B: epoch-1 evaluations, four-step shortcut sampling, 2,048 metric points and 40 ICP iterations. Bold marks the best value within each protocol group; the two groups are not a controlled ranking. Baseline SAM3D uses one input frame, MV-SAM3D/Stream3D four, video models 16, and SAM-only LoRA one.

| Model | Checkpoint / protocol | Loss ↓ | Coarse IoU ↑ | Aligned IoU ↑ | Chamfer ↓ | F-score@0.02 ↑ |
|---|---|---:|---:|---:|---:|---:|
| SAM3D | Pretrained / A | — | 0.2041 | 0.2471 | 0.1545 | 0.6107 |
| MV-SAM3D | Pretrained / A | — | 0.2475 | 0.2994 | 0.0990 | 0.7193 |
| Stream3D | Pretrained / A | — | **0.2513** | **0.3065** | **0.0923** | **0.7393** |
| Fusion adapter | Epoch 1 / B | — | 0.1847 | 0.2292 | 0.1944 | 0.3398 |
| Fusion + SAM LoRA | Epoch 1 / B | — | 0.1893 | 0.2426 | 0.1847 | 0.3467 |
| V-JEPA pooling | Epoch 1 / B | — | 0.0577 | 0.0889 | 0.3740 | 0.1717 |
| V-JEPA pooling + SAM LoRA | Epoch 1 / B | — | 0.1013 | 0.1393 | 0.2866 | 0.2253 |
| DINOv2 pooling | Epoch 1 / B | — | 0.0841 | 0.1105 | 0.3217 | 0.2014 |
| SAM-only LoRA | Epoch 1 / B | — | **0.1944** | **0.2479** | **0.1809** | **0.3543** |

SAM-only LoRA has the best displayed epoch-1 metrics among the six trained models. A separate pretrained SAM3D control under protocol B scores IoU 0.2111, aligned IoU 0.2581, Chamfer 0.1815 and F-score@0.02 0.3594. Thus no trained epoch-1 model beats that matched quick control on IoU or F-score; SAM-only LoRA has marginally lower Chamfer. This control is reported here rather than duplicating the SAM3D table row.

V-JEPA pooling produced one empty prediction: its IoU averages all 200 objects, while aligned IoU, Chamfer and F-scores average the 199 nonempty outputs. All other rows have 200 valid outputs and no empty predictions.

Epoch-1 artifacts: `/work/MathiasElmegaardNielsen#6181/epoch-eval-2026-10-30-11-30/epoch-01/<variant>/results/eval/`, including per-object JSONL and summary JSON. All six summaries declare complete=true, and their unique object-ID sets were checked against the fixed GSO-200 selection. Baseline subset summary: `/work/MathiasElmegaardNielsen#6181/gso-baselines-250-2026-10-29-22-33-26/coarse-baseline-summary-250-and-200.json`.

The earlier full-250 refined-mesh Chamfer / F-score@0.02 results remain: SAM3D 0.0746 / 0.6297; MV-SAM3D 0.0478 / 0.7300; Stream3D 0.0469 / 0.7408. These are separate mesh metrics, not the coarse metrics above. Full refined-mesh evaluations for trained checkpoints are being handled by the evaluation task.

### Synthetic validation: 50 held-out objects, coarse geometry

Latest completed epoch evaluations, checked 2026-10-30 at 11:40 CET. Bold marks the best displayed value per metric. Epochs differ, so these rows are not an equal-training-budget comparison and bold does not establish a final winner. Each epoch has 648 optimizer updates; each full run has five epochs (3,240 updates).


| Model                     | Current update | Evaluated epoch | Loss ↓ | IoU ↑  | Aligned IoU ↑ | Chamfer ↓ | [F-score@0.02](mailto:F-score@0.02) ↑ |
| ------------------------- | -------------- | --------------- | ------ | ------ | ------------- | --------- | ------------------------------------- |
| Fusion adapter            | 1305           | 2               | 0.3089 | **0.1213** | **0.1540**        | **0.2660**    | **0.2513**                                |
| Fusion + SAM LoRA         | 1213           | 1               | 0.2729 | 0.1104 | 0.1361        | 0.3072    | 0.2227                                |
| V-JEPA pooling            | 1402           | 2               | 0.3223 | 0.0367 | 0.0546        | 0.4563    | 0.1326                                |
| V-JEPA pooling + SAM LoRA | 1324           | 2               | **0.2453** | 0.0995 | 0.1080        | 0.3316    | 0.2010                                |
| DINOv2 pooling            | 1467           | 2               | 0.3272 | 0.0587 | 0.0768        | 0.4131    | 0.1537                                |
| SAM-only LoRA             | 923            | 1               | 0.2932 | 0.0926 | 0.1101        | 0.3688    | 0.1984                                |


At the 11:40 CET training snapshot, all six runs had completed epoch-1 synthetic validation and four had completed epoch-2 validation. Synthetic validation is not GSO. The original training scripts schedule final GSO evaluation after five epochs, but a separate evaluation task has now completed the epoch-1 GSO backfills shown above. The earlier statement that no trained-model GSO evaluation existed was outdated.

Measured epoch-validation loss passes took 11–13 seconds for the original five trainable models, and geometry passes took 16–22 seconds across the six models. Together this is roughly 30–40 seconds including small checkpoint overhead. The separate frozen SAM3D coarse GSO-200 evaluation loop took 75 seconds (about 108 seconds from model setup to job completion). These measurements do not estimate the runtime of full refined-mesh evaluation, and the video-conditioned GSO timings are not measured yet.

Training optimizes Stage-1 shape-latent flow matching: a ground-truth shape latent is interpolated with noise at a sampled time, and the model predicts the target flow. Trainable adapters/poolers and optional SAM LoRA receive that loss; later mesh-refinement stages are not trained. Four-step shortcut sampling is an evaluation setting, not the training objective.

Do not compare the two tables as one ranking: their objects differ. The refined-mesh baseline metrics use longest-side-1 normalization; coarse metrics use longest-side-2 normalization. A matched GSO comparison still requires identical object IDs and metric/output settings, and must report the differing sampling and input-view budgets.

# Video-conditioned 3D reconstruction research log

Last reconciled: 2026-10-24. This log records completed work and its limits; the linked run reports retain the detailed measurements and artifact locations.

On 2026-10-24, the training branch was consolidated into `main`. Local contract verification passed 18 tests; six tests were skipped because optional upstream source trees were unavailable. This local check does not repeat the earlier remote GPU runs. Local result copies, datasets, notebook assets, and environments were preserved in the main checkout.

## Research question and controls

Does a masked object video improve coarse canonical geometry, particularly newly revealed surfaces, over one reference image? The pipeline reuses pretrained SAM 3D geometry generation and a V-JEPA video encoder. Controls are untouched native image-only SAM 3D, an adapted model receiving a repeated reference frame, and the same adapted model receiving a full video.

## 2026-10-22: Averaged video-conditioning pilot

Source: `d1cee366ecd246b8639efd63bf813f3408009aa9`. UCloud job `12429369`. Four configurations completed 25 updates each: A connector only, B with video LoRA, C with SAM LoRA, D with both. Two training and two held-out validation objects were active; the underlying video-capable object split was 24/4/4. Each clip supplied 16 frames. Test objects were not evaluated.


| Configuration | Repeated-reference Chamfer | Full-video Chamfer | Newly revealed recall, reference/video |
| ------------- | -------------------------- | ------------------ | -------------------------------------- |
| A             | 0.07351                    | 0.06115            | 14.06% / 16.47%                        |
| B             | 0.06990                    | 0.06112            | 15.47% / 17.64%                        |
| C             | 0.07634                    | 0.06155            | 14.70% / 16.95%                        |
| D             | 0.07694                    | 0.07225            | 15.18% / 13.43%                        |


Untouched native image-only SAM 3D achieved 0.03542 Chamfer, 0.32138 F-score at 0.02, and 34.84% newly revealed recall. Lower Chamfer is better; higher F-score and recall are better. Video improved Chamfer over each adapted repeated-frame control, but every adapted model remained worse than the native image baseline. Newly revealed recall improved for A/B/C and decreased for D. This does not establish a general benefit from video or rank the adapter strategies.

All 24 reconstructions exported successfully. Training throughput was 10.53–12.73 input-frame exposures/s. Integration fixes covered the actual 32 video-capable objects, mixed camera-array types, NAVI positive-Z/positive-image-Y projection, SAM FP32 normalization under BF16, and profiling in training mode. Three failed preparation allocations preceded the successful batch; total allocation time was about 43 minutes 17 seconds. The successful job terminated and released its GPU.

Evidence: [complete results, W&B links, and persistent artifact locations](code/RUN_RESULTS.md), [initial specification](code/RUN_SPEC_INITIAL.md).

## 2026-10-22: Mask-aware attentive-pooler pilot

Source: `32f534db266237125ea7d75be8dec51301509c88`. UCloud job `12429715`. The official attentive pooler used 64 learned queries and foreground tubelet masks; its connector replaced native RGB conditioning while native mask conditioning remained. The same A/B/C/D optional-LoRA configurations completed 25 updates. Pooling, RGB replacement, and video preprocessing changed together, so the comparison cannot isolate their individual effects.


| Configuration | Repeated-reference Chamfer | Full-video Chamfer | Full-video F-score at 0.02 |
| ------------- | -------------------------- | ------------------ | -------------------------- |
| A             | 0.090681                   | 0.103101           | 0.133471                   |
| B             | 0.090678                   | 0.102823           | 0.139235                   |
| C             | 0.090245                   | 0.102898           | 0.138128                   |
| D             | 0.090629                   | 0.102817           | 0.139259                   |


Fixed training losses fell from about 0.547 to 0.483 and validation losses from about 0.950 to 0.920. Geometry nevertheless regressed against both the native baseline (0.035421 Chamfer) and the earlier averaging pilot (best video Chamfer 0.061122). The earlier loss trace used a different validation procedure and is not directly comparable. Every variant generated nondegenerate meshes. Training throughput was 10.64–12.23 input-frame exposures/s.

All 24 contract tests passed at the recorded run, all enabled trainable groups had finite nonzero remote-GPU gradients, all 24 paired reconstruction artifacts contained the required files, and each run exposed 22 direct media panels. The foreground job exited successfully and its provider allocation ended automatically.

Evidence: [complete pooler results and W&B links](code/RUN_RESULTS_POOLER.md), [run specification](code/RUN_SPEC.md), [architecture](architecture.md), [comparison notebook](code/notebooks/compare_reconstructions.ipynb).

## Interpretation and next experiment

Both runs are single-seed integration pilots with only two validation objects and 25 updates. Canonical scan/SAM rotation remains unresolved; normalization does not establish alignment. The data demonstrate a working training/evaluation pipeline and an observed regression, rather than a successful reconstruction improvement.

The next controlled experiment should preserve native RGB/mask conditioning and add video through a zero-initialized gate, checking equality to the native baseline at initialization. A longer single-configuration overfit/calibration test should precede scaling or ranking LoRA choices. These are proposals; no additional experiment is recorded as launched.