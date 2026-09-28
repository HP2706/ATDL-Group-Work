# DINO photo assets

The diagrams are editable TikZ directly in `Presentation2/main.tex` and
`Presentation2/snd_presentation.tex`. Compile the presentation normally;
there is no separate figure build.

- `global-a.png`: 256 × 256 global crop.
- `global-a-hires.png`: identical crop coordinates at 512 × 512.
- `local-a.png`: 112 × 112 local crop.
- `dog.jpg`: original [PyTorch Hub example photograph](https://github.com/pytorch/hub/blob/master/images/dog.jpg), downloaded 2026-09-28.
- `prepare_assets.py`: crop coordinates and reproducible resizing; run only when changing the photos, using the project virtual environment.

Technical source: `papers/dinov3/ocr/paper.md`, sections 3, 4.2, 4.3 and
Appendix C. The slides show DINOv3 crop sizes, CLS matching across views,
masked patch prediction, and feature-map downsampling for Gram anchoring.
Prediction bars and feature-grid colors are schematic, not model outputs.
