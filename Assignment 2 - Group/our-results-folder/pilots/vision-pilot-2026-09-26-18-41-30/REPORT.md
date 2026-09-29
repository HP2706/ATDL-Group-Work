# A100 CIFAR timing pilot

Run: `vision-pilot-2026-09-26-18-41-30` on Modal profile `mleagent`. The [timing summary](pilot_summary.json) and 36 [vision Parquet endpoints](data/vision/) are saved here. Full checkpoints, data, and epoch timing logs remain in the Modal Volume `atdl-double-descent-vision-pilot` under `runs/vision-pilot-2026-09-26-18-41-30/`.

## Work completed

Each of ResNet18 and the five-layer CNN ran on CIFAR-10 and CIFAR-100 at widths 1, 30, and 64 and fixed label-noise rates 0%, 10%, and 20%: **36 conditions**. Every condition trained for **two full epochs** on the 50,000-image training split with batch size 128, augmentation, seed 0, and clean test labels. ResNet used Adam at `0.0001` with a constant learning rate; CNN used SGD at `0.1` with the existing inverse-square-root schedule. These are short timing/data-path pilots, not paper reproductions.

The first width-64 ResNet/CIFAR-100 run used an NVIDIA A100-SXM4-40GB. The other 35 ran on an NVIDIA A100 80GB PCIe. The table gives the median **second-epoch training wall time** across the three noise settings, in seconds, including the current data loader and optimizer step. It excludes the final evaluation, prediction export, and Modal startup.

| Family | Width 1 | Width 30 | Width 64 |
| --- | ---: | ---: | ---: |
| ResNet18, CIFAR-10 | 7.72 | 9.39 | 10.45 |
| ResNet18, CIFAR-100 | 8.05 | 9.61 | 10.68 |
| Five-layer CNN, CIFAR-10 | 5.85 | 6.74 | 6.64 |
| Five-layer CNN, CIFAR-100 | 6.25 | 6.46 | 7.60 |

## Full-grid compute forecast

For a *hypothetical identical eight-width, three-noise grid* in each family, the unmeasured widths 10, 20, 40, 50, and 60 were estimated by linear interpolation between the measured widths. Multiply the resulting per-epoch times by 4,000 epochs for ResNet. For the CNN, use the current runner's 500,000-update CIFAR-10 and 1,000,000-update CIFAR-100 horizons and 391 updates per full epoch.

| Full grid | Estimated A100 training hours |
| --- | ---: |
| ResNet18, CIFAR-10, 4,000 epochs | 249 |
| ResNet18, CIFAR-100, 4,000 epochs | **256** |
| Five-layer CNN, CIFAR-10, 500,000 updates | 55 |
| Five-layer CNN, CIFAR-100, 1,000,000 updates | 116 |
| **All four hypothetical grids, sequentially** | **676** |

The requested ResNet/CIFAR-100 grid alone is therefore about **10.6 A100 days of training time** on one device, before periodic evaluation, checkpoint writes, startup, or retries. At Modal's published A100 80GB GPU rate of about **$2.50 per hour**, its GPU charge would be roughly **$640 before CPU and memory**. All four hypothetical grids would be roughly **28 A100 days** and **$1,690 in GPU charges alone**. These are planning estimates, not measured full-run bills. [Modal pricing](https://modal.com/pricing) can change.

The full grid's evaluation interval in `resnet_cifar100_sweep.py` is ten epochs. Those evaluations and checkpoint commits add time beyond the table. The two-epoch pilots do not establish long-run throughput, convergence, or double descent. One pilot used a different A100 variant, the unsampled widths are interpolated, and the noise repeats provide a small observed timing range rather than a confidence interval. Reprofile on the exact GPU allocation and final evaluation cadence before committing to the full grid.

The **completed pilot itself** used about **0.25 GPU-hours summed across the 36 training calls**, excluding Modal startup and idle intervals. Its result rows are useful for validating the pipeline and file format, not for comparison with the authors' converged measurements.
