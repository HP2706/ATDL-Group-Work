# B200 timing pilot for the reduced ResNet sweep

The requested sweep is augmented ResNet18 on CIFAR-10 and CIFAR-100, every integer width from 1 through 40, label noise 0%, 10%, and 20%, one seed per condition, Adam at constant learning rate `1e-4`, batch size 128, and 400 epochs. This is **240 runs**, **96,000 model-epochs**, and **37,536,000 optimizer updates**. The full sweep was **not** launched.

## Measured setup

- UCloud job `12403590`, one full NVIDIA B200, 183,359 MiB visible VRAM, 48 vCPUs allocated by cgroup.
- The repository's `vision_train.py` path, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, four OpenMP/MKL threads per process, and the full 50,000-image training split with 391 batches per epoch.
- Both original CIFAR archives came from the BrainChip mirror. Their MD5 checksums matched torchvision's expected values; torchvision loaded 50,000 train and 10,000 test images for each dataset.
- All pilots used 20% fixed incorrect-label noise unless a run name says otherwise. Noise changes labels, not the image/model workload. Evaluations and checkpoints were scheduled every 3,910 updates (10 epochs), matching the planned measurement interval. All 53 recorded timing runs completed. Per-run times are in [timings.json](timings.json).

## Throughput

| Simultaneous models | Short-run group wall time | Steady epoch time per model | Reading |
| ---: | ---: | ---: | --- |
| 1 | 15–17.5 s for one 3-epoch run | Median 2.48 s | Single-model baseline. |
| 2 | 17.8–18.9 s for two 3-epoch runs | 2.8–3.4 s | Large gain in aggregate throughput. |
| 4 | 27.4 s for four 3-epoch runs | Median 5.47 s | Similar aggregate training throughput to two; better throughput in the longer mixed run. |
| 8 | 49.6 s for eight 3-epoch runs | Median 11.1 s | Little extra aggregate training throughput. |
| 16 | 95.6 s for sixteen 3-epoch runs | Median 23.0 s | No useful aggregate training gain over eight. |

The 40-epoch group ran CIFAR-10 and CIFAR-100 at widths 1 and 40 simultaneously. It completed **160 model-epochs in 227 seconds**, including four periodic evaluations/checkpoints per run and final result exports. A second four-way test with only width-40 models took **86 seconds for 10 epochs**. These runs show that wide models slow down when they share the GPU with other wide models. They also show why the scheduler should immediately fill a free slot with another condition: width-1 models finished well before width-40 models in the mixed group.

## GPU-hour estimate

The measured solo epoch times, interpolated across widths 1–40, imply about **69.3 B200 hours of training** if conditions run one at a time: 32.8 hours for CIFAR-10 and 36.5 hours for CIFAR-100. Periodic evaluation adds roughly 3–4 hours, giving about **73 hours sequentially** before unexpected retries.

With **four concurrent models on one B200**, the 40-epoch mixed group projects to **37.8 GPU-hours** if repeated in batches of four for all 240 conditions and scaled from 40 to 400 epochs. This includes the scheduled evaluations and checkpoints. It also repeats startup/final export overhead ten times, so it is conservative for the same mix of widths. The all-width-40 test projects to about 57 hours if *every* condition were as wide as width 40; the requested grid has only six such conditions. Allowing for the measured width spread and imperfect scheduling, use **40–45 B200 GPU-hours as the working estimate**, or **50–60 GPU-hours as the provisioning budget**. On a single B200, GPU-hours and elapsed hours are approximately the same; four simultaneous models improve utilization rather than multiplying the rented GPU count.

This is an extrapolation from at most 40 epochs, so it does not measure every width or any 400-epoch stability effect. Benchmark results remain on the UCloud work drive at `/work/atdl-b200-benchmark-2026-09-26`; no full sweep was submitted. The interactive UCloud GPU job was left running for follow-up work.
