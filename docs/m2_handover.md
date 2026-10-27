# M2 handover: render the 10k train set on UCloud

This brief is for whoever runs M2, and for their coding agent: give the agent this file plus `CLAUDE.md` as its starting context. The commands are in `docs/m2_runbook.md`; this file says what to achieve, where to stop and check in, and what counts as done.

## Goal

Produce the **train split**: 10,000 clips (seeds 1000000-1009999) rendered from filtered Objaverse assets on SA-1B backgrounds. Validate them, and pack them into training shards (RGB, visible masks, poses, voxels) for the model team.

**Deliverables:**
1. Training shards plus `index.json` and the voxels file. Expected 40-50 GB; this is what the model team trains on.
2. A short run report: GPU type, seconds per clip, worker count, rejection rate, failed seeds, test results, total size.
3. The review contact sheets from step 6 of the runbook, for sign-off by Julian.

The full-format clips (about 185 GB, inferred from M1) stay on cluster storage until the shards are accepted, and can then be deleted.

## Context in five sentences

The generator places 1-4 static objects in a scene, orbits a camera around them over 32 frames, renders with Blender/Cycles through BlenderProc, and composites the result over an SA-1B photo. Every clip is validated by geometric tests (T1-T5) that compare the rendered masks and depth against ray-cast ground truth. The eval split (M1, 500 clips on GSO objects) is done and passes everything. M2 is the same pipeline with Objaverse assets and 20x the volume, on bigger GPUs. The only new risk is the hardware: the B200 (Blackwell) needs Blender 4.5 LTS instead of the default 4.2.1, and this has been validated on an RTX 3060 but **not yet on a B200**.

## Steps and checkpoints

Stop at each **checkpoint** and send the listed output to Julian before continuing.

| # | Step | Runbook | Checkpoint |
|---|---|---|---|
| 1 | Clone the repo on a UCloud GPU job; run `scripts/m2/setup_node.sh` | §1 | - |
| 2 | Get the SA-1B backgrounds (your own license and links file) and build the pools | §2 | Pool sizes printed (expect 7,258 train) |
| 3 | Download and preprocess the Objaverse assets; check the filter contact sheet | §2 | **Contact sheet + accept rate** |
| 4 | `scripts/m2/smoke_test.sh` on the B200 | §3 | **Full output** (does Blender render on the B200, s/clip for 1/2/4 workers) |
| 5 | Pilot: seeds 1000000-1000099 | §4 | **failures.jsonl, s/clip, rejection rate, 20 review videos** |
| 6 | Wait for T6 sign-off (Julian); then the full render in seed-range jobs | §4 | Progress once a day is enough |
| 7 | Validate (2% sample + T3/T4 on all), review videos | §5 | **Test summary + contact sheets** |
| 8 | Export shards, check the round trip on a few clips | §6 | **Shard index, total size** |

Step 6 is gated on T6, the SAM 3D convention check, which can still change render settings. Steps 1-5 can start right away.

## UCloud notes (fill in as you learn them)

The scripts assume nothing about the scheduler: each job runs `render_shard.sh START END ROOT WORKERS_PER_GPU`, uses all GPUs it sees, and can be re-run to resume. To size a seed range per job:

```
clips per job ≈ job hours × 3600 × GPUs × workers_per_gpu / (s per clip per worker)
```

Take s/clip from the smoke test, and leave 20% headroom for retries.

- Job type / app used: _
- Storage path for the dataset root (needs ~250 GB and ~3M files): _
- Max job length, GPUs per job: _
- glibc version (`ldd --version`, needs ≥ 2.28): _

## Acceptance criteria

- ≥ 99% of seeds produce a clip. Failed seeds are logged with a reason; replace them with seeds from 1010000 upward and record which.
- `SAM3DVID_SAMPLE_FRAC=0.02` test run: all tests pass. T3, T4 and the format checks run on every clip; T1 and T1b run on the sample.
- Review videos look right to a human (Julian signs off).
- Every clip's `meta.json` says Blender `4.5.15 LTS`.
- Shards: the reader round-trips masks bit-exactly, poses equal the clip JSON, RGB PSNR > 40 dB.

## When something goes wrong

- **The first render fails in the smoke test:** look at the end of the Blender log the script prints.
  - "no compatible GPU" or OptiX/CUDA errors mean the Blender build doesn't support this GPU. Rendering prefers OptiX and falls back to CUDA (`blender/render_clip.py`); the denoiser is OptiX either way.
  - Report before trying fixes. The likely fix is a newer Blender via `blender_custom_path`. Don't switch to CPU rendering (100x slower).
- **`GLIBC_2.xx not found`:** the node's OS is too old for Blender; it needs a container (ask).
- **A job is killed or times out:** run the same command again; finished clips are skipped.
- **Many rejections, e.g. > 30% of clips needing retries:** stop and report. Objaverse assets may be behaving differently from GSO (thin or degenerate meshes); the geometric filter may need tightening.
- **Disk or inode quota errors:** stop; the dataset root needs to move.

Never change sampling or render parameters to make something pass without checking in: the eval split was rendered with them, and train and eval must match.

## For the agent

> You are running M2 for the sam3dvid dataset generator on UCloud. Read `CLAUDE.md`, `docs/m2_handover.md` (this file) and `docs/m2_runbook.md`. Follow the steps table in order. At each checkpoint, summarise the requested output for your user and stop until they confirm. Never print `data/sa1b_links.txt`, never commit data or rendered images, and don't modify sampling, rendering or filter parameters. If a step fails, report the exact error and the relevant log lines instead of working around it.
