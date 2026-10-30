# SAM 3D baseline verification, 29 October 2026

Both installed upstream libraries completed real pretrained-checkpoint inference
and our shared dataset/metric evaluation on UCloud B200 job
[12439107](https://cloud.sdu.dk/app/jobs/properties/12439107).
The ten-hour interactive workstation is retained until 30 October, 07:56 CET.
The task's Modal apps were stopped before moving verification to UCloud.

Remote checkout: ~/atdl-baselines, based on source commit
e6125cf8fda35c3bf46773f296a879098970d1ad, with the local source delta uploaded
and checked by SHA-256. Model and dataset caches were reused from the existing
shared training directory. Each upstream fork has its own uv environment.

Runtime: PyTorch 2.7.1+cu128, NVIDIA B200 (183359 MiB).
Checkpoint revision: 2e73555018d2741ccd486e56c24fac41155a1dc6.
Both runs used the first selected training sample, four frames, four shortcut
sampling steps, matched per-sample seeds, supplied masks and native MoGe
pointmaps. Stream3D used chunk_size=2, topk=2 and memory_depth=5.

| Check | MV-SAM3D | Stream3D |
|---|---:|---:|
| Backend | SAM 3D | SAM 3D |
| Scope | Coarse Stage 1 | Coarse Stage 1 |
| Occupied cells in the 64³ output | 11,192 | 16,561 |
| Final selected input-frame positions | 0, 1, 2, 3 | 0, 3 |
| Selected positions after each chunk | 0, 1, 2, 3 | 0, 1 → 0, 3 |
| Reconstruction time, excluding model load and metrics | 7.68 s | 19.58 s |
| Shared geometry metrics completed | Yes | Yes |

The Stream3D selection demonstrates retention of an earlier-chunk frame.
These are single-sample smoke tests, not estimates of relative reconstruction
quality or throughput. The methods use different inference-time view budgets.
No texture, layout optimization, full paper protocol or held-out benchmark
claim is implied.

Evidence copies:
[MV-SAM3D evaluation](../results/baseline-verification-2026-10-29/mvsam3d-evaluation.jsonl),
[Stream3D evaluation](../results/baseline-verification-2026-10-29/stream3d-evaluation.jsonl),
and corresponding occupancy arrays in the same ignored results directory.
Persistent remote results:
/work/MathiasElmegaardNielsen#6181/sam3d-baselines/{mvsam3d-smoke,stream3d-smoke}/.

Six focused baseline tests pass locally, including complete dataset-to-result
evaluation with an injected generator, malformed inputs and Hugging Face
snapshot symlinks. Ruff and the uv lockfile checks pass. uv refuses to install
the two mutually exclusive extras into one environment.

The real run caught and fixed a checkpoint-path bug: resolving a symlinked
pipeline.yaml before selecting its parent pointed at Hugging Face blobs/,
rather than its snapshot directory. A regression test covers that case.
