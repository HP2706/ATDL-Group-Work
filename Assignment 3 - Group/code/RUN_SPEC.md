# Mask-aware pooler rerun specification

Status: completed as UCloud job 12429715, source `32f534db266237125ea7d75be8dec51301509c88`. All four runs completed 25 updates, evaluation and verified media uploads; the batch exited zero and released its allocation. See [validated results](RUN_RESULTS_POOLER.md). The run used one small UCloud B200 MIG allocation with a two-hour provider cap. Source is committed and pushed before submission. The earlier completed specification is [preserved here](RUN_SPEC_INITIAL.md).

Architecture: [mask-aware pooler](../../architecture.md). Reuse pinned upstream model/loss code and cached NAVI geometry/reference targets. New runtime input is full-scene RGB with aligned foreground masks, 16 frames at 384 square. Official V-JEPA attentive pooler: depth 1, 64 queries, width 768, 12 heads; 6,546,432 trainable parameters. Linear connector: 787,456. Replace native RGB tokens, retain mask tokens and disabled pointmap slots. Condition length 4852 instead of the prior appended length 8104.

| Variant | Trainable components | Parameters |
|---|---|---:|
| A | pooler and connector | 7,333,888 |
| B | plus video LoRA | 7,776,256 |
| C | plus SAM LoRA | 8,710,144 |
| D | plus both LoRAs | 9,152,512 |

Same 2 train / 2 validation objects, up to 2 clips each; object-disjoint pools 24/4/4, test untouched. Same AdamW 1e-4, zero weight decay, 5-update warmup, cosine schedule, clipping 1, seed 42, 25 updates, effective batch 4. New pooler weights are FP32 under BF16 autocast. Frozen backbones are BF16 with FP32 native normalization. Checkpoint all trainable parameters, optimizer, scheduler, RNG and progress every 5 updates and at exit.

Profile complete optimizer steps at physical batch 1, 2 and 4, preserving 2 GiB memory headroom. Added maximum FP32 trainable weight/gradient/Adam storage is about 140 MiB; attention activation cost is expected to fall with the shorter conditioning but must be measured. Previous training throughput was 10.5–12.7 input frames/s; new throughput is unmeasured. No training or benchmark runs locally.

Fixed-seed validation flow loss across every held-out clip and one fixed training probe at step 0 and every 5 updates. Final generation: 25 sampler steps, 2 held-out objects, native image, repeated reference and full video controls with matched generation seeds. Report Chamfer, F-score, occupancy IoU, regional recall and failures. Interpret absolute geometry metrics cautiously because scan/SAM canonical rotation is not established.

W&B: hprjdk/atdl-video-to-3d. Direct final video and actual GLB mesh panels, reference RGB/mask, clearly labeled point-cloud comparison table, downloadable paired reconstruction artifacts, and separate visible training-object assets with encoded targets. Check first-update gradients for pooler, connector and enabled LoRA groups. Verify all media remotely after upload.

One foreground native UCloud batch runs preparation reuse, profiling and A–D sequentially, then persists results and releases allocation. Provider cap 2 hours; foreground cap 115 minutes; profiling cap 5 minutes; per-variant training cap 18 minutes. Cleanup does not depend on this laptop staying online. Verify provider terminal state, script exit code and completed sweep marker before reporting success.
