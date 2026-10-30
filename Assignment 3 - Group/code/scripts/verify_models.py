"""Portable real-GPU integration check for the merged package."""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import fire
import torch
from huggingface_hub import snapshot_download
from video3d.config import load_config, replace_config
from video3d.data.targets import cache_latents
from video3d.models.sam3d import build_sam3d
from video3d.train import train
from video3d.evaluate import evaluate


def real_loss_parity(sam: object, cfg: object, directory: Path) -> None:
    from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching
    from video3d.loss import FlowMatchingLoss
    from video3d.data.samples import ClipDataset
    from video3d.data.targets import load_latent
    sample = ClipDataset(cfg, "train", train=False, clip_len=4)[0]
    inputs = sample["inputs"]
    cond = sam.condition(inputs["ref_rgb"][None].cuda(), inputs["ref_mask"][None].cuda())
    truth = load_latent(cfg, sample["info"]["asset_id"])[None].cuda()
    noise = torch.randn_like(truth)
    tau = torch.full((1,), 0.3, device="cuda")
    layout = {key:torch.zeros(1,*shape,device="cuda") for key,shape in sam.layout_shape.items()}
    scale = sam.generator.time_scale

    def velocity(x: dict[str, torch.Tensor], time: torch.Tensor,
                 conditioning: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prediction,_ = sam.velocity(x["shape"],time/scale,conditioning,layout)
        return {"shape":prediction.float()}

    reference = FlowMatching(reverse_fn=velocity,time_scale=scale,loss_weights={"shape":1.0})
    reference._generate_t = lambda x: tau
    reference._generate_x0 = lambda x: {"shape":noise}
    a = {key:value.detach().clone().requires_grad_(True) for key,value in cond.items()}
    b = {key:value.detach().clone().requires_grad_(True) for key,value in cond.items()}
    with torch.autocast("cuda",dtype=torch.bfloat16):
        expected,_ = reference.loss({"shape":truth},a)
        actual,_ = FlowMatchingLoss(velocity,noise,tau,time_scale=scale).loss({"shape":truth},b)
    grad_a = torch.autograd.grad(expected,tuple(a.values()),allow_unused=True)
    grad_b = torch.autograd.grad(actual,tuple(b.values()),allow_unused=True)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    for left,right in zip(grad_a,grad_b):
        if left is None or right is None:
            assert left is None and right is None
        else:
            torch.testing.assert_close(left,right,rtol=3e-3,atol=5e-5)
    (directory/"loss-parity.json").write_text(json.dumps({"upstream":float(expected),
        "copied":float(actual),"conditioning_gradients_match":True}))


def verify(output: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model verification requires a remote CUDA GPU")
    import subprocess, sys
    subprocess.run([sys.executable,"-m","pytest","-q","tests/gpu/test_rng_gpu.py"],check=True)
    directory=Path(output)
    directory.mkdir(parents=True,exist_ok=True)
    from tests.fixtures.make_fake_dataset import make_fake_dataset
    data=directory/"data"
    if not (data/"train/index.json").is_file():
        data=make_fake_dataset(data)
    checkpoint=Path(snapshot_download("facebook/sam-3d-objects",
        revision="2e73555018d2741ccd486e56c24fac41155a1dc6",
        allow_patterns=["checkpoints/*.yaml","checkpoints/*.ckpt"],
        token=os.environ.get("HF_TOKEN") or os.environ.get("HF_READ_TOKEN")))/"checkpoints"
    cfg=load_config("c3",overrides=[f"paths.data_root={data}",
        f"paths.output_root={directory}/outputs",f"paths.scratch_root={directory}/scratch",
        f"paths.cache_root={directory}/cache",f"paths.sam3d_checkpoint_dir={checkpoint}",
        "train.batch_size=1","train.effective_batch_size=1","train.max_steps=2","data.val_asset_frac=0",
        "train.ckpt_every=1","train.max_seconds=600","data.num_workers=0",
        "data.clip_lengths=[4]","data.clip_length_probs=[1.0]",
        "eval.k=4","sampler.steps=1","logging.media_samples=1",
        "eval.metrics.device=cpu","eval.metrics.max_points=1000","eval.metrics.icp_iterations=10"])
    sam=build_sam3d(cfg,torch.device("cuda"))
    cache_latents(cfg,"train",sam,torch.device("cuda"))
    real_loss_parity(sam,cfg,directory)
    rgb_sets=[name for name in sam.set_names if ("rgb" in name or "image" in name)
              and "mask" not in name and "pointmap" not in name]
    if not rgb_sets:
        raise ValueError(f"No RGB conditioning sets: {sam.set_names}")
    cfg=replace_config(cfg,**{"adapter.query_sets":rgb_sets})
    (directory/"conditioning.json").write_text(json.dumps({"sets":sam.set_names,"rgb_sets":rgb_sets}))
    del sam
    gc.collect()
    torch.cuda.empty_cache()
    summary_path=directory/"summary.json"
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for strategy, lora in [("fusion", False), ("pooler", False), ("fusion", True), ("pooler", True)]:
        key=f"{strategy}-lora{int(lora)}"
        if key in summary:
            continue
        selected=replace_config(cfg,**{"adapter.strategy":strategy,
                                      "adapter.foreground_threshold":0.01,
                                      "encoder.vjepa.video_lora":lora, "sam3d.lora":lora,
                                      "run_name":f"{strategy}-lora{int(lora)}-{directory.name}"})
        record=train(selected,stop_at=1)
        if record["step"] not in {1,2} or train(selected)["step"] != 2:
            raise AssertionError("Training/resume did not complete")
        result=evaluate(selected,str(directory/"outputs"/selected.run_name),limit=1)
        summary[key]={"results":str(result),"steps":2}
        from video3d.storage import atomic_json
        atomic_json(summary_path,summary)
        gc.collect()
        torch.cuda.empty_cache()
    from scripts.verify_generation import check
    check(str(directory/"generation-smoke"))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    fire.Fire(verify)
