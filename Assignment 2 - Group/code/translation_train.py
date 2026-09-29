"""Train the paper-style Transformer on prepared IWSLT'14 de–en BPE text."""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import fire
import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn import functional as F

from src.text_experiment.data import ParallelText, Vocabulary, build_vocabulary, make_batches, read_parallel_split
from src.text_experiment.model import make_transformer
from src.training.results import export_translation
from src.training.common import (
    append_metric,
    code_fingerprints,
    create_run_directory,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_config,
    seed_everything,
    StopRequested,
    TrainingConfig,
)


class TranslationConfig(TrainingConfig):
    data_dir: str
    output_dir: str = "runs/translation"
    sweep: Literal["model_width", "sample_size"] = "model_width"
    embedding_dim: int = Field(default=64, gt=0)
    source_vocab_size: int | None = Field(default=None, gt=4)
    target_vocab_size: int | None = Field(default=None, gt=4)
    max_tokens: int = Field(default=4096, gt=0)
    max_steps: int = Field(default=80_000, gt=0)
    warmup_steps: int = Field(default=4000, gt=0)
    learning_rate_scale: float = Field(default=1.0, gt=0)
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    optimizer: Literal["adam"] = "adam"

    @model_validator(mode="after")
    def check_width(self) -> "TranslationConfig":
        if self.embedding_dim % 8:
            raise ValueError("embedding_dim must be divisible by eight")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(config: TranslationConfig, run_dir: Path, resume: bool) -> tuple[ParallelText, ParallelText, ParallelText]:
    root = Path(config.data_dir)
    splits = {split: read_parallel_split(root, split) for split in ("train", "valid", "test")}
    fingerprints = {f"{split}.{language}": file_sha256(root / f"{split}.{language}") for split in splits for language in ("de", "en")}
    fingerprint_path = run_dir / "data_fingerprints.json"
    if resume:
        if fingerprints != json.loads(fingerprint_path.read_text()):
            raise ValueError("translation data changed since the checkpoint")
    else:
        fingerprint_path.write_text(json.dumps(fingerprints, indent=2) + "\n")
    full_train_pairs = splits["train"]
    train_pairs = full_train_pairs
    if config.sample_count is not None:
        if config.sample_count > len(train_pairs):
            raise ValueError("sample_count exceeds the training split")
        subset_order = torch.randperm(len(train_pairs), generator=torch.Generator().manual_seed(config.seed)).tolist()
        train_pairs = [train_pairs[index] for index in subset_order[: config.sample_count]]
    source_vocab_path = run_dir / "source_vocab.json"
    target_vocab_path = run_dir / "target_vocab.json"
    if resume:
        source_vocab = Vocabulary.load(source_vocab_path)
        target_vocab = Vocabulary.load(target_vocab_path)
    else:
        source_vocab = build_vocabulary([source for source, _ in full_train_pairs], config.source_vocab_size)
        target_vocab = build_vocabulary([target for _, target in full_train_pairs], config.target_vocab_size)
        source_vocab.save(source_vocab_path)
        target_vocab.save(target_vocab_path)
    return (
        ParallelText(train_pairs, source_vocab, target_vocab),
        ParallelText(splits["valid"], source_vocab, target_vocab),
        ParallelText(splits["test"], source_vocab, target_vocab),
    )


def learning_rate(config: TranslationConfig, step: int) -> float:
    update = step + 1
    return config.learning_rate_scale * config.embedding_dim**-0.5 * min(
        update**-0.5, update * config.warmup_steps**-1.5
    )


def evaluate(model: nn.Module, data: ParallelText, config: TranslationConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    nll_sum = 0.0
    token_count = 0
    token_errors = 0
    with torch.no_grad():
        for source, previous, target in make_batches(data, config.max_tokens, config.seed, 0, False):
            source, previous, target = source.to(device), previous.to(device), target.to(device)
            logits = model(source, previous)
            nll_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=data.target_vocab.pad_id, reduction="sum",
            ))
            token_count += int(target.ne(data.target_vocab.pad_id).sum())
            token_errors += int(((logits.argmax(dim=-1) != target) & target.ne(data.target_vocab.pad_id)).sum())
    token_nll = nll_sum / token_count
    return {
        "token_nll": token_nll, "perplexity": math.exp(token_nll),
        "token_error_percent": 100 * token_errors / token_count, "tokens": token_count,
    }


def train(config: TranslationConfig, resume: str | None = None) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = Path(resume) if resume else create_run_directory(
        config.output_dir, f"iwslt14-de-en-d{config.embedding_dim}", config.seed
    )
    if resume:
        saved = TranslationConfig.model_validate_json((run_dir / "config.json").read_text())
        if saved != config:
            raise ValueError("resume config does not match the saved run")
    else:
        save_config(run_dir, config)
    train_data, valid_data, test_data = prepare_data(config, run_dir, bool(resume))
    model = make_transformer(
        len(train_data.source_vocab), len(train_data.target_vocab),
        train_data.source_vocab.pad_id, train_data.target_vocab.pad_id, config.embedding_dim,
    ).to(device)
    metadata_path = run_dir / "run_metadata.json"
    if not resume:
        metadata_path.write_text(json.dumps({
            "train_size": len(train_data),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "source_language": "de", "target_language": "en",
            "loss_unit": "unsmoothed teacher-forced token NLL in natural-log units",
            "error_unit": "percent of non-padding target tokens with wrong argmax",
            "perplexity": "exp(token_nll)",
            "source_row": "null because this is not a row from the authors' CSV",
            "translation_comparability": "original CSV error/loss definitions and optimizer are not fully verified",
            "code_fingerprints": code_fingerprints(Path(__file__).resolve().parent),
        }, indent=2) + "\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(config, 0), betas=(0.9, 0.98), eps=1e-9)
    step, epoch, batch_offset = load_checkpoint(run_dir, model, optimizer) if resume else (0, 0, 0)
    checkpoint_epoch = epoch
    checkpoint_offset = batch_offset
    stop = StopRequested()
    stop.install()
    while step < config.max_steps:
        batches = make_batches(train_data, config.max_tokens, config.seed, epoch, True)
        consumed = 0
        for offset, (source, previous, target) in enumerate(batches):
            consumed += 1
            if offset < batch_offset:
                continue
            if step >= config.max_steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, step)
            model.train()
            source, previous, target = source.to(device), previous.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, previous)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=train_data.target_vocab.pad_id, label_smoothing=config.label_smoothing,
            )
            loss.backward()
            optimizer.step()
            step += 1
            checkpoint_epoch, checkpoint_offset = epoch, offset + 1
            if step % config.eval_every == 0 or step == config.max_steps:
                metric = {
                    "step": step, "epoch": epoch, "train_smoothed_loss": float(loss.detach()),
                    "train": evaluate(model, train_data, config, device),
                    "valid": evaluate(model, valid_data, config, device),
                    "test": evaluate(model, test_data, config, device),
                }
                append_metric(run_dir, metric)
                print(json.dumps(metric), flush=True)
            if step % config.checkpoint_every == 0 or step == config.max_steps:
                save_checkpoint(run_dir, model, optimizer, step, epoch, offset + 1)
            if stop.requested:
                if step % config.eval_every != 0:
                    append_metric(run_dir, {
                        "step": step, "epoch": epoch,
                        "train": evaluate(model, train_data, config, device),
                        "valid": evaluate(model, valid_data, config, device),
                        "test": evaluate(model, test_data, config, device),
                    })
                save_checkpoint(run_dir, model, optimizer, step, epoch, offset + 1)
                export_translation(run_dir, Path(config.results_dir))
                return run_dir
        if consumed == 0:
            raise ValueError("training data produced no batches")
        epoch += 1
        batch_offset = 0
    save_checkpoint(run_dir, model, optimizer, step, checkpoint_epoch, checkpoint_offset)
    append_metric(run_dir, {"step": step, "epoch": checkpoint_epoch, "test": evaluate(model, test_data, config, device)})
    export_translation(run_dir, Path(config.results_dir))
    return run_dir


def run(**kwargs: object) -> str:
    resume = kwargs.pop("resume", None)
    config = TranslationConfig.model_validate(kwargs)
    return str(train(config, str(resume) if resume is not None else None))


if __name__ == "__main__":
    fire.Fire(run)
