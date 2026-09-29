"""Aligned, BPE-tokenized IWSLT'14 German–English sentence pairs."""

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor


SPECIAL_TOKENS = ("<s>", "<pad>", "</s>", "<unk>")


class Vocabulary:
    def __init__(self, tokens: list[str]) -> None:
        if tokens[:4] != list(SPECIAL_TOKENS):
            raise ValueError("vocabulary must begin with the four special tokens")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary contains duplicate tokens")
        self.tokens = tokens
        self.lookup = {token: index for index, token in enumerate(tokens)}
        self.bos_id = 0
        self.pad_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def encode(self, line: str) -> list[int]:
        return [self.lookup.get(token, self.unk_id) for token in line.split()] + [self.eos_id]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text()))

    def __len__(self) -> int:
        return len(self.tokens)


def read_parallel_split(root: Path, split: str) -> list[tuple[str, str]]:
    source_path = root / f"{split}.de"
    target_path = root / f"{split}.en"
    sources = source_path.read_text().splitlines()
    targets = target_path.read_text().splitlines()
    if len(sources) != len(targets):
        raise ValueError(f"{split}: source and target line counts differ")
    if not sources or any(not source.strip() or not target.strip() for source, target in zip(sources, targets)):
        raise ValueError(f"{split}: empty split or empty sentence")
    return list(zip(sources, targets))


def build_vocabulary(lines: list[str], max_size: int | None) -> Vocabulary:
    counts = Counter(token for line in lines for token in line.split())
    ranked = sorted(counts, key=lambda token: (-counts[token], token))
    if max_size is not None:
        if max_size < len(SPECIAL_TOKENS):
            raise ValueError("max_size must include the special tokens")
        ranked = ranked[: max_size - len(SPECIAL_TOKENS)]
    return Vocabulary(list(SPECIAL_TOKENS) + [token for token in ranked if token not in SPECIAL_TOKENS])


class ParallelText:
    def __init__(self, pairs: list[tuple[str, str]], source_vocab: Vocabulary, target_vocab: Vocabulary) -> None:
        self.examples = [(source_vocab.encode(source), target_vocab.encode(target)) for source, target in pairs]
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.examples)


def make_batches(
    data: ParallelText, max_tokens: int, seed: int, epoch: int, shuffle: bool
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed + epoch)).tolist() if shuffle else list(range(len(data)))
    group: list[int] = []
    max_source = 0
    max_target = 0
    for index in order:
        source, target = data.examples[index]
        next_source = max(max_source, len(source))
        next_target = max(max_target, len(target))
        if next_source + next_target > max_tokens:
            raise ValueError("one sentence pair exceeds max_tokens")
        if group and (len(group) + 1) * (next_source + next_target) > max_tokens:
            yield collate(data, group)
            group = []
            max_source = 0
            max_target = 0
        group.append(index)
        max_source = max(max_source, len(source))
        max_target = max(max_target, len(target))
    if group:
        yield collate(data, group)


def collate(data: ParallelText, indices: list[int]) -> tuple[Tensor, Tensor, Tensor]:
    pairs = [data.examples[index] for index in indices]
    source_length = max(len(source) for source, _ in pairs)
    target_length = max(len(target) for _, target in pairs)
    source_batch = torch.full((len(pairs), source_length), data.source_vocab.pad_id, dtype=torch.long)
    previous_batch = torch.full((len(pairs), target_length), data.target_vocab.pad_id, dtype=torch.long)
    target_batch = torch.full((len(pairs), target_length), data.target_vocab.pad_id, dtype=torch.long)
    for row, (source, target) in enumerate(pairs):
        source_batch[row, : len(source)] = torch.tensor(source)
        previous_batch[row, : len(target)] = torch.tensor([data.target_vocab.eos_id] + target[:-1])
        target_batch[row, : len(target)] = torch.tensor(target)
    return source_batch, previous_batch, target_batch
