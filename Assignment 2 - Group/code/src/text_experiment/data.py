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

    # Match Fairseq Dictionary.finalize():
    # frequency descending, lexical ordering for equal-frequency tokens.
    ranked = sorted(
        (
            token
            for token in counts
            if token not in SPECIAL_TOKENS
        ),
        key=lambda token: (-counts[token], token),
    )

    if max_size is not None:
        if max_size < len(SPECIAL_TOKENS):
            raise ValueError("max_size must include the special tokens")
        ranked = ranked[: max_size - len(SPECIAL_TOKENS)]

    tokens = list(SPECIAL_TOKENS) + ranked

    # Fairseq Dictionary.finalize() pads the dictionary size
    # to a multiple of 8 by default.
    madeup_index = 0
    while len(tokens) % 8 != 0:
        token = f"madeupword{madeup_index:04d}"
        if token not in counts and token not in tokens:
            tokens.append(token)
        madeup_index += 1

    return Vocabulary(tokens)

class ParallelText:
    def __init__(self, pairs: list[tuple[str, str]], source_vocab: Vocabulary, target_vocab: Vocabulary) -> None:
        self.examples = [(source_vocab.encode(source), target_vocab.encode(target)) for source, target in pairs]
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.examples)


def make_batches(
    data: ParallelText,
    max_tokens: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    generator = torch.Generator().manual_seed(seed + epoch)

    if shuffle:
        order = torch.randperm(len(data), generator=generator).tolist()
    else:
        order = list(range(len(data)))

    # Match Fairseq LanguagePairDataset.ordered_indices():
    # shuffle first, then stable-sort by target length and source length.
    order.sort(key=lambda index: len(data.examples[index][1]))
    order.sort(key=lambda index: len(data.examples[index][0]))

    batches: list[list[int]] = []
    group: list[int] = []
    max_num_tokens = 0

    for index in order:
        source, target = data.examples[index]

        # Fairseq LanguagePairDataset.num_tokens(index)
        # returns max(source_length, target_length).
        sample_tokens = max(len(source), len(target))

        if sample_tokens > max_tokens:
            raise ValueError("one sentence pair exceeds max_tokens")

        next_max_num_tokens = max(max_num_tokens, sample_tokens)
        next_batch_size = len(group) + 1

        if group and next_batch_size * next_max_num_tokens > max_tokens:
            # Fairseq normally requires batch sizes to be multiples of 8.
            if len(group) >= 8:
                split = len(group) - (len(group) % 8)

                if split > 0 and split < len(group):
                    batches.append(group[:split])
                    group = group[split:]

                    if group:
                        max_num_tokens = max(
                            max(
                                len(data.examples[i][0]),
                                len(data.examples[i][1]),
                            )
                            for i in group
                        )
                    else:
                        max_num_tokens = 0
                else:
                    batches.append(group)
                    group = []
                    max_num_tokens = 0
            else:
                batches.append(group)
                group = []
                max_num_tokens = 0

            next_max_num_tokens = max(max_num_tokens, sample_tokens)

        group.append(index)
        max_num_tokens = next_max_num_tokens

    if group:
        batches.append(group)

    if shuffle:
        permutation = torch.randperm(
            len(batches),
            generator=generator,
        ).tolist()
        batches = [batches[i] for i in permutation]

    for indices in batches:
        yield collate(data, indices)


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
