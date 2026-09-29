"""Standalone training path adapted from Fairseq's 2019 Transformer.

Source: facebookresearch/fairseq v0.9.0, commit
df2f84ce619edddb80e720a56abc74d5490fed99. Adapted from its Transformer,
attention, and sinusoidal-position modules. Copyright (c) Facebook, Inc.
and its affiliates. MIT license: ../../LICENSE.fairseq.

This keeps Fairseq's embeddings, initialization, attention projections,
post-normalization, and teacher-forced encoder-decoder path. Incremental
generation, checkpoint migration, LayerDrop, and other unused options
are omitted.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_embedding(vocab_size: int, embedding_dim: int, padding_idx: int) -> nn.Embedding:
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(embedding.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(embedding.weight[padding_idx], 0)
    return embedding


def make_linear(in_features: int, out_features: int) -> nn.Linear:
    linear = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.constant_(linear.bias, 0)
    return linear


class SinusoidalPositionalEmbedding(nn.Module):
    """Fairseq's sinusoidal positions, with padding positions kept at zero."""

    def __init__(self, embedding_dim: int, padding_idx: int, max_positions: int = 1024) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(max_positions + padding_idx + 1, embedding_dim, padding_idx)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: int) -> Tensor:
        half_dim = embedding_dim // 2
        if half_dim < 2:
            raise ValueError("embedding_dim must be at least four for Fairseq sinusoidal positions")
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float) * (-math.log(10000) / (half_dim - 1))
        )
        angles = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros(num_embeddings, 1)], dim=1)
        embedding[padding_idx, :] = 0
        return embedding

    def forward(self, tokens: Tensor) -> Tensor:
        mask = tokens.ne(self.padding_idx).long()
        positions = mask.cumsum(dim=1) * mask + self.padding_idx
        max_position = tokens.size(1) + self.padding_idx + 1
        if max_position > self.weights.size(0):
            self.weights = self.get_embedding(max_position, self.embedding_dim, self.padding_idx)
        weights = self.weights.to(device=tokens.device)
        batch_size, sequence_length = tokens.shape
        return weights.index_select(0, positions.reshape(-1)).view(batch_size, sequence_length, -1).detach()


class MultiheadAttention(nn.Module):
    """Fairseq's separate Q/K/V projections and scaled initialization."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        gain = 1 / math.sqrt(2)
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(projection.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        output, _ = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            None, torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            None, None, False, 0.0, self.out_proj.weight, self.out_proj.bias,
            self.training, key_padding_mask, False, attention_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )
        return output


class TransformerEncoderLayer(nn.Module):
    """The Fairseq post-normalization encoder layer."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(embedding_dim, 8)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = make_linear(embedding_dim, 4 * embedding_dim)
        self.fc2 = make_linear(4 * embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor, padding_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.self_attn_layer_norm(residual + x)
        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        return self.final_layer_norm(residual + x)


class TransformerDecoderLayer(nn.Module):
    """The Fairseq post-normalization decoder layer with cross-attention."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(embedding_dim, 8)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_attn = MultiheadAttention(embedding_dim, 8)
        self.encoder_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = make_linear(embedding_dim, 4 * embedding_dim)
        self.fc2 = make_linear(4 * embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        encoder_padding_mask: Tensor | None,
        self_padding_mask: Tensor | None,
        future_mask: Tensor,
    ) -> Tensor:
        residual = x
        x = self.self_attn(x, x, x, key_padding_mask=self_padding_mask, attention_mask=future_mask)
        x = self.self_attn_layer_norm(residual + x)
        residual = x
        x = self.encoder_attn(x, encoder_output, encoder_output, key_padding_mask=encoder_padding_mask)
        x = self.encoder_attn_layer_norm(residual + x)
        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        return self.final_layer_norm(residual + x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, embedding_dim: int) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embed_tokens = make_embedding(vocab_size, embedding_dim, padding_idx)
        self.embed_positions = SinusoidalPositionalEmbedding(embedding_dim, padding_idx)
        self.embed_scale = math.sqrt(embedding_dim)
        self.layers = nn.ModuleList([TransformerEncoderLayer(embedding_dim) for _ in range(6)])

    def forward(self, source_tokens: Tensor) -> tuple[Tensor, Tensor | None]:
        x = self.embed_scale * self.embed_tokens(source_tokens) + self.embed_positions(source_tokens)
        x = x.transpose(0, 1)
        padding_mask = source_tokens.eq(self.padding_idx)
        if not bool(padding_mask.any()):
            padding_mask = None
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x, padding_mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, embedding_dim: int) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embed_tokens = make_embedding(vocab_size, embedding_dim, padding_idx)
        self.embed_positions = SinusoidalPositionalEmbedding(embedding_dim, padding_idx)
        self.embed_scale = math.sqrt(embedding_dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer(embedding_dim) for _ in range(6)])
        self.embed_out = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.normal_(self.embed_out, mean=0, std=embedding_dim**-0.5)

    def forward(
        self,
        previous_output_tokens: Tensor,
        encoder_output: Tensor,
        encoder_padding_mask: Tensor | None,
    ) -> Tensor:
        x = self.embed_scale * self.embed_tokens(previous_output_tokens) + self.embed_positions(previous_output_tokens)
        x = x.transpose(0, 1)
        self_padding_mask = previous_output_tokens.eq(self.padding_idx)
        if not bool(self_padding_mask.any()):
            self_padding_mask = None
        sequence_length = x.size(0)
        future_mask = torch.triu(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=x.device), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_padding_mask, self_padding_mask, future_mask)
        return F.linear(x.transpose(0, 1), self.embed_out)


class TransformerModel(nn.Module):
    """Fairseq-style encoder-decoder for teacher-forced translation training."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_padding_idx: int,
        target_padding_idx: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 8:
            raise ValueError("embedding_dim must be a positive multiple of eight")
        if not 0 <= source_padding_idx < source_vocab_size:
            raise ValueError("source_padding_idx must be within the source vocabulary")
        if not 0 <= target_padding_idx < target_vocab_size:
            raise ValueError("target_padding_idx must be within the target vocabulary")
        self.encoder = TransformerEncoder(source_vocab_size, source_padding_idx, embedding_dim)
        self.decoder = TransformerDecoder(target_vocab_size, target_padding_idx, embedding_dim)

    def forward(self, source_tokens: Tensor, previous_output_tokens: Tensor) -> Tensor:
        encoder_output, encoder_padding_mask = self.encoder(source_tokens)
        return self.decoder(previous_output_tokens, encoder_output, encoder_padding_mask)


def make_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_padding_idx: int,
    target_padding_idx: int,
    embedding_dim: int,
) -> TransformerModel:
    """Build the paper's stated six-layer, eight-head, zero-dropout model."""
    return TransformerModel(
        source_vocab_size, target_vocab_size, source_padding_idx, target_padding_idx, embedding_dim
    )
