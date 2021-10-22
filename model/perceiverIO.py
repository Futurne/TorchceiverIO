"""
PerceiverIO implementation.
All the blocks used by PerceiverIO are also implemented here.
"""
from typing import Optional

import torch
import torch.nn as nn

import einops


class AttentionModule(nn.Module):
    """Class that implements a custom Attention module.
    This module is used by the `CrossAttention` layer. It does the specific
    queries-keys-values attentions.
    """
    def __init__(
        self, 
        input_size: int,
        latent_size: int,
        project_size: int,
        nheads: int,
        dropout: int,
        output_size: int,
    ):
        """
        params : 
            - input_size: Embedding dimension of the input tokens (C).
            - latent_size: Embedding dimension of the latent tokens (D).
            - project_size : Project dimension for the keys and queries (F).
            - nhead, dropout: Parameters of the `nn.MultiheadAttention`.
            - output_size : Output embedding dimension (E).
        """
        super().__init__()

        self.project_queries = nn.Linear(latent_size, project_size)
        self.project_keys = nn.Linear(input_size, project_size)
        self.project_values = nn.Linear(input_size, project_size)

        self.mha = nn.MultiheadAttention(
            embed_dim=project_size,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )

        self.project_mha = nn.Linear(project_size, output_size)

    def forward(
        self,
        queries: torch.tensor,
        keys_values: torch.tensor,
    ):
        """
        params
            - queries: Query input of shape [N, D]
            - keys_values: Key-value input of shape [M, C]
        returns :
            - output: key-value-query attention of shape [N,D] (encoder) or [O,E] (decoder)
        """
        Q = self.project_queries(queries)
        K = self.project_keys(keys_values)
        V = self.project_values(keys_values)

        output, attn_scores = self.mha(Q, K, V)
        return self.project_mha(output)


class CrossAttention(nn.Module):
    """Class that implements cross attention with GELU activation and 2 MLP.
    This module is used by the encoder and the decoder.
    By default, the output size is the same as the latent size of the queries.
    However, the decoder may need to change the output size, which can be done
    with the value `output_size`.
    """
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        project_size: int,
        nheads: int,
        dropout: int,
        output_size: Optional[int] = None
    ):
        """
        params :
            - input_size: Embedding dimension of the input tokens (C).
            - latent_size: Embedding dimension of the latent tokens (D).
            - project_size : Project dimension for the keys and queries (F).
            - nheads, dropout: Parameters of the `nn.MultiheadAttention`.
            - output_size : Output embedding dimension (E).
                Default value is None (sets the output size to the latent size).
        """
        super().__init__()
        output_size = latent_size if output_size is None else output_size

        self.attn = AttentionModule(input_size, latent_size, project_size, nheads, dropout, output_size)
        self.norm_xq = nn.LayerNorm(latent_size)
        self.norm_xkv = nn.LayerNorm(input_size)

        self.norm_xqkv = nn.LayerNorm(output_size)

        self.mlp = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, queries, keys_values):
        """
        params :
            - queries: Query input of shape [N, D]
            - keys_values: Key-value input of shape [M, C]
        returns :
            - output : key-value-query attention of shape [N,D] (encoder) or [O,E] (decoder)
        """
        # Ne pas normaliser Xkv si C = 1 !
        # Xqkv = self.attn(self.norm_xq(Xq), Xkv)
        Xqkv = self.attn(self.norm_xq(queries), self.norm_xkv(keys_values))
        if keys_values.shape == queries.shape:
            Xqkv = Xqkv + queries
        Xqkv = Xqkv + self.mlp(self.norm_xqkv(Xqkv))
        return Xqkv


class PerceiverProcess(nn.Module):
    """Perceiver Process class."""
    def __init__(
        self,
        latent_size: int,
        nlayers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: int,
    ):
        """
        Args:
            - latent_size: Embedding dimension of the latent tokens (D).
            - nlayers: Number of transformer encoder's layers (L).
            - nhead, dim_feedforward, dropout: Parameters of the `nn.TransformerEncoderLayer`.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = latent_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.process = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args
            - x: Latent array of shape [batch_size, seq_len, latent_size].
        Return
            - self_attention: Over x.
                Output shape is [batch_size, seq_len, latent_size].
        """
        return self.process(x, src_key_padding_mask=src_key_padding_mask)


class PerceiverIO(nn.Module):
    """Perceiver IO class implementation."""

    def __init__(
        self,
        nlatents: int,
        input_size: int,
        project_size: int,
        latent_size: int,
        out_emb_size: int,
        nlayers: int,
        output_dim: int,
        nheads: int,
        dim_feedforward: int,
        dropout: int,
    ):
        """
        params :
            - nlatents: Number of latent tokens (N).
            - input_size: Embedding dimension of the input tokens (C).
            - project_size : Project dimension for the keys and queries (F).
            - latent_size: Embedding dimension of the latent tokens (D).
            - out_emb_size : Output embedding dimension (E). Default to None for the Encoder.
            - nlayers: Number of transformer encoder's layers (L).
            - output_dim: Dimension of output query (O).
            - nhead, dim_feedforward, dropout: Parameters of the `nn.TransformerEncoderLayer`.
        """

        super().__init__()
        self.latent_array = nn.Parameter(torch.randn(nlatents, latent_size))
        nn.init.xavier_uniform_(self.latent_array)

        self.encoder = CrossAttention(input_size,latent_size,project_size, nheads, dropout)
        self.process = PerceiverProcess(latent_size, nlayers, nheads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = CrossAttention(latent_size, output_dim, project_size, nheads, dropout, output_size=out_emb_size)

    def forward(self, x, query):
        """Forward."""
        batch_size = x.shape[0]
        Xq = einops.repeat(self.latent_array, 's d -> b s d', b=batch_size)

        x = self.encoder(Xq, x)
        x = self.process(x)
        x = self.decoder(query, x)
        return x
