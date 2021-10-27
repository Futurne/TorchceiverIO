"""
Attention blocks used by PercerverIO are implemented here.
"""
import torch
import torch.nn as nn

from typing import Optional


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
        params:
            - input_size: Embedding dimension of the input tokens (C).
            - latent_size: Embedding dimension of the latent tokens (D).
            - project_size : Project dimension for the keys and queries (F).
            - nheads, dropout: Parameters of the `nn.MultiheadAttention`.
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
        params:
            - queries: Query input of shape [batch_size, ntokens_query, latent_size].
            - keys_values: Key-value input of shape [batch_size, ntokens, input_size].
        returns:
            - output: key-value-query attention of shape [batch_size, ntokens_query, output_size].
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
        output_size: Optional[int] = None,
    ):
        """
        params:
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
        params:
            - queries: Query input of shape [batch_size, ntokens_query, latent_size].
            - keys_values: Key-value input of shape [batch_size, ntokens, input_size].
        returns:
            - output : key-value-query attention of shape [batch_size, ntokens_query, output_size].
        """
        # Ne pas normaliser Xkv si C = 1 !
        # Xqkv = self.attn(self.norm_xq(Xq), Xkv)
        Xqkv = self.attn(self.norm_xq(queries), self.norm_xkv(keys_values))
        if keys_values.shape == queries.shape:
            Xqkv = Xqkv + queries
        Xqkv = Xqkv + self.mlp(self.norm_xqkv(Xqkv))
        return Xqkv
