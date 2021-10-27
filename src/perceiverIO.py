"""
PerceiverIO implementation.
All the blocks used by PerceiverIO are also implemented here.
"""
import torch
import torch.nn as nn

import einops

from .attention import CrossAttention


class PerceiverIO(nn.Module):
    """Main PerceiverIO module.

    Encapsulate the encoder, process and decoder modules.
    """

    def __init__(
        self,
        nlatents: int,  # N
        input_size: int,  # C
        project_size: int,  # F
        latent_size: int,  # D
        output_size: int,  # E
        nlayers: int,  # L
        decoder_query_size: int,  # O
        nheads: int,
        dim_feedforward: int,
        dropout: int,
    ):
        """
        params:
            - nlatents: Number of latent tokens (N).
            - input_size: Embedding dimension of the input tokens (C).
            - project_size: Project dimension for the keys and queries (F).
            - latent_size: Embedding dimension of the latent tokens (D).
            - output_size: Output embedding dimension (E).
            - nlayers: Number of transformer process's layers (L).
            - decoder_query_size: Dimension of the output query (O).
            - nheads, dim_feedforward, dropout: Parameters of the `nn.TransformerEncoderLayer`.
        """
        super().__init__()
        self.latent_array = nn.Parameter(torch.randn(nlatents, latent_size))
        nn.init.xavier_uniform_(self.latent_array)

        self.encoder = CrossAttention(
            input_size,
            latent_size,
            project_size,
            nheads,
            dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_size,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.process = nn.TransformerEncoder(encoder_layer, nlayers)

        self.decoder = CrossAttention(
            latent_size,
            decoder_query_size,
            project_size,
            nheads,
            dropout,
            output_size=output_size
        )

    def forward(self, input_tokens, decoder_queries):
        """Project `input_tokens` to the latent embedding,
        and decode those embeddings with the `decoder_queries`.

        params:
            - input_tokens: input tokens of shape [batch_size, ntokens_input, input_size].
            - decoder_queries: decoder queries of shape [batch_size, ntokens_output, decoder_query_size].
        returns:
            - output: PerceiverIO's output of shape [batch_size, ntokens_output, output_size].
        """
        batch_size = input_tokens.shape[0]
        latent_queries = einops.repeat(self.latent_array, 's d -> b s d', b=batch_size)

        latent_tokens = self.encoder(latent_queries, input_tokens)
        latent_tokens = self.process(latent_tokens)
        return self.decoder(decoder_queries, latent_tokens)
