"""Defines PerceiverIO variant for images."""
import torch
import torch.nn as nn

import einops

from .perceiverIO import PerceiverIO


class PerceiverIOImage(PerceiverIO):
    """Module built uppon PerceiverIO to defines some default behaviour with images.

    Add a learnable positional encoding for all pixels.
    Concatenate those embeddings to the RGB channels of the pixels.

    Create a unique learnable decoder query token.
    The token is unique, so this model should be used for classification.
    """
    def __init__(
        self,
        im_size: int,
        decoder_query_size: int,
        pos_encoding_size: int,
        channel_size: int,
        **kwargs,
    ):
        """
        params:
            - im_size: image length.
                The image is supposed to be a square of shape [im_size, im_size].
            - pos_encoding_size: Embedding dimension of the positional encodings.
            - channel_size: Number of channel per pixel (3 for 'RGB').

        The rest of the parameters are the same as the PerceiverIO.
        Do not provide an `input_size` parameter.

        Since the positional encodings are concatenated, the final `input_size`
        for the PerceiverIO is `channel_size` + `pos_encoding_size`.
        """
        super(PerceiverIOImage, self).__init__(
            input_size=channel_size + pos_encoding_size,
            decoder_query_size=decoder_query_size,
            **kwargs
        )

        self.pos_encoding = nn.Parameter(torch.randn(im_size * im_size, pos_encoding_size))
        self.query_output = nn.Parameter(torch.randn(1, decoder_query_size))

        nn.init.xavier_uniform_(self.query_output)
        nn.init.xavier_uniform_(self.pos_encoding)

    def forward(self, x) :
        """
        parms :
            - x : image of shape [batch_size, channel_size, width, height]
        returns :
            - PerceiverIO prediction of size [batch_size, output_size]
        """
        batch_size = x.shape[0]
        pos_encoding = einops.repeat(self.pos_encoding, 'i e -> b i e', b=batch_size)
        query = einops.repeat(self.query_output, 's d -> b s d', b=batch_size)
        x = einops.rearrange(x, 'b x w h -> b (w h) x')  # [batch_size, im_size * im_size, input_size]

        x = torch.cat((x, pos_encoding), dim=2)  #  [batch_size, im_size * im_size, input_size + pos_encoding_size]

        x = super(PerceiverIOImage, self).forward(x, query)
        return x.squeeze(dim=1)  # Remove unique token dimension
