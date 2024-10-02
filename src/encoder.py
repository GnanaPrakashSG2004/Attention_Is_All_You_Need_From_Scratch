""" Encoder module for the Transformer model. """

from typing import Optional

import torch
from torch import nn

from src.utils import PositionalEncoding, MultiheadAttention

class EncoderBlock(nn.Module):
    """ Encoder block for the Transformer model. """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ffn:   int,
            dropout: float = 0.1
        ) -> None:
        """ Initialize the encoder block.

        Args:
            d_model: Dimensionality of the model.
            n_heads: Number of attention heads.
            d_ffn:   Dimensionality of the feedforward network.
            dropout: Dropout rate for the model.
        """
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.pre_norm  = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )
        self.post_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass of the encoder block.

        Args:
            x:    Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model).
        """
        attn = self.self_attn(x, x, x, mask)
        x    = x + self.dropout(attn)
        x    = self.pre_norm(x)

        ffn = self.ffn(x)
        x   = x + self.dropout(ffn)
        x   = self.post_norm(x)

        return x

class Encoder(nn.Module):
    """ Encoder class for the Transformer model. """

    def __init__(
            self,
            vocab_size: int,
            seq_len:    int,
            d_model:    int,
            n_heads:    int,
            n_blocks:   int,
            d_ffn:      Optional[int] = None,
            dropout:    float = 0.1
        ) -> None:
        """ Initialize the encoder.

        Args:
            vocab_size: Size of the vocabulary.
            seq_len:    Sequence length for the model.
            d_model:    Dimensionality of the model.
            n_heads:    Number of attention heads.
            n_blocks:   Number of blocks in the model.
            d_ffn:      Dimensionality of the feedforward network.
            dropout:    Dropout rate for the model.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, seq_len)

        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_blocks)
        ])

    def forward(
            self,
            x:    torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """ Forward pass of the encoder.

        Args:
            x:    Input tensor of shape (batch_size, seq_len).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.embedding(x)
        x = self.pos_enc(x)

        for blk in self.blocks:
            x = blk(x, mask)

        return x
