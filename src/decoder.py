""" Decoder module for the Transformer model. """

from typing import Optional

import torch
from torch import nn

from src.utils import PositionalEncoding, MultiheadAttention

class DecoderBlock(nn.Module):
    """ Decoder block for the Transformer model. """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ffn:   int,
            dropout: float = 0.1
        ) -> None:
        """ Initialize the decoder block.

        Args:
            d_model: Dimensionality of the model.
            n_heads: Number of attention heads.
            d_ffn:   Dimensionality of the feedforward network.
            dropout: Dropout rate for the model.
        """
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.pre_norm  = nn.LayerNorm(d_model)

        self.cross_attn = MultiheadAttention(d_model, n_heads)
        self.cross_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )
        self.post_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x:          torch.Tensor,
            enc_out:    torch.Tensor,
            self_mask:  Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """ Forward pass of the decoder block.

        Args:
            x:          Input tensor of shape (batch_size, seq_len, d_model).
            enc_out:    Output of the encoder block of shape (batch_size, seq_len_enc, d_model).
            self_mask:  Optional mask tensor of shape (batch_size, seq_len_dec, seq_len_dec).
            cross_mask: Optional mask tensor of shape (batch_size, seq_len_dec, seq_len_enc).

        Returns:
            Decoded tensor of shape (batch_size, seq_len, d_model).
        """
        masked_self_attn = self.self_attn(x, x, x, self_mask)
        x                = x + self.dropout(masked_self_attn)
        x                = self.pre_norm(x)

        cross_attn = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x          = x + self.dropout(cross_attn)
        x          = self.cross_norm(x)

        ffn = self.ffn(x)
        x   = x + self.dropout(ffn)
        x   = self.post_norm(x)

        return x

class Decoder(nn.Module):
    """ Decoder class for the Transformer model. """

    def __init__(
            self,
            vocab_size: int,
            seq_len:    int,
            d_model:    int,
            n_heads:    int,
            n_blocks:   int,
            d_ffn:      int,
            dropout:    float = 0.1
        ) -> None:
        """ Initialize the decoder.

        Args:
            vocab_size: Number of words in the vocabulary.
            seq_len:    Length of the input sequences.
            d_model:    Dimensionality of the model.
            n_heads:    Number of attention heads.
            n_blocks:   Number of blocks in the decoder model.
            d_ffn:      Dimensionality of the feedforward network.
            dropout:    Dropout rate for the model.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, seq_len)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_blocks)
        ])

    def forward(
            self,
            x:          torch.Tensor,
            enc_out:    torch.Tensor,
            self_mask:  Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """ Forward pass of the decoder.

        Args:
            x:          Input tensor of shape (batch_size, seq_len, d_model).
            enc_out:    Output of the encoder block of shape (batch_size, seq_len, d_model).
            self_mask:  Optional mask tensor of shape (batch_size, seq_len_dec, seq_len_dec).
            cross_mask: Optional mask tensor of shape (batch_size, seq_len_dec, seq_len_enc).

        Returns:
            Decoded tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.embedding(x)
        x = self.pos_enc(x)

        for blk in self.blocks:
            x = blk(x, enc_out, self_mask, cross_mask)

        return x
