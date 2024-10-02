""" Training pipeline for the English to French machine translation task using the Transformer model. """

from typing import List, Optional

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.encoder import Encoder
from src.decoder import Decoder
from src.utils import (
    CorpusDataset, TranslationDataset,
    get_padding_mask, get_encoder_mask, get_decoder_mask
)

class Transformer(nn.Module):
    """ Transformer model for machine translation. """

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            enc_seq_len:    int,
            dec_seq_len:    int,
            d_model:        int,
            n_heads:        int,
            n_blocks:       int,
            d_ffn:          int,
            dropout:        float = 0.1
        ) -> None:
        """ Initialize the Transformer model.

        Args:
            src_vocab_size: Source vocabulary size.
            tgt_vocab_size: Target vocabulary size.
            enc_seq_len:    Sequence length for the encoder model.
            dec_seq_len:    Sequence length for the decoder model.
            d_model:        Dimensionality of the model.
            n_heads:        Number of attention heads.
            n_blocks:       Number of encoder/decoder blocks.
            d_ffn:          Dimensionality of the feedforward network.
            dropout:        Dropout rate for the model.
        """
        super().__init__()

        self.encoder = Encoder(src_vocab_size, enc_seq_len, d_model, n_heads, n_blocks, d_ffn, dropout)
        self.decoder = Decoder(tgt_vocab_size, dec_seq_len, d_model, n_heads, n_blocks, d_ffn, dropout)

        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(
            self,
            src:            torch.Tensor,
            tgt:            torch.Tensor,
            src_mask:       torch.Tensor,
            trg_self_mask:  torch.Tensor,
            trg_cross_mask: torch.Tensor
        ) -> torch.Tensor:
        """ Forward pass of the Transformer model.

        Args:
            src:            Source input tensor of shape (batch_size, src_len).
            tgt:            Target input tensor of shape (batch_size, tgt_len).
            src_mask:       Source mask tensor of shape (batch_size, src_len, src_len).
            trg_self_mask:  Target self-attn mask tensor of shape (batch_size, tgt_len, tgt_len).
            trg_cross_mask: Target cross-attn mask tensor of shape (batch_size, tgt_len, src_len).

        Returns:
            Output tensor of shape (batch_size, tgt_len, tgt_vocab_size).
        """
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, trg_self_mask, trg_cross_mask)

        out = self.out(dec_out)

        return out

def train(
        model:      Transformer,
        optimizer:  torch.optim.Optimizer,
        criterion:  nn.Module,
        loader:     DataLoader,
        device:     torch.device,
        epoch:      int,
        num_epochs: int
    ) -> List[float]:
    """ Training pipeline for the Transformer model.

    Args:
        model:      Transformer model to train.
        optimizer:  Optimizer for the model.
        criterion:  Loss function for the model.
        loader:     DataLoader for the training dataset.
        device:     Device to run the model on.
        epoch:      Current epoch number.
        num_epochs: Total number of epochs.

    Returns:
        List of losses for each batch per epoch.
    """
    model.train()

    total_loss = 0
    losses = []

    pbar = tqdm(loader, desc=f"Train Epoch: {epoch + 1}/{num_epochs}")

    for i, (en_batch, fr_batch) in enumerate(pbar):
        en_seqs, en_lens = en_batch

        enc_inp  = en_seqs.to(device)
        enc_mask = get_encoder_mask(en_lens, en_seqs.size(1)).to(device)

        fr_seqs, fr_lens = fr_batch
        fr_seqs = fr_seqs.to(device)

        dec_inp        = fr_seqs[:, :-1]
        dec_out        = fr_seqs[:, 1:]

        dec_self_mask  = get_decoder_mask(fr_lens, dec_inp.size(1)).to(device)
        dec_cross_mask = get_padding_mask(en_lens, en_seqs.size(1), dec_inp.size(1)).to(device)

        out = model(enc_inp, dec_inp, enc_mask, dec_self_mask, dec_cross_mask)

        loss = criterion(out.reshape(-1, out.size(-1)), dec_out.reshape(-1))

        total_loss += loss.item()
        pbar.set_postfix_str(f"Running Avg Loss: {total_loss / (i + 1):.4f}")
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pbar.set_postfix_str(f"Train Loss: {total_loss / len(loader):.4f}")

    return losses

@torch.no_grad()
def validate(
        model:      Transformer,
        criterion:  nn.Module,
        loader:     DataLoader,
        device:     torch.device,
        epoch:      Optional[int] = None,
        num_epochs: Optional[int] = None
    ) -> List[float]:
    """ Validation pipeline for the Transformer model.
    
    Args:
        model:      Transformer model to validate.
        criterion:  Loss function for the model.
        loader:     DataLoader for the validation dataset.
        device:     Device to run the model on.
        epoch:      Current epoch number.
        num_epochs: Total number of epochs.

    Returns:
        List of losses for each batch per epoch.
    """
    model.eval()

    total_loss = 0
    losses = []

    pbar = tqdm(loader)

    if epoch is not None and num_epochs is not None:
        pbar.set_description_str(f"Validation Epoch: {epoch + 1}/{num_epochs}")
    else:
        pbar.set_description_str("Validating using pre-trained model")

    for i, (en_batch, fr_batch) in enumerate(pbar):
        en_seqs, en_lens = en_batch

        enc_inp  = en_seqs.to(device)
        enc_mask = get_encoder_mask(en_lens, en_seqs.size(1)).to(device)

        fr_seqs, fr_lens = fr_batch
        fr_seqs = fr_seqs.to(device)

        dec_inp        = fr_seqs[:, :-1]
        dec_out        = fr_seqs[:, 1:]
        dec_self_mask  = get_decoder_mask(fr_lens, dec_inp.size(1)).to(device)
        dec_cross_mask = get_padding_mask(en_lens, en_seqs.size(1), dec_inp.size(1)).to(device)

        out = model(enc_inp, dec_inp, enc_mask, dec_self_mask, dec_cross_mask)

        loss = criterion(out.reshape(-1, out.size(-1)), dec_out.reshape(-1))

        total_loss += loss.item()
        losses.append(loss.item())

        pbar.set_postfix_str(f"Running Avg Loss: {total_loss / (i + 1):.4f}")

    pbar.set_postfix_str(f"Validation Loss: {total_loss / len(loader):.4f}")

    return losses

def parse_args() -> argparse.Namespace:
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Train a Transformer model for machine translation.")

    parser.add_argument("--train_en_path", type=str,   default="corpus/processed_train.en", help="Path to the English train dataset.")
    parser.add_argument("--train_fr_path", type=str,   default="corpus/processed_train.fr", help="Path to the French train dataset.")
    parser.add_argument("--dev_en_path",   type=str,   default="corpus/processed_dev.en",   help="Path to the English dev dataset.")
    parser.add_argument("--dev_fr_path",   type=str,   default="corpus/processed_dev.fr",   help="Path to the French dev dataset.")
    parser.add_argument("--model_path",    type=str,   default="weights/transformer.pt",    help="Path to save the trained model.")
    parser.add_argument("--d_model",       type=int,   default=512,                         help="Dimensionality of the model.")
    parser.add_argument("--n_heads",       type=int,   default=8,                           help="Number of attention heads.")
    parser.add_argument("--n_blocks",      type=int,   default=6,                           help="Number of encoder/decoder blocks.")
    parser.add_argument("--d_ffn",         type=int,   default=2048,                        help="Dimensionality of the feedforward network.")
    parser.add_argument("--dropout",       type=float, default=0.1,                         help="Dropout rate for the model.")
    parser.add_argument("--seq_len",       type=int,   default=-1,                          help="Sequence length for the model.")
    parser.add_argument("--batch_size",    type=int,   default=64,                          help="Batch size for training.")
    parser.add_argument("--lr",            type=float, default=1e-4,                        help="Learning rate for training.")
    parser.add_argument("--epochs",        type=int,   default=10,                          help="Number of epochs for training.")
    parser.add_argument("--device",        type=str,   default="cuda",                      help="Device to run the model on.")
    parser.add_argument("--seed",          type=int,   default=42,                          help="Seed for reproducibility.")

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """ Main function to train the Transformer model. """

    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    en_train = CorpusDataset(args.train_en_path, seq_len=args.seq_len)
    fr_train = CorpusDataset(args.train_fr_path, seq_len=args.seq_len)

    train_dataset = TranslationDataset(en_train, fr_train)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    en_dev = CorpusDataset(args.dev_en_path, seq_len=args.seq_len, word2idx=en_train.word2idx, idx2word=en_train.idx2word)
    fr_dev = CorpusDataset(args.dev_fr_path, seq_len=args.seq_len, word2idx=fr_train.word2idx, idx2word=fr_train.idx2word)

    dev_dataset = TranslationDataset(en_dev, fr_dev)
    dev_loader  = DataLoader(dev_dataset, batch_size=args.batch_size)

    enc_seq_len = args.seq_len + 2 if args.seq_len != -1 else en_train.seq_len + 2
    dec_seq_len = args.seq_len + 1 if args.seq_len != -1 else fr_train.seq_len + 1

    model = Transformer(
        en_train.vocab_size,
        fr_train.vocab_size,
        enc_seq_len,
        dec_seq_len,
        args.d_model,
        args.n_heads,
        args.n_blocks,
        args.d_ffn,
        args.dropout
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=fr_train.pad_idx)

    all_train_losses = []
    all_dev_losses   = []

    for epoch in range(args.epochs):
        train_losses = train(model, optimizer, criterion, train_loader, args.device, epoch, args.epochs)
        all_train_losses.extend(train_losses)

        dev_losses = validate(model, criterion, dev_loader, args.device, epoch, args.epochs)
        all_dev_losses.extend(dev_losses)

        print()

    torch.save({
        "args":                 args,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "word2idx_en":          en_train.word2idx,
        "word2idx_fr":          fr_train.word2idx,
        "idx2word_en":          en_train.idx2word,
        "idx2word_fr":          fr_train.idx2word,
        "train_losses":         all_train_losses,
        "dev_losses":           all_dev_losses
    }, args.model_path)

if __name__ == "__main__":
    train_args = parse_args()
    main(train_args)
