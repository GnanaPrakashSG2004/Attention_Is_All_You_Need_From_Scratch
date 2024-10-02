""" Testing pipeline for the English to French machine translation task using the Transformer model. """

from typing import List, Optional, Tuple

import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train import Transformer
from src.utils import (
    CorpusDataset, TranslationDataset,
    get_padding_mask, get_encoder_mask, get_decoder_mask,
    get_sentence_bleu_score
)

def test(
        model:        Transformer,
        criterion:    nn.CrossEntropyLoss,
        loader:       DataLoader,
        device:       torch.device,
        corpus_path:  str,
        metrics_path: Optional[str] = None
    ) -> Tuple[List[float], List[float]]:
    """ Evaluate the model on the test set.

    Args:
        model:        Transformer model.
        criterion:    Loss function.
        loader:       DataLoader for the test set.
        device:       Device to run the model on.
        corpus_path:  Path to the corpus file.
        metrics_path: Path to save the BLEU and ROUGE scores.

    Returns:
        List of BLEU and ROUGE scores.
    """
    model.eval()
    pbar = tqdm(loader, desc='Testing')

    bleu_scores = []

    fr_word2idx = loader.dataset.fr_corpus.word2idx
    fr_idx2word = loader.dataset.fr_corpus.idx2word

    with open(corpus_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    with torch.no_grad():
        for _, (en_batch, fr_batch) in enumerate(pbar):
            en_seqs, en_lens = en_batch

            enc_inp  = en_seqs.to(device)
            enc_mask = get_encoder_mask(en_lens, en_seqs.size(1)).to(device)

            enc_out  = model.encoder(enc_inp, enc_mask)

            dec_idx = [fr_word2idx['<BOS>']]

            fr_seqs, fr_lens = fr_batch
            fr_seqs = fr_seqs.to(device)

            for _ in range(fr_seqs.size(1)):
                dec_inp        = torch.tensor(dec_idx).unsqueeze(0).to(device)
                dec_inp_len    = torch.tensor(dec_inp.size(1)).unsqueeze(0)
                dec_self_mask  = get_decoder_mask(dec_inp_len, fr_seqs.size(1) - 1).to(device)
                dec_cross_mask = get_padding_mask(en_lens, en_seqs.size(1), dec_inp.size(1)).to(device)

                dec_out = model.decoder(dec_inp, enc_out, dec_self_mask, dec_cross_mask)
                dec_out = model.out(dec_out)

                next_word = dec_out[:, -1, :].argmax().item()

                if next_word == fr_word2idx['<EOS>']:
                    break

                dec_idx.append(next_word)

            sent_bleu_score = get_sentence_bleu_score(torch.tensor(dec_idx).unsqueeze(0), fr_seqs[:, 1:], fr_idx2word)

            bleu_scores.append(sent_bleu_score)

    if metrics_path:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                f.write(f'{sentence.strip()} BLEU: {bleu_scores[i]:.4f}\n')

    return bleu_scores

def parse_args() -> argparse.Namespace:
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Test a pre-trained Transformer model for machine translation.")

    parser.add_argument("--test_en_path",  type=str,   default="corpus/processed_test.en", help="Path to the English test dataset.")
    parser.add_argument("--test_fr_path",  type=str,   default="corpus/processed_test.fr", help="Path to the French test dataset.")
    parser.add_argument("--model_path",    type=str,   default="weights/transformer.pt",   help="Path to the save pre-trained model.")
    parser.add_argument("--device",        type=str,   default="cuda:1",                   help="Device to run the model on.")
    parser.add_argument("--seed",          type=int,   default=42,                         help="Seed for reproducibility.")

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """ Main function to train the Transformer model. """

    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    params = torch.load(args.model_path)

    saved_args  = params['args']
    en_seq_len  = params['en_seq_len']
    fr_seq_len  = params['fr_seq_len']
    word2idx_en = params['word2idx_en']
    word2idx_fr = params['word2idx_fr']
    idx2word_en = params['idx2word_en']
    idx2word_fr = params['idx2word_fr']

    model = Transformer(
        len(word2idx_en),
        len(word2idx_fr),
        en_seq_len + 2,
        fr_seq_len + 1,
        saved_args.d_model,
        saved_args.n_heads,
        saved_args.n_blocks,
        saved_args.d_ffn,
        saved_args.dropout
    ).to(args.device)

    model.load_state_dict(params['model_state_dict'])

    en_test = CorpusDataset(args.test_en_path, seq_len=en_seq_len, word2idx=word2idx_en, idx2word=idx2word_en)
    fr_test = CorpusDataset(args.test_fr_path, seq_len=fr_seq_len, word2idx=word2idx_fr, idx2word=idx2word_fr)

    test_dataset = TranslationDataset(en_test, fr_test)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx_fr['<PAD>'])

    test(model, criterion, test_loader, args.device, args.test_fr_path, metrics_path='metrics/testbleu.txt')

if __name__ == "__main__":
    test_args = parse_args()
    main(test_args)
