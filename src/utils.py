""" Dataloader classes and other utility classes needed by the Transformer architecture. """

import re
from typing import Optional, List

import torch
from torch import nn
from torch.utils.data import Dataset

from tqdm import tqdm
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

class CorpusDataset(Dataset):
    """ Dataset class for the corpus. """

    def __init__(
            self,
            corpus_file:    str,
            seq_len:        Optional[int] = -1,
            process_corpus: bool = False,
            save_path:      Optional[str]  = None,
            word2idx:       Optional[dict] = None,
            idx2word:       Optional[dict] = None
        ) -> None:
        """ Initialize the dataset.

        Args:
            corpus_file:    Path to the corpus file.
            seq_len:        Sequence length for the model.
            process_corpus: Whether to process the corpus or not.
            save_path:      Path to save the processed corpus.
            word2idx:       Dictionary mapping words to indices.
            idx2word:       Dictionary mapping indices to words.
        """
        self.corpus_file = corpus_file
        self.corpus      = self.__load_corpus(process_corpus, save_path)

        if seq_len != -1:
            self.seq_len = seq_len
        else:
            self.seq_len = max(len(line.split()) for line in self.corpus)

        if word2idx is None or idx2word is None:
            self.__build_vocab()

        else:
            self.word2idx = word2idx
            self.idx2word = idx2word

    def __load_corpus(
            self,
            process_corpus: bool,
            save_path:      Optional[str]
        ) -> List[str]:
        """ Load the corpus from the corpus file.

        Args:
            process_corpus: Whether to process the corpus or not.

        Returns:
            List of strings containing the corpus.
        """
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.readlines()

        if process_corpus:
            corpus = self.__process_corpus(corpus)

            if save_path is not None:
                with open(save_path, 'w', encoding='utf-8') as f:
                    for line in corpus:
                        f.write(line + '\n')

        return corpus

    def __process_corpus(self, corpus: List[str]) -> List[str]:
        """ Process the corpus to remove HTML tags and other unwanted characters.

        Args:
            corpus: List of strings containing the corpus.

        Returns:
            List of strings containing the processed corpus.
        """
        processed_corpus = []

        for line in tqdm(corpus, desc='Processing corpus'):
            if line.startswith('<title>') or line.startswith('<description>'):
                line = re.sub(r'<.*?>', '', line)

            elif line.startswith('<'):
                continue

            line = line.replace('--', '')
            line = re.sub(r'[^\w\s]', '', line, flags=re.UNICODE)
            line = line.strip()

            if line:
                processed_corpus.append(line)

        return processed_corpus

    def __build_vocab(self) -> None:
        """ Build the vocabulary for the corpus. """
        vocab = set()
        for line in self.corpus:
            vocab.update(line.split())

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        special_tokens = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']

        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token

    @property
    def vocab_size(self) -> int:
        """ Return the size of the vocabulary. """
        return len(self.word2idx)

    @property
    def start_idx(self) -> int:
        """ Return the index of the start token. """
        return self.word2idx['<BOS>']

    @property
    def end_idx(self) -> int:
        """ Return the index of the end token. """
        return self.word2idx['<EOS>']

    @property
    def pad_idx(self) -> int:
        """ Return the index of the padding token. """
        return self.word2idx['<PAD>']

    @property
    def unk_idx(self) -> int:
        """ Return the index of the unknown token. """
        return self.word2idx['<UNK>']

    def __len__(self) -> int:
        """ Return the length of the dataset. """
        return len(self.corpus)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Get an item from the dataset.

        Args:
            idx: Index of the sentence in the corpus.

        Returns:
            Tensor containing the indices of the words in the sentence from the vocabulary
            and the original length of the sentence.
        """
        line = self.corpus[idx].split()[:self.seq_len]

        line     = ['<BOS>'] + line + ['<EOS>']
        orig_len = len(line)

        line += ['<PAD>'] * (self.seq_len + 2 - orig_len)

        line = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in line]

        return torch.tensor(line), orig_len

class TranslationDataset(Dataset):
    """ Dataset class for the translation task. """

    def __init__(
            self,
            en_corpus: CorpusDataset,
            fr_corpus: CorpusDataset
        ) -> None:
        """ Initialize the dataset.

        Args:
            en_corpus: English corpus dataset.
            fr_corpus: French corpus dataset.
        """
        self.en_corpus = en_corpus
        self.fr_corpus = fr_corpus

    def __len__(self) -> int:
        """ Return the length of the dataset. """
        return len(self.en_corpus)

    def __getitem__(self, idx: int) -> tuple:
        """ Get an item from the dataset.

        Args:
            idx: Index of the sentence in the corpus.

        Returns:
            Tensors containing the indices of the words in the English and French sentences
            from the vocabulary and the original lengths of the sentences.
        """
        en_line, en_len = self.en_corpus[idx]
        fr_line, fr_len = self.fr_corpus[idx]

        return (en_line, en_len), (fr_line, fr_len)

class PositionalEncoding(nn.Module):
    """ Positional encoding class for the Transformer model. """

    def __init__(self, d_model: int, seq_len: int) -> None:
        """ Initialize the positional encoding.

        Args:
            d_model: Dimensionality of the model.
            seq_len: Sequence length for the model.
        """
        super().__init__()

        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))

        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(pos.float() * div)
        pos_enc[:, 1::2] = torch.cos(pos.float() * div)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the positional encoding.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pos_enc[:x.size(1), :].unsqueeze(0).to(x.device)

class MultiheadAttention(nn.Module):
    """ Multihead attention layer for the Transformer model. """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
        ) -> None:
        """ Initialize the multihead attention layer.

        Args:
            d_model: Dimensionality of the model.
            n_heads: Number of attention heads.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def __scaled_dot_product_attention(
            self,
            query: torch.Tensor,
            key:   torch.Tensor,
            value: torch.Tensor,
            mask:  Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """ Scaled dot product attention.

        Args:
            query: Input query tensor of shape (batch_size, n_heads, seq_len, head_dim).
            key:   Input key tensor of shape (batch_size, n_heads, seq_len, head_dim).
            value: Input value tensor of shape (batch_size, n_heads, seq_len, head_dim).
            mask:  Optional mask tensor of shape (batch_size, seq_len_q, seq_len_k).

        Returns:
            Output tensor of shape (batch_size, n_heads, seq_len, head_dim).
        """
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask   = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        softmax_scores = torch.softmax(scores, dim=-1)

        attention = torch.matmul(softmax_scores, value)

        return attention

    def forward(
            self,
            query: torch.Tensor,
            key:   torch.Tensor,
            value: torch.Tensor,
            mask:  Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """ Forward pass of the multihead attention layer.

        Args:
            query: Input query tensor of shape (batch_size, seq_len, d_model).
            key:   Input key tensor of shape (batch_size, seq_len, d_model).
            value: Input value tensor of shape (batch_size, seq_len, d_model).
            mask:  Optional mask tensor of shape (batch_size, seq_len_q, seq_len_k).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size = query.shape[0]

        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        x = self.__scaled_dot_product_attention(q, k, v, mask)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.out(x)

def get_padding_mask(
        orig_lens_k: torch.Tensor,
        seq_len_k:   int,
        seq_len_q:   Optional[int] = None
    ) -> torch.Tensor:
    """ Create a padding mask.

    Args:
        orig_lens_k: Original lengths of the keys in the batch.
        seq_len_q:   Sequence length for the queries.
        seq_len_k:   Sequence length for the keys.

    Returns:
        Boolean padding mask tensor of shape (batch_size, seq_len_q, seq_len_k).
    """
    if seq_len_q is None:
        seq_len_q = seq_len_k

    mask = torch.arange(seq_len_k).unsqueeze(0) >= orig_lens_k.unsqueeze(1)
    mask = mask.unsqueeze(1).repeat(1, seq_len_q, 1)

    return mask.bool()

def get_causal_mask(seq_len: int) -> torch.Tensor:
    """ Create a causal mask.

    Args:
        seq_len: Sequence length for the model.

    Returns:
        Boolean causal mask tensor of shape (seq_len, seq_len).
    """
    mask = torch.ones(seq_len, seq_len)
    mask = torch.triu(mask, diagonal=1)

    return mask.bool()

def get_encoder_mask(orig_lens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """ Create an encoder mask.

    Args:
        orig_lens: Original lengths of the keys in the batch.
        seq_len:   Sequence length for the model.

    Returns:
        Boolean encoder mask tensor of shape (batch_size, seq_len, seq_len).
    """
    return get_padding_mask(orig_lens, seq_len)

def get_decoder_mask(orig_lens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """ Create a decoder mask.

    Args:
        orig_lens: Original lengths of the keys in the batch.
        seq_len:   Sequence length for the model.

    Returns:
        Boolean decoder mask tensor of shape (batch_size, seq_len, seq_len).
    """
    pad_mask    = get_padding_mask(orig_lens, seq_len)
    causal_mask = get_causal_mask(seq_len)

    return pad_mask | causal_mask

if __name__ == "__main__":
    corpus_names = [
        'train.en',
        'dev.en',
        'test.en',
        'train.fr',
        'dev.fr',
        'test.fr'
    ]

    for corpus_name in corpus_names:
        corpus_file_path = f'corpus/{corpus_name}'
        corpus_save_path = f'corpus/processed_{corpus_name}'

        CorpusDataset(
            corpus_file_path,
            seq_len=None,
            process_corpus=True,
            save_path=corpus_save_path
        )

        print(f'Processed {corpus_name} dataset.')
        print()

def get_sentence_from_idx(
        idx:     torch.Tensor,
        idx2word: dict
    ) -> str:
    """ Get the sentence from the indices.

    Args:
        idx:      Tensor containing the indices of the sentence.
        idx2word: Dictionary mapping indices to words.

    Returns:
        Sentence formed from the indices.
    """
    words = []

    for i in idx:
        word = idx2word[i.item()]
        if word == '<EOS>':
            break
        words.append(word)

    return ' '.join(words)

def get_sentence_rouge_score(
        pred_idx:   torch.Tensor,
        target_idx: torch.Tensor,
        idx2word:   dict
    ) -> float:
    """ Calculate the ROUGE score for a sentence.

    Args:
        pred_idx:   Tensor containing the indices of the predicted sentence.
        target_idx: Tensor containing the indices of the target sentence.
        idx2word:   Dictionary mapping indices to words.

    Returns:
        ROUGE score for the sentence.
    """
    pred_sentence   = get_sentence_from_idx(pred_idx, idx2word)
    target_sentence = get_sentence_from_idx(target_idx, idx2word)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(target_sentence, pred_sentence)

    return scores['rougeL'].fmeasure

def get_sentence_bleu_score(
        pred_idx:   torch.Tensor,
        target_idx: torch.Tensor,
        idx2word:   dict
    ) -> float:
    """ Calculate the BLEU score for a sentence.

    Args:
        pred_idx:   Tensor containing the indices of the predicted sentence.
        target_idx: Tensor containing the indices of the target sentence.
        idx2word:   Dictionary mapping indices to words.

    Returns:
        BLEU score for the sentence.
    """
    pred_sentence   = get_sentence_from_idx(pred_idx, idx2word)
    target_sentence = get_sentence_from_idx(target_idx, idx2word)

    return corpus_bleu([pred_sentence], [[target_sentence]]).score
