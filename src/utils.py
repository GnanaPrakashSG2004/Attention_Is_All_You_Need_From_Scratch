""" Dataloader classes and other utility functions. """

import re
from typing import Optional, List

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

class CorpusDataset(Dataset):
    """ Dataset class for the corpus. """

    def __init__(
            self,
            corpus_file:    str,
            seq_len:        Optional[int],
            process_corpus: bool = False,
            save_path:      Optional[str]  = None,
            word2idx:       Optional[dict] = None,
            idx2word:       Optional[dict] = None
        ):
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

        if seq_len is not None:
            self.seq_len = seq_len
        else:
            self.seq_len = max(len(line.split()) for line in self.corpus)

        self.__build_vocab()
        self.word2idx = word2idx if word2idx is not None else self.word2idx
        self.idx2word = idx2word if idx2word is not None else self.idx2word

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
                    f.writelines(corpus)

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

    def __build_vocab(self):
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

    def __len__(self) -> int:
        """ Return the length of the dataset. """
        return len(self.corpus)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Get an item from the dataset.

        Args:
            idx: Index of the sentence in the corpus.

        Returns:
            Tensor containing the indices of the words in the sentence from the vocabulary.
        """
        line = self.corpus[idx].split()[:self.seq_len]

        line = ['<BOS>'] + line + ['<EOS>']
        line += ['<PAD>'] * (self.seq_len + 2 - len(line))

        line = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in line]

        return torch.tensor(line)
