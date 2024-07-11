import os
import re
import collections
import json

def get_stats(text):
    counts = {}
    charset = sorted(set(text))
    for pair in zip(text, text[:1]):
        counts[pair] = counts.get(pair, 0) + 1
    return charset, counts

def merge(ids, pair, idx):
    """
    text is a list of token ids
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class Tokenizer():
    def __init__(self):
        self.merges = []
        self.vocab = {}
        self.vocab_size = 0

    def train(self, text, vocab_size):
        # assert (merges := vocab_size - 256) >= 0
        charset, self.stats = get_stats(text)
        self.vocab = self._build_vocab(charset, self.stats)
        self.vocab_size = len(self.vocab)
        merges = vocab_size - self.vocab_size

        for i in range(merges):
            print(i)
            best = max(self.stats, key=self.stats.get)
            self.merges.append(''.join(best))
            self._update_vocab(best, i + self.vocab_size)
            self.stats = self._update_stats(text)            

    def encode(self, text):
        ids = list(text)
        for i, merge in enumerate(self.merges):
            ids = merge(ids, tuple(merge), len(self.vocab) + i)
        return [self.vocab.index(token) for token in ids]

    def decode(self, ids):
        return ''.join(self.vocab[id] for id in ids)

    def _build_vocab(self, charset, stats):
        vocab = {}
        for i, char in enumerate(charset):
            vocab[i] = char

        for i, pair in enumerate(stats.keys(), start=len(charset)):
            vocab[i] = ''.join(pair)

        return vocab

    def _update_vocab(self, pair, idx):
        self.vocab[idx] = ''.join(pair)

    def _update_stats(self, text):
        stats = {}
        if self.vocab is None:
            self.vocab = {}
        i = 0
        while i < len(text) - 1:
            count = 1
            pair = (text[i], text[i+1])
            while ''.join(pair) in self.merges and i + count < len(text):
                new_char = ''.join(pair)
                if i + count + 1 < len(text):
                    pair = (new_char, text[i+count+1])
                else:
                    break
                count += 1

            stats[pair] = stats.get(pair, 0) + 1
            i += count

        return stats

tokenizer = Tokenizer()

with open('beemovie.txt', 'r') as f:
    text = f.read()

tokenizer.train(text, 1000)

print(tokenizer.vocab)


