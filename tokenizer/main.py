import os
import re
import collections
import json

def get_stats(text):
    counts = {}
    for pair in zip(text, text[:1]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
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
        self.merges = {}
        self.special_tokens = {}
        self.vocab = _build_vocab() 

    def train(self, text, vocab_size):
        vocab = get_stats(text)

        assert (merges := vocab_size - 256) >= 0

        for _ in range(merges):
            best = max(vocab, key=vocab.get)
            

    def encode(self, text):
        pass

    def decode(self, ids):
        pass

    def 
