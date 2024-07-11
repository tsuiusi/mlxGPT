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
        encoded = []
        i = 0
        while i < len(text):
            longest_match = None
            longest_match_length = 0
            for token_id, token in self.vocab.items():
                if text.startswith(token, i):
                    if len(token) > longest_match_length:
                        longest_match = token_id
                        longest_match_length = len(token)
            
            if longest_match is not None:
                encoded.append(longest_match)
                i += longest_match_length
            else:
                # If no match found, you might want to handle unknown characters
                # For now, we'll just skip it
                i += 1
        
        return encoded

    def decode(self, ids):
        return ''.join(self.vocab[id] for id in ids)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.vocab = json.load(f) 
        print(f'Vocab size: {len(self.vocab)}') 

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.vocab, f)
        print(f'Saved as {filename + ".json"}')

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

with open('input.txt', 'r') as f:
    text = f.read()

# tokenizer.train(text, 1000)
# tokenizer.save('tokens.json')
tokenizer.load('tokens.json')

print(enc := tokenizer.encode('"Watashi Dake Yuurei" (ワタシダケユウレイ , lit. "I\'m the Only Ghost") is an insert song for the tenth episode of the anime series, Bocchi the Rock!. It is performed by SICK HACK and sung by Kikuri Hiroi. The song was released on April 26, 2023, as a part of a bonus CD in Blu-ray and DVD volume 5.[1]'))


print(tokenizer.decode(enc))
