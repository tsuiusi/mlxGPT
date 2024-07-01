import os
import numpy as np
import re, collections
import json

with open('input.txt', 'r') as f:
    text = f.read()

# All the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def get_stats(vocab):
    pairs = collections.defaultdict(int)

    for word, freq in vocab.items():
        symbols = re.findall(r'</w>|.', word)

        for i in range(len(symbols)-1):
            if symbols[i] + symbols[i+1] not in vocab:
                pairs[symbols[i] + symbols[i+1]] += freq

    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    p = re.compile(r'(?<!\S)' + pair + r'(?!\S)')
    count = 0

    for word in v_in:
        w_out = p.sub(pair, word)
        v_out[w_out] = v_in[word]
        count += w_out.count(pair) * v_in[word]

    v_out[pair] = count

    return v_out

vocab = {'low</w>': 5, 'lower</w>': 2, 'newest</w>': 6, 'widest</w>': 3} 

num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

print(vocab)

text = text.replace('.', '</w>')

textlist = text.split(' ')

def get_stats_list(vocab_list):
    pairs = collections.defaultdict(int)

    for word in vocab_list:
        symbols = re.findall(r'</w>|.', word)

        for i in range(len(symbols)-1):
            pair = symbols[i] + symbols[i+1]
            if pair not in vocab_list:
                pairs[pair] += 1

    return pairs

vocablist = ['alpha', 'beta', 'bebebebebebebeb', 'bebebebebe', 'bebebebebe', 'off']
vocab = {}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats_list(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

# with open('statslist.txt', 'w') as f:
 #    json.dump(stats, f)

print(vocab)



