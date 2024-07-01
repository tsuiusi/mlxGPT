import os
import re
import collections
import json

"""
word level tokenizer
bigrams count the number of occurrences of each word pair 

this one creates x new tokens as specified by the number of loops

the starting vocab is a set of all bigrams : their frequencies

so basically it should, in every iteration, find the most frequent tuple pair and add it as a single symbol, then take that symbol and count the tuples of combinations

input is a string? think about how the input will be handled if it's 5PB

there should also be a policy to how it encodes/decodes strings
"""

def get_vocab(file):
    vocab = collections.Counter()
    
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                if word:
                    # Split the word into a tuple of characters
                    char_tuple = tuple(word)
                    # Add '</w>' to the last character
                    char_tuple_with_end = char_tuple[:-1] + (char_tuple[-1] + '</w>',)
                    vocab[char_tuple_with_end] += 1

    return vocab 
    
def get_starting_stats(vocab):
    pairs = collections.defaultdict(int)

    for first, second in zip(vocab, vocab[1:] + '.'):
        pairs[first + ' ' + second] += 1
    
    return pairs

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[word[i], word[i+1]] += freq

    return pairs

def get_stats_list(vocab):
    counts = {}
    for pair in zip(vocab, vocab[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts

vocab = get_vocab('beemovie.txt')
print(f"Initial vocab size: {len(vocab)}")

with open('beemovie.txt', 'r') as f:
    tokens = f.read().encode('utf-8')
    tokens = list(map(int, tokens))
print(tokens)

stats = get_stats_list(tokens)
# print(sorted(((v, k) for k, v in stats.items()), reverse=True))
print(stats)

best = max(stats, key=stats.get)

print(best)

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

ids = list(tokens)
vocab_size = 276
num_merges = vocab_size - 256

merges = {}
for i in range(num_merges):
    stats = get_stats_list(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f'merging {pair} into {idx}')
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print(merges)
