import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)

    for word, freq in vocab.items():
        symbols = re.findall(r'</w>|.', word)

        for i in range(len(symbols)-1):
            if symbols[i] + symbols[i+1] not in vocab:
                pairs[symbols[i] + symbols[i+1]] += freq

    return pairs

def merge(pair, v_in):
    v_out = {}
    p = re.compile(r'(?<!\S)' + pair + r'(?!\S)')
    count = 0

    for word in v_in:
        w_out = p.sub(pair, word)
        v_out[w_out] = v_in[word]
        count += w_out.count(pair) * v_in[word]

    v_out[pair] = count

    return v_out

def get_stats_list(vocab_list):
    pairs = collections.defaultdict(int)

    for word in vocab_list:
        symbols = re.findall(r'</w>|.', word)

        for i in range(len(symbols)-1):
            pair = symbols[i] + symbols[i+1]
            if pair not in vocab_list:
                pairs[pair] += 1

    return pairs

