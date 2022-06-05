import os
import torch
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = "./fra-eng"
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

# raw_text = read_data_nmt()
# print(raw_text[:75])

def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

# text = preprocess_nmt(raw_text)
# print(text[:80])

def tokenize_nmt(num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    text = preprocess_nmt(read_data_nmt())
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    # print("There are " + str(i) + " lines in the data.")
    return source, target

# source, target = tokenize_nmt(num_examples = 600)
# source[:6], target[:6]

def show_list_len_pair_hist(source, target):
    plt.clf()
    """Plot the histogram for list length pairs."""
    len_source = np.array([len(source[i]) for i in range(len(source))])
    len_target = np.array([len(target[i]) for i in range(len(target))])
    bins = np.linspace(0, 60, 20)
    plt.hist(len_source, bins, alpha = 0.5, label = 'source')
    plt.hist(len_target, bins, alpha = 0.5, label = 'target')
    plt.legend(loc = 'upper right')
    plt.show()
    
# show_list_len_pair_hist(source, target)
# src_vocab = d2l.Vocab(source, min_freq = 2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
# list(src_vocab.token_to_idx.items())[:10]

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

# truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
# list(src_vocab.token_to_idx.items())[47]
# list(src_vocab.token_to_idx.items())[4]
# list(src_vocab.token_to_idx.items())[1]

def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    source, target = tokenize_nmt(num_examples = num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

# train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
# for X, X_valid_len, Y, Y_valid_len in train_iter:
#     print('X:', X.type(torch.int32))
#     print('valid lengths for X:', X_valid_len)
#     print('Y:', Y.type(torch.int32))
#     print('valid lengths for Y:', Y_valid_len)
#     break
