!pip install d2l==0.17.5
import os
import torch
import numpy as np
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
os.chdir("./STA221")
os.chdir("/content/drive/MyDrive/Colab Notebooks/STA221")
from DataUtils import load_data_nmt, tokenize_nmt, truncate_pad, build_array_nmt
from TransformerLayers import TransformerEncoder, TransformerDecoder, EncoderDecoder, BERTEncoderDecoder, train_seq2seq, train_seq2seq_BERT
from predict import bleu, predict_seq2seq, predict_seq2seq_BERT
from BERTmodel import train_bert, get_bert_encoding
torch.manual_seed(321)

num_examples, num_hiddens, num_layers, dropout, batch_size, num_steps = 5000, 64, 2, 0.1, 64, 10
lr, num_epochs, device = 5e-3, 500, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 64, 64, 4
key_size, query_size, value_size = 64, 64, 64
norm_shape = [64]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps, num_examples = num_examples)
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net_naive = EncoderDecoder(encoder, decoder)
train_seq2seq(net_naive, train_iter, lr, num_epochs, tgt_vocab, device)

source, target = tokenize_nmt(num_examples = num_examples)
engs = [" ".join(src) for src in source]
fras = [" ".join(tar) for tar in target]
bleuScore = 0
count = 0
for eng, fra in zip(engs, fras):
    translation, _ = predict_seq2seq(net_naive, eng, src_vocab, tgt_vocab, num_steps, device, True)
    bleuScore += bleu(translation, fra, k=2)
    if(count % 500 == 0):
        print(f'{eng} => {translation}, ', f'bleu {bleu(translation, fra, k=2):.3f}')
        print(str(count) + ". total is " + str(num_examples))
    count += 1
bleuScore /= num_examples
print(bleuScore)

### BERT
batch_size, max_len = 512, 64
pretrain_iter, pre_vocab = d2l.load_data_wiki(batch_size, max_len)

BertEncoder = d2l.BERTModel(len(pre_vocab), num_hiddens=64, norm_shape=[64],
                            ffn_num_input=64, ffn_num_hiddens=256, num_heads=2,
                            num_layers=2, dropout=0.2, key_size=64, query_size=64,
                            value_size=64, hid_in_features=64, mlm_in_features=64,
                            nsp_in_features=64)

devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()

train_bert(pretrain_iter, BertEncoder, loss, len(pre_vocab), devices, num_epochs)

## Use the dictionary from wiki to rebuild the source language
src_padded = [truncate_pad(src, num_steps, '<pad>') for src in source]
src_token_ids = torch.tensor(pre_vocab[src_padded], device=devices[0]).unsqueeze(0)
src_segments = src_token_ids * 0
src_valid_len = torch.tensor([num_steps] * (num_examples + 1), device=devices[0]).unsqueeze(0)

## build the target language using the original dictionary
tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens = ['<pad>', '<bos>', '<eos>'])
tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
tgt_array = tgt_array.to(device)
tgt_valid_len = tgt_valid_len.to(device)

decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)

net_BERT = BERTEncoderDecoder(BertEncoder, decoder)

lr, batch_size = 5e-3, 64
train_seq2seq_BERT(net_BERT, src_token_ids, src_segments, src_valid_len, 
                   tgt_array, tgt_valid_len, lr, num_epochs, tgt_vocab, device, batch_size)
                   
source, target = tokenize_nmt(num_examples = num_examples)
engs = [" ".join(src) for src in source]
fras = [" ".join(tar) for tar in target]
bleuScore = 0
count = 0
for i in range(len(source)):
    src_token_id = src_token_ids[0, i, :].reshape(1, num_steps)
    src_segment = src_segments[0, i, :].reshape(1, num_steps)
    src_valid = src_valid_len[0, i].reshape(1)
    translation, _ = predict_seq2seq_BERT(net_BERT, src_token_id, src_segment, src_valid, tgt_vocab, num_steps, device, True)
    bleuScore += bleu(translation, fras[i], k=2)
    if(count % 500 == 0):
        print(f'{engs[i]} => {translation}, ', f'bleu {bleu(translation, fras[i], k=2):.3f}')
        print(str(count) + ". total is " + str(num_examples))    
    count += 1
bleuScore /= num_examples
print(bleuScore)





