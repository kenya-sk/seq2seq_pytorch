import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
from torch.autograd import Variable

from model import Encoder, Decoder, Seq2Seq


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx.keys():
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        else:
            return self.word2idx[word]
        
    def __len__(self):
        return self.idx


def tensor2numpy(x):
    return x.data.cpu().numpy()


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def caption_tensor(caption, vocab):
    target = []
    target.append(vocab("<start>"))
    target.extend([vocab(char) for char in caption])
    target.append(vocab("<end>"))
    target = torch.Tensor(target)

    return target


def load_model(model_path, params):
    # load encoder model
    encoder = Encoder(params["decoder_dim"], params["embed_dim"],
                params["hidden_dim"], n_layers=1, dropout=0.0
    )

    # load decoder model
    decoder = Decoder(params["embed_dim"], params["hidden_dim"],
                params["encoder_dim"], n_layers=1, dropout=0.0
    )

    seq2seq_model = Seq2Seq(encoder, decoder)
    seq2seq_model.load_dict(torch.load(model_path))

    # if only use CPU
    # seq2seq_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    print("DONE: loaded model")

    return seq2seq_model