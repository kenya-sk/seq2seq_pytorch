import numpy as np
import pickle
import torch
from torch.autograd import Variable

from model import Encoder, Decoder, Seq2Seq


def conv_text2id(text, vocab):
    id_lst = [vocab.word2idx["<start>"]]
    for ch in str(text):
        id_lst.append(vocab.word2idx[ch])

    id_lst.append(vocab.word2idx["<end>"])
    return id_lst


def prediction(src_text, en_vocab, de_vocab, model):
    model.eval()

    id_lst = conv_text2id(src_text, en_vocab)
    conv_text = Variable(torch.Tensor([id_lst])).long()
    output = model(src=conv_text, trg=conv_text, teacher_forcing_ratio=0.0)[0]

    pred_lst = []
    for i in range(output.shape[0]):
        max_idx = int(np.argmax(output[i]).data.cpu())
        pred_lst.append(de_vocab.idx2word[max_idx])

    return pred_lst
        

if __name__ == "__main__":
    # load encoder vocaluraly
    en_vocab_path = "../data/vocab/en_vocab.pth"
    with open(en_vocab_path, "rb") as f:
        en_vocab = pickle.load(f)
    # load decoder vocaburaly
    de_vocab_path = "../data/vocab/de_vocab.pth"
    with open(de_vocab_path, "rb") as f:
        de_vocab = pickle.load(f)

    # load model
    en_size = len(en_vocab)
    de_size = len(de_vocab)
    embed_dim = 256
    hidden_dim = 512
    en_n_layers = 2 
    en_dropout = 0.0
    de_n_layers = 1
    de_dropout = 0.0

    encoder = Encoder(en_size, embed_dim, hidden_dim,
                    n_layers=en_n_layers, dropout=en_dropout)
    decoder = Decoder(embed_dim, hidden_dim, de_size,
                    n_layers=de_n_layers, dropout=de_dropout)
    model_path = "../data/model/best_model.pth"
    seq2seq = Seq2Seq(encoder, decoder)
    seq2seq.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # source text
    src_text = "森のぼかしの使い方と揺れ方"

    pred_lst = prediction(src_text, en_vocab, de_vocab, seq2seq)
    print(pred_lst)