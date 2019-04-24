import numpy as np
import pickle
from janome.tokenizer import Tokenizer
import torch
from torch.autograd import Variable

from model import Encoder, Decoder, Seq2Seq
from sent_utils import Vocabulary, caption_tensor


def padding_text(sources):
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    src_len = [len(src) for src in sources]

    conv_src = torch.zeros(len(sources), max(src_len)).long()
    for i, cap in enumerate(sources):
        end = src_len[i]
        conv_src[i, :end] = torch.Tensor(cap[:end]).long() 

    return conv_src


def predict(text_lst, en_vocab, de_vocab, model, tokenizer, thresh=-0.5):
    model.eval()

    id_lst = []
    for i in range(len(text_lst)):
        id_lst.append(caption_tensor(text_lst[i], en_vocab, tokenizer, reverse=False))

    conv_text = padding_text(id_lst)
    output = model(src=conv_text, trg=conv_text, teacher_forcing_ratio=0.0)

    pred_lst = []
    for batch in range(output.shape[0]):
        conv_out_lst = []
        text_word = []
        for token in tokenizer.tokenize(text_lst[batch][1:-1]):
            text_word.append(token.surface)
        for idx in range(output.shape[1]):
            max_idx = int(np.argmax(output[batch][idx].detach().numpy()))
            # max prob range is -inf to 0
            max_prob = output[batch][idx][max_idx].detach().numpy()
            de_word = de_vocab.idx2word[max_idx]
            if (de_word == "<unk>") or (max_prob < thresh):
                # replace source word
                conv_out_lst.append(text_lst[batch][idx])
            else:
                conv_out_lst.append(de_vocab.idx2word[max_idx])
        pred_lst.append("".join(conv_out_lst))

    return pred_lst


if __name__ == "__main__":
    # load encoder vocaluraly
    en_vocab_path = "../data/vocab/format_en_word_vocab.pth"
    with open(en_vocab_path, "rb") as f:
        en_vocab = pickle.load(f)
    # load decoder vocaburaly
    de_vocab_path = "../data/vocab/format_de_word_vocab.pth"
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
    model_path = "../data/model/format_best_model.pth"
    seq2seq = Seq2Seq(encoder, decoder)
    seq2seq.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # define tokenizer
    tokenizer = Tokenizer()

    # source text
    src_text = ["私は上から来ます！気をつけて！！", "\"道路の脇に二羽の小鳥。\""]

    pred_lst = predict(src_text, en_vocab, de_vocab, seq2seq, tokenizer, thresh=-0.5)
    print(pred_lst)