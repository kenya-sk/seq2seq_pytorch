import os
import math
import logging
import pickle
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import Vocabulary
from data_loader import get_dataset

logger = logging.getLogger(__name__)

def evaluate(model, val_iter, vocab_size, en_vocab, de_vocab):
    model.eval()
    pad = en_vocab.word2idx['<pad>']
    total_loss = 0
    for src, trg in val_iter:
        src = src.data.cuda()
        trg = trg.data.cuda()
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.item()
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, en_vocab, de_vocab):
    model.train()
    total_loss = 0
    pad = en_vocab.word2idx['<pad>']
    for src, trg in train_iter:
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    logger.debug("TRAIN LOSS [{0}]: {1}".format(e, total_loss))


def main():
    # hyper parameter
    epochs = 100
    batch_size = 32
    lr = 0.0001
    grad_clip = 10.0
    hidden_size = 512
    embed_size = 256
    shuffle = True
    num_workers = 1

    # Is GPU usable?
    assert torch.cuda.is_available()

    # load vocab
    logger.debug("[!] loading vocabulary...")
    en_vocab_path = "../data/vocab/en_vocab.pth"
    with open(en_vocab_path, "rb") as f:
        en_vocab = pickle.load(f)
    logger.debug("encoder vocab size: {}".format(len(en_vocab)))
        
    de_vocab_path = "../data/vocab/de_vocab.pth"
    with open(de_vocab_path, "rb") as f:
        de_vocab = pickle.load(f)
    logger.debug("decoder vocab size: {}".format(len(de_vocab)))
    en_size, de_size = len(en_vocab), len(de_vocab)
    logger.debug("[formal_vocab]:%d [tweet_vocab]:%d" % (en_size, de_size))

    logger.debug("[!] preparing dataset...")
    train_iter = get_dataset("../data/datasets/train.csv", en_vocab, de_vocab, batch_size, shuffle, num_workers)
    val_iter = get_dataset("../data/datasets/val.csv", en_vocab, de_vocab, batch_size, shuffle, num_workers)

    logger.debug("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                    n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                    n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)
    logger.debug(seq2seq)

    best_val_loss = 1000000000
    for e in range(1, epochs+1):
        train(e, seq2seq, optimizer, train_iter,
            de_size, grad_clip, en_vocab, de_vocab)
        val_loss = evaluate(seq2seq, val_iter, en_size, en_vocab, de_vocab)
        logger.debug("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
        % (e, val_loss, math.exp(val_loss)))

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        logger.debug("save model (epoch: {0})".format(e))
        torch.save(seq2seq.state_dict(), '/data/sakka/seq2seq_model/best_model.pth')
        best_val_loss = val_loss

if __name__ == "__main__":
    logs_path = "../logs/train.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)