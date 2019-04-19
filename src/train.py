import os
import math
import logging
import argparse
import pickle
import torch
import torch.nn as nn
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


def train(epoch, model, optimizer, train_iter, vocab_size, grad_clip, en_vocab, de_vocab):
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

    logger.debug("TRAIN LOSS: {0:.5f} (epoch={1})".format(total_loss/len(train_iter), epoch))


def main(args):
    # Is GPU usable?
    assert torch.cuda.is_available()

    # load encoder decoder vocab
    logger.debug("Loading vVcabulary...")
    # encoder vocaluraly
    with open(args.en_vocab_path, "rb") as f:
        en_vocab = pickle.load(f)
    logger.debug("Encoder vocab size: {}".format(len(en_vocab)))
    # decoder vocaburaly
    with open(args.de_vocab_path, "rb") as f:
        de_vocab = pickle.load(f)
    logger.debug("Decoder vocab size: {}".format(len(de_vocab)))
    en_size, de_size = len(en_vocab), len(de_vocab)
    logger.debug("[source_vocab]:%d [target_vocab]:%d" % (en_size, de_size))

    # setting train and val dataloader
    logger.debug("Preparing dataset...")
    train_iter = get_dataset(args.train_path, en_vocab, de_vocab, args.batch_size, args.shuffle, args.num_workers)
    val_iter = get_dataset(args.val_path, en_vocab, de_vocab, args.batch_size, args.shuffle, args.num_workers)

    # setting seq2seq model 
    logger.debug("Instantiating models...")
    encoder = Encoder(en_size, args.embed_size, args.hidden_size,
                    n_layers=2, dropout=0.5)
    decoder = Decoder(args.embed_size, args.hidden_size, de_size,
                    n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    logger.debug(seq2seq)

    # Training and validation model
    best_val_loss = None
    for epoch in range(1, args.epochs+1):
        train(epoch, seq2seq, optimizer, train_iter,
            de_size, args.grad_clip, en_vocab, de_vocab)
        val_loss = evaluate(seq2seq, val_iter, de_size, en_vocab, de_vocab)
        logger.debug("VAL LOSS: {0:.5f} (epoch={1})".format(val_loss, epoch))

    # Save the model if the validation loss is the best we've seen so far.
    if (best_val_loss is None) or (val_loss < best_val_loss):
        logger.debug("save model (epoch={0})".format(epoch))
        torch.save(seq2seq.state_dict(), args.save_model_path)
        best_val_loss = val_loss


def make_parse():
    parser = argparse.ArgumentParser(
        prog="train.py",
        usage="training seq2seq model ",
        description="description",
        epilog="end",
        add_help=True
    )

    # data Argument
    parser.add_argument("--train_path", type=str, default="/content/gdrive/My Drive/seq2seq_pytorch/data/datasets/train.csv")
    parser.add_argument("--val_path", type=str, default="/content/gdrive/My Drive/seq2seq_pytorch/data/datasets/val.csv")
    parser.add_argument("--en_vocab_path", type=str, default="/content/gdrive/My Drive/seq2seq_pytorch/data/vocab/en_vocab.pth")
    parser.add_argument("--de_vocab_path", type=str, default="/content/gdrive/My Drive/seq2seq_pytorch/data/vocab/de_vocab.pth")
    parser.add_argument("--save_model_path", type=str, default="/content/gdrive/My Drive/seq2seq_pytorch/data/model/best_model.pth")

    # params Argument
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    logs_path = "/content/gdrive/My Drive/seq2seq_pytorch/logs/train.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    
    args = make_parse
    logger.debug("Running with args: {0}".format(args))
    
    try:
        main(args)
    except KeyboardInterrupt as eroor:
        print("[STOP]", error)