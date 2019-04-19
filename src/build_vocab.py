import pandas as pd
import pickle
import logging
from collections import defaultdict

from utils import Vocabulary


logger = logging.getLogger("__name__")
log_path = "../logs/build_vocab.log"
logging.basicConfig(filename=log_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def build_vocab(word_lst, size=10000):
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    for word in word_lst[:size-4]:
        vocab.add_word(word)
        
    return vocab


def build(cap_lst, save_path, min_freq=2):
    # freq of each character
    char2cnt = defaultdict(int)
    for cap in cap_lst:
        try:
            for ch in cap:
                char2cnt[str(ch)] += 1
        except TypeError:
            pass

    # extract top freq character
    top_ch_lst = []
    for ch, cnt in sorted(char2cnt.items(), key=lambda x:-x[1]):
        if cnt < min_freq:
            break
        top_ch_lst.append(ch)

    # build vocabulary
    vocab = build_vocab(top_ch_lst)
    with open(save_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.debug("Number of vocabulary: {}".format(len(vocab)))
    logger.debug("Save the vocabulary wrapper to {}".format(save_path))


def main():
    data_df = pd.read_csv("../data/datasets/train.csv")
    # threshold of word frequency
    min_freq = 2

    # build encoder vocabulary
    en_cap_lst = list(data_df["src"])
    en_save_path = "../data/vocab/en_vocab.pth"
    build(en_cap_lst, en_save_path, min_freq)

    # build decoder vocabulary
    de_cap_lst = list(data_df["trg"])
    de_save_path = "../data/vocab/de_vocab.pth" 
    build(de_cap_lst, de_save_path, min_freq)


if __name__ == "__main__":
    main()


