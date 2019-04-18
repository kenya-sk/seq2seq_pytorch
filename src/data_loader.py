import pandas as pd
import numpy as np
import os
import torch
import torch.utils.data as data

from utils import caption_tensor


class SeqDataset(data.Dataset):
    def __init__(self, file_path, en_vocab, de_vocab):
        self._file_path = file_path
        self._data_df = pd.read_csv(file_path)
        self._en_vocab = en_vocab
        self._de_vocab = de_vocab

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, idx):
        src_cap = caption_tensor(self._data_df["src"][idx], self._en_vocab)
        trg_cap = caption_tensor(self._data_df["trg"][idx], self._de_vocab)

        return src_cap, trg_cap


def collate_fn(data):
    # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: len(x[1]), reverse=True)

    sources, targets = zip(*data)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    src_len = [len(src) for src in sources]
    conv_src = torch.zeros(len(sources), max(src_len)).long()
    for i, cap in enumerate(sources):
        end = src_len[i]
        conv_src[i, :end] = cap[:end] 
        
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    trg_len = [len(trg) for trg in targets]
    conv_trg = torch.zeros(len(targets), max(trg_len)).long()
    for i, cap in enumerate(targets):
        end = trg_len[i]
        conv_trg[i, :end] = cap[:end] 
        
    return conv_src, conv_trg

def get_dataset(file_path, en_vocab, de_vocab, batch_size, shuffle, num_workers):
    dataset = SeqDataset(file_path, en_vocab, de_vocab)
    data_loader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size,
                shuffle=shuffle, num_workers=num_workers,
                collate_fn=collate_fn)

    return data_loader


        