#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2022.02.14
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : main.py

import torch
import torch.utils.data as Data
import pickle

residue2idx = pickle.load(open('./data/residue2idx.pkl', 'rb'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_token(sequences):
    token2index = residue2idx
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)
    token_index = list()
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_index.append(seq_id)
    return token_index


def pad_sequence(token_list):
    data = []
    for i in range(len(token_list)):
        n_pad = 29 - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])
    return data

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

def construct_dataset(seq_ids):
    if device == "cuda":
        seq_ids = torch.cuda.LongTensor(seq_ids)
    else:
        seq_ids = torch.LongTensor(seq_ids)
    data_loader = Data.DataLoader(MyDataSet(seq_ids),batch_size=128,shuffle=False,drop_last=False)

    return data_loader

def load_data(sequence_list):
    token_list = transform_token(sequence_list)
    data_train = pad_sequence(token_list)
    test_loader = construct_dataset(data_train)
    return test_loader

