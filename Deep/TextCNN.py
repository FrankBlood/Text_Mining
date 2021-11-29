#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
TextCNN
======
A class for something.
@author: Guoxiu He
@contact: gxhe@fem.ecnu.edu.cn
@site: https://scholar.google.com/citations?user=2NVhxpAAAAAJ
@time: 14:16, 2021/11/28
@copyright: "Copyright (c) 2021 Guoxiu He. All Rights Reserved"
"""

import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.metrics import cal_all
from Deep.Base_Model import Base_Model


class TextCNN(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_filters, filter_sizes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(TextCNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        self.metrics_num = 4
        if 'metrics_num' in kwargs:
            self.metrics_num = kwargs['metrics_num']

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.num_filters * len(self.filter_sizes), self.hidden_dim)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        embed = embed.unsqueeze(1)
        cnn_out = torch.cat([self.conv_and_pool(embed, conv) for conv in self.convs], 1)
        cnn_out = self.dropout(cnn_out)
        hidden = self.fc1(cnn_out)
        out = self.fc_out(hidden)
        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("What the F**K! There is no {} function.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
