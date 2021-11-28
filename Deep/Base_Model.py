#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Base_Model
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
import numpy as np
from utils.metrics import cal_all

class Base_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Base_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.criterion_name = criterion_name

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss
        }
        self.optimizer_dict = {
            'Adam': torch.optim.Adam
        }

        if criterion_name not in self.criterion_dict:
            raise ValueError("There is no criterion_name: {}.".format(criterion_name))
        self.criterion = self.criterion_dict[criterion_name]()

        if optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(optimizer_name))
        self.optimizer = self.optimizer_dict[optimizer_name](self.parameters(), lr=self.learning_rate)

        self.gpu = gpu

        self.device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        avg_embed = torch.mean(embed, dim=1)
        # out = self.softmax(self.fc(avg_embed))
        out = self.fc(avg_embed)
        return out

    def train_model(self, model, data_generator, input_path, output_path, word_dict):
        model.to(self.device)
        model.train()
        for epoch in range(self.num_epochs):
            total_y, total_pred_label = [], []
            for x, y in data_generator(input_path, output_path, word_dict, batch_size=self.batch_size):
                batch_x = torch.LongTensor(x).to(self.device)
                batch_y = torch.LongTensor(y).to(self.device)
                batch_pred_y = model(batch_x)
                model.zero_grad()
                loss = self.criterion(batch_pred_y, batch_y)
                loss.backward()
                self.optimizer.step()
                y_label = list(y)
                total_y += y_label
                pred_y_label = list(np.argmax(batch_pred_y.cpu().detach().numpy(), axis=-1))
                total_pred_label += pred_y_label
            metric_score = cal_all(np.array(total_y), np.array(total_pred_label))
            sorted_metric_score = sorted(metric_score.items(), key=lambda x: x[0])
            metrics_string = '\t'.join([metric_name[1:] for metric_name, _ in sorted_metric_score])
            score_string = '\t'.join(['{:.2f}'.format(score) for _, score in sorted_metric_score])
            print("{}\t{}\t{}".format('train', epoch, metrics_string))
            print("{}\t{}\t{}".format('train', epoch, score_string))


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
