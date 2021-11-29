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
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Base_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion_name = criterion_name
        self.optimizer_name = optimizer_name

        self.metrics_num = 4
        if 'metrics_num' in kwargs:
            self.metrics_num = kwargs['metrics_num']

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
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
        print("Device: {}.".format(self.device))

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        avg_embed = torch.mean(embed, dim=1)
        # out = self.softmax(self.fc(avg_embed))
        hidden = self.fc1(avg_embed)
        out = self.fc_out(hidden)
        return out

    def train_model(self, model, data_generator, input_path, output_path, word_dict,
                    input_path_val=None, output_path_val=None,
                    input_path_test=None, output_path_test=None,
                    save_folder=None):
        model.to(self.device)
        model.train()
        best_score = 0
        for epoch in range(self.num_epochs):
            total_y, total_pred_label = [], []
            total_loss = 0
            step_num = 0
            sample_num = 0
            for x, y in data_generator(input_path, output_path, word_dict, batch_size=self.batch_size):
                batch_x = torch.LongTensor(x).to(self.device)
                batch_y = torch.LongTensor(y).to(self.device)
                batch_pred_y = model(batch_x)
                model.zero_grad()
                loss = self.criterion(batch_pred_y, batch_y)
                loss.backward()
                self.optimizer.step()
                pred_y_label = list(np.argmax(batch_pred_y.cpu().detach().numpy(), axis=-1))

                total_y += list(y)
                total_pred_label += pred_y_label

                total_loss += loss.item() * len(y)
                step_num += 1
                sample_num += len(y)
            print("Have trained {} steps.".format(step_num))
            metric = cal_all
            if self.metrics_num == 4:
                metric = cal_all
            metric_score = metric(np.array(total_y), np.array(total_pred_label))
            sorted_metric_score = sorted(metric_score.items(), key=lambda x: x[0])
            metrics_string = '\t'.join(['loss'] + [metric_name[1:] for metric_name, _ in sorted_metric_score])
            score_string = '\t'.join(['{:.2f}'.format(total_loss/sample_num)] + ['{:.2f}'.format(score) for _, score in sorted_metric_score])
            print("{}\t{}\t{}".format('train', epoch, metrics_string))
            print("{}\t{}\t{}".format('train', epoch, score_string))

            if input_path_val and output_path_val:
                metric_score = \
                    self.eval_model(model, data_generator, input_path_val, output_path_val, word_dict, 'val', epoch)
                acc = metric_score['1acc']
                torch.save(model, '{}{}.ckpt'.format(save_folder, epoch))
                print("Save model to {}.".format('{}{}.ckpt'.format(save_folder, epoch)))
                if acc > best_score:
                    best_score = acc
                    torch.save(model, '{}{}.ckpt'.format(save_folder, 'best'))
                    print("Save model to {}.".format('{}{}.ckpt'.format(save_folder, 'best')))

            if input_path_test and output_path_test:
                self.eval_model(model, data_generator, input_path_test, output_path_test, word_dict, 'test', epoch)

        if input_path_test and output_path_test:
            model = torch.load('{}{}.ckpt'.format(save_folder, 'best'))
            model.eval()
            self.eval_model(model, data_generator, input_path_test, output_path_test, word_dict, 'test', 'final')

    def eval_model(self, model, data_generator, input_path, output_path, word_dict, phase, epoch):
        model.to(self.device)
        total_y, total_pred_label = [], []
        total_loss = 0
        step_num = 0
        sample_num = 0
        for x, y in data_generator(input_path, output_path, word_dict, batch_size=self.batch_size):
            batch_x = torch.LongTensor(x).to(self.device)
            batch_y = torch.LongTensor(y).to(self.device)
            batch_pred_y = model(batch_x)
            loss = self.criterion(batch_pred_y, batch_y)
            pred_y_label = list(np.argmax(batch_pred_y.cpu().detach().numpy(), axis=-1))
            total_y += list(y)
            total_pred_label += pred_y_label

            total_loss += loss.item() * len(y)
            step_num += 1
            sample_num += len(y)
        print("Have {} {} steps.".format(phase, step_num))
        metric = cal_all
        if self.metrics_num == 4:
            metric = cal_all
        metric_score = metric(np.array(total_y), np.array(total_pred_label))
        sorted_metric_score = sorted(metric_score.items(), key=lambda x: x[0])
        metrics_string = '\t'.join(['loss'] + [metric_name[1:] for metric_name, _ in sorted_metric_score])
        score_string = '\t'.join(
            ['{:.2f}'.format(total_loss / sample_num)] + ['{:.2f}'.format(score) for _, score in sorted_metric_score])
        print("{}\t{}\t{}".format(phase, epoch, metrics_string))
        print("{}\t{}\t{}".format(phase, epoch, score_string))
        return metric_score


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
