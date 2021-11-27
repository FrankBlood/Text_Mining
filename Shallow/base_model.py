#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
base_model
======
A class for something.
@author: Guoxiu He
@contact: gxhe@fem.ecnu.edu.cn
@site: https://scholar.google.com/citations?user=2NVhxpAAAAAJ
@time: 20:26, 2021/11/26
@copyright: "Copyright (c) 2021 Guoxiu He. All Rights Reserved"
"""

import os
import sys
import argparse
import datetime

from joblib import dump, load

class base_model(object):
    def __init__(self):
        print('Init...')
        self.model_name = 'ml'

    def build(self):
        self.model = None

    def train(self, X, Y):
        self.model.fit(X, Y)
        return self.model

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        model = load(path)
        return model

    def test(self, X, path=None):
        if path:
            self.model = self.load_model(path)
        Y = self.model.predict(X)
        return Y

    def evaluate(self, X, Y, path=None, phase='train'):
        if path:
            self.model = self.load_model(path)
        Pred_Y = self.model.predict(X)
        acc = accuracy_score(Y, Pred_Y)
        precision = precision_score(Y, Pred_Y, average='micro')
        recall = recall_score(Y, Pred_Y, average='micro')
        print("{} {}: acc\tprecision\trecall".format(self.model_name, phase))
        print("{} {}: {}\t{}\t{}".format(self.model_name, phase, acc, precision, recall))


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

    print('Done base_model!')
