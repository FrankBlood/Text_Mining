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
from utils.metrics import cal_all


class Base_Model(object):
    def __init__(self):
        print('Init...')
        self.model_name = 'ml'

    def build(self):
        self.model = None

    def train(self, x, y):
        self.model.fit(x, y)
        return self.model

    def save_model(self, path):
        dump(self.model, path)
        print("Successfully save {} model to {}.".format(self.model_name, path))

    def load_model(self, path):
        model = load(path)
        return model

    def test(self, x, path=None):
        if path:
            self.model = self.load_model(path)
        y = self.model.predict(x)
        return y

    def evaluate(self, x, y, path=None, phase='train'):
        if path:
            self.model = self.load_model(path)
        pred_y = self.model.predict(x)
        acc, precision, recall, f1_score = cal_all(y, pred_y)
        print("{}\t{}\tacc\tprecision\trecall\tf1score".format(self.model_name, phase))
        print("{}\t{}\t{}\t{}\t{}\t{}".format(self.model_name, phase, acc, precision, recall, f1_score))


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
