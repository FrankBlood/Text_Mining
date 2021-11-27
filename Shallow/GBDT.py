#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
svm
======
A class for something.
@author: Guoxiu He
@contact: gxhe@fem.ecnu.edu.cn
@site: https://scholar.google.com/citations?user=2NVhxpAAAAAJ
@time: 12:49, 2021/11/27
@copyright: "Copyright (c) 2021 Guoxiu He. All Rights Reserved"
"""

import os
import sys
import argparse
import datetime
from Shallow.Base_Model import Base_Model
from sklearn.ensemble import GradientBoostingClassifier


class GBDT(Base_Model):
    def __init__(self, metrics_num, n_estimators=100):
        super(GBDT, self).__init__()
        self.model_name = 'gbdt'
        self.n_estimators = n_estimators

    def build(self):
        self.model = GradientBoostingClassifier()


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

    print('Done svm!')
