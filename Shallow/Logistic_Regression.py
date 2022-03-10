#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
svm
======
A class for something.
"""

import os
import sys
import argparse
import datetime
from Shallow.Base_Model import Base_Model
from sklearn.linear_model import LogisticRegression


class Logistic_Regression(Base_Model):
    def __init__(self, metrics_num):
        super(Logistic_Regression, self).__init__()
        self.model_name = 'lr'

    def build(self):
        self.model = LogisticRegression()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done svm!')
