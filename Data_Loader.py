#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Data_Loader
======
A class for something.
@author: Guoxiu He
@contact: gxhe@fem.ecnu.edu.cn
@site: https://scholar.google.com/citations?user=2NVhxpAAAAAJ
@time: 18:48, 2021/11/25
@copyright: "Copyright (c) 2021 Guoxiu He. All Rights Reserved"
"""

import os
import sys
import argparse
import datetime
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from Data_Processor import Data_Processor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class Data_Loader(Data_Processor):
    def __init__(self):
        super(Data_Loader, self).__init__()

    def data_load(self, data_name='aapr', phase='train', fold=0, feature='tf'):
        input_path = '{}{}/{}_{}.input'.format(self.data_root, data_name, phase, fold)
        output_path = '{}{}/{}_{}.output'.format(self.data_root, data_name, phase, fold)
        with open(input_path, 'r') as fp:
            input_data = list(map(lambda x: x.strip(), fp.readlines()))

        with open(output_path, 'r') as fp:
            output_data = list(map(lambda x: int(x.strip()), fp.readlines()))





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

    print('Done Data_Loader!')