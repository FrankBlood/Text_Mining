#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Data_Processor
======
A class for something.
@author: Guoxiu He
@contact: gxhe@fem.ecnu.edu.cn
@site: https://scholar.google.com/citations?user=2NVhxpAAAAAJ
@time: 20:44, 2021/11/22
@copyright: "Copyright (c) 2021 Guoxiu He. All Rights Reserved"
"""

import os
import sys
import argparse
import datetime

import json

class Data_Processor(object):
    def __init__(self):
        print('Init...')
        self.data_root = './Datasets/'
        self.aapr_root = self.data_root + 'AAPR_Dataset/'

    def show_json_data(self):
        for i in range(4):
            path = self.aapr_root + 'data{}'.format(i+1)
            with open(path, 'r') as fp:
                data = json.load(fp)
            print(len(data))


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    data_processor = Data_Processor()
    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    elif args.phase == 'show_json_data':
        data_processor.show_json_data()
    else:
        print("What the F**K! There is no {} function.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
