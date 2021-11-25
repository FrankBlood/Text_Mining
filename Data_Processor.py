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
        self.original_root = self.data_root + 'original/'
        self.aapr_root = self.original_root + 'AAPR_Dataset/'

    def show_json_data(self):
        for i in range(4):
            path = self.aapr_root + 'data{}'.format(i+1)
            with open(path, 'r') as fp:
                data = json.load(fp)
            print(len(data))
            for paper_id, info in data.items():
                for key, value in info.items():
                    print(key)
                break

    def extract_abs_label(self):
        abs_list = []
        category_list = []
        venue_list = []
        count = 0
        error_count = 0
        for i in range(4):
            path = self.aapr_root + 'data{}'.format(i+1)
            with open(path, 'r') as fp:
                data = json.load(fp)
                for paper_id, info in data.items():
                    abs = info['abstract']
                    category = info['category']
                    venue = info['venue']
                    if abs and category and venue:
                        abs_list.append(abs)
                        category_list.append(category)
                        venue_list.append(venue)
                    else:
                        print("Error abs: {}".format(abs))
                        print("Error label: {}".format(category))
                        print("Error venue: {}".format(venue))
                        error_count += 1
                    count += 1

        top_num = 5
        print("Print top {} abs:".format(top_num))
        for abs in abs_list[:top_num]:
            print(abs)

        print("Print top {} category:".format(top_num))
        for category in category_list[:top_num]:
            print(category)

        print("Print top {} venue:".format(top_num))
        for venue in venue_list[:top_num]:
            print(venue)

        print("There are {} papers.".format(count))
        print("There are {} error abs or labels.".format(error_count))


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
    elif args.phase == 'extract_abs_label':
        data_processor.extract_abs_label()
    else:
        print("What the F**K! There is no {} function.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
