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

    ##############################
    # AAPR
    ##############################
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
        category_dict = {}
        venue_list = []
        venue_dict = {}
        label_list = []

        count = 0
        error_count = 0
        for i in range(4):
            path = self.aapr_root + 'data{}'.format(i+1)
            with open(path, 'r') as fp:
                data = json.load(fp)
                for paper_id, info in data.items():
                    abs = info['abstract'].strip()
                    category = info['category'].strip()
                    venue = info['venue'].strip()
                    if abs and category and venue:
                        abs_list.append(abs)
                        category_list.append(category)
                        if category not in category_dict:
                            category_dict[category] = 1
                        else:
                            category_dict[category] += 1

                        venue_list.append(venue)
                        if venue not in venue_dict:
                            venue_dict[venue] = 1
                        else:
                            venue_dict[venue] += 1

                        if venue in {'CoRR', 'No'}:
                            label_list.append('Rejected')
                        else:
                            label_list.append('Accepted')

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

        print("category_dict:\n", category_dict)
        print("venue_dict:\n", venue_dict)

        print("There are {} papers.".format(count))
        print("There are {} error abs or labels.".format(error_count))
        return abs_list, label_list

    def save_abs_label(self):
        save_path = self.data_root + 'aapr/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        abs_list, label_list = self.extract_abs_label()
        fw_input = open(save_path + 'data.input', 'w')
        fw_output = open(save_path + 'data.output', 'w')
        for abs, label in zip(abs_list, label_list):
            fw_input.write(abs.strip() + '\n')
            fw_output.write(label.strip() + '\n')
        fw_input.close()
        fw_output.close()


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
    elif args.phase == 'save_abs_label':
        data_processor.save_abs_label()
    else:
        print("What the F**K! There is no {} function.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
