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
import random

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

    def save_pair(self, data_input, data_output, input_path, output_path):
        fw_input = open(input_path, 'w')
        fw_output = open(output_path, 'w')
        for input, output in zip(data_input, data_output):
            fw_input.write(input + '\n')
            fw_output.write(output + '\n')
        fw_input.close()
        fw_output.close()
        print("Done for saving input to {}.".format(input_path))
        print("Done for saving output to {}.".format(output_path))

    def save_abs_label(self):
        save_path = self.data_root + 'aapr/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        abs_list, label_list = self.extract_abs_label()
        input_path = save_path + 'data.input'
        output_path = save_path + 'data.output'
        self.save_pair(data_input=abs_list, data_output=label_list, input_path=input_path, output_path=output_path)

    def split_data(self, data_name='aapr', fold=10, rate=0.7):
        with open(self.data_root + '{}/data.input'.format(data_name), 'r') as fp:
            data_input = list(map(lambda x: x.strip(), fp.readlines()))
            print("Successfully load input data from {}.".format(self.data_root + '{}/data.input'.format(data_name)))

        with open(self.data_root + '{}/data.output'.format(data_name), 'r') as fp:
            data_output = list(map(lambda x: x.strip(), fp.readlines()))
            print("Successfully load output data from {}.".format(self.data_root + '{}/data.output'.format(data_name)))

        for i in range(fold):
            print("Processing fold {}...".format(i))
            random.seed(i)
            data = list(zip(data_input, data_output))
            random.shuffle(data)
            data_input, data_output = zip(*data)

            data_size = len(data_output)
            train_input = data_input[:int(data_size*rate)]
            train_output = data_output[:int(data_size*rate)]
            val_input = data_input[int(data_size*rate): int(data_size * (rate + (1-rate)/2))]
            val_output = data_output[int(data_size*rate): int(data_size * (rate + (1-rate)/2))]
            test_input = data_input[int(data_size * (rate + (1-rate)/2)):]
            test_output = data_output[int(data_size * (rate + (1-rate)/2)):]

            train_input_path = self.data_root + '{}/train_{}.input'.format(data_name, i)
            train_output_path = self.data_root + '{}/train_{}.output'.format(data_name, i)
            self.save_pair(data_input=train_input, data_output=train_output,
                           input_path=train_input_path, output_path=train_output_path)

            val_input_path = self.data_root + '{}/val_{}.input'.format(data_name, i)
            val_output_path = self.data_root + '{}/val_{}.output'.format(data_name, i)
            self.save_pair(data_input=val_input, data_output=val_output,
                           input_path=val_input_path, output_path=val_output_path)

            test_input_path = self.data_root + '{}/test_{}.input'.format(data_name, i)
            test_output_path = self.data_root + '{}/test_{}.output'.format(data_name, i)
            self.save_pair(data_input=test_input, data_output=test_output,
                           input_path=test_input_path, output_path=test_output_path)


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
    elif args.phase == 'split_data':
        data_processor.split_data()
    else:
        print("What the F**K! There is no {} function.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
