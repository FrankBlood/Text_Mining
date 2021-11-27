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
import nltk

class Data_Processor(object):
    def __init__(self):
        print('Init...')
        self.data_root = './Datasets/'
        self.original_root = self.data_root + 'original/'
        self.aapr_root = self.original_root + 'AAPR_Dataset/'

        self.exp_root = './exp/'

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
                            label_list.append('0')
                        else:
                            label_list.append('1')

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

    def save_single(self, data, path, clean=0):
        count = 0
        with open(path, 'w') as fw:
            for line in data:
                if clean:
                    line = self.clean_line(line)
                fw.write(line + '\n')
                count += 1
        print("Done for saving {} lines to {}.".format(count, path))

    def save_pair(self, data_input, data_output, input_path, output_path, clean=0):
        self.save_single(data_input, input_path)
        self.save_single(data_output, output_path)

    def save_abs_label(self):
        save_path = self.data_root + 'aapr/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        abs_list, label_list = self.extract_abs_label()
        input_path = save_path + 'data.input'
        output_path = save_path + 'data.output'
        self.save_pair(data_input=abs_list, data_output=label_list, input_path=input_path, output_path=output_path)
        print("There are {} 1 labels.".format(sum(list(map(int, label_list)))/len(label_list)))

    def clean_line(self, line):
        new_line = nltk.word_tokenize(line.lower())
        return ' '.join(new_line)

    def split_data(self, data_name='aapr', fold=10, rate=0.7, clean=0):
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

            if clean:
                mode = '_'.join(['clean'])
                train_input_path = self.data_root + '{}/train_{}_{}.input'.format(data_name, mode, i)
                train_output_path = self.data_root + '{}/train_{}_{}.output'.format(data_name, mode, i)
            else:
                train_input_path = self.data_root + '{}/train_{}.input'.format(data_name, i)
                train_output_path = self.data_root + '{}/train_{}.output'.format(data_name, i)
            self.save_pair(data_input=train_input, data_output=train_output,
                           input_path=train_input_path, output_path=train_output_path,
                           clean=clean)
            print("There are {} 1 labels.".format(sum(list(map(int, train_output)))/len(train_output)))

            if clean:
                mode = '_'.join(['clean'])
                val_input_path = self.data_root + '{}/val_{}_{}.input'.format(data_name, mode, i)
                val_output_path = self.data_root + '{}/val_{}_{}.output'.format(data_name, mode, i)
            else:
                val_input_path = self.data_root + '{}/val_{}.input'.format(data_name, i)
                val_output_path = self.data_root + '{}/val_{}.output'.format(data_name, i)
            self.save_pair(data_input=val_input, data_output=val_output,
                           input_path=val_input_path, output_path=val_output_path,
                           clean=clean)
            print("There are {} 1 labels.".format(sum(list(map(int, val_output))) / len(val_output)))

            if clean:
                mode = '_'.join(['clean'])
                test_input_path = self.data_root + '{}/test_{}_{}.input'.format(data_name, mode, i)
                test_output_path = self.data_root + '{}/test_{}_{}.output'.format(data_name, mode, i)
            else:
                test_input_path = self.data_root + '{}/test_{}.input'.format(data_name, i)
                test_output_path = self.data_root + '{}/test_{}.output'.format(data_name, i)
            self.save_pair(data_input=test_input, data_output=test_output,
                           input_path=test_input_path, output_path=test_output_path,
                           clean=clean)
            print("There are {} 1 labels.".format(sum(list(map(int, test_output))) / len(test_output)))

    def clean_data(self, data_name='aapr', phase='train', fold=0, clean=1, clear=0, *args, **kwargs):


        input_path = '{}{}/{}_{}.input'.format(self.data_root, data_name, phase, fold)

        mode = '_'.join(['clean'])
        save_input_path = '{}{}/{}_{}_{}.input'.format(self.data_root, data_name, phase, fold, mode)
        fw_input = open(save_input_path, 'w')

        with open(input_path, 'r') as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                if clean:
                    new_line = clean_line(line)
                else:
                    new_line = line
                fw_input.write(new_line + '\n')


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
    elif args.phase.split('+')[0] == 'split_data':
        data_processor.split_data(clean=int(args.phase.split('+')[1]))
    else:
        print("What the F**K! There is no {} function.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
