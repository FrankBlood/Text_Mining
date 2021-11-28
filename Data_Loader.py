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
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from joblib import load, dump
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical


class Data_Loader(Data_Processor):
    def __init__(self):
        super(Data_Loader, self).__init__()

    def data_load(self, data_name='aapr', phase='train', fold=0, feature='tf', clean=0, clear=0, *args, **kwargs):
        if clean:
            mode = '_'.join(['clean'])
            input_path = '{}{}/{}_{}_{}.input'.format(self.data_root, data_name, phase, mode, fold)
        else:
            input_path = '{}{}/{}_{}.input'.format(self.data_root, data_name, phase, fold)
        output_path = '{}{}/{}_{}.output'.format(self.data_root, data_name, phase, fold)
        with open(input_path, 'r') as fp:
            input_data = list(map(lambda x: x.strip(), fp.readlines()))
        with open(output_path, 'r') as fp:
            output_data = list(map(lambda x: int(x.strip()), fp.readlines()))

        save_folder = self.exp_root + data_name
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = save_folder + '/ml_feature.{}.{}'.format(feature, fold)

        if phase == 'train' and (not os.path.exists(save_path) or clear):
            if feature == 'tf':
                feature_extractor = CountVectorizer().fit(input_data)
            elif feature == 'tfidf':
                feature_extractor = TfidfVectorizer().fit(input_data)
            elif feature == 'lda':
                dictionary = Dictionary([text.strip().split() for text in input_data])
                dictionary_save_path = save_path + '.dict'
                if not os.path.exists(dictionary_save_path) or clear:
                    dump(dictionary, dictionary_save_path)
                    print("Successfully save dict to {}.".format(dictionary_save_path))
                corpus = [dictionary.doc2bow(text.strip().split()) for text in input_data]
                num_topics = 20
                if 'num_topics' in kwargs:
                    num_topics = kwargs['num_topics']
                feature_extractor = LdaModel(corpus, num_topics=num_topics)
            else:
                raise RuntimeError("Please confirm which feature you need.")
            if not os.path.exists(save_path) or clear:
                dump(feature_extractor, save_path)
                print("Successfully save features to {}.".format(save_path))
        else:
            feature_extractor = load(save_path)

        if feature == 'lda':
            dictionary = load(save_path+'.dict')
            x = [feature_extractor.get_document_topics(dictionary.doc2bow(text.strip().split()), minimum_probability=0)
                 for text in input_data]
            x = [[prob for (topic, prob) in line] for line in x]
        else:
            x = feature_extractor.transform(input_data)
        y = output_data

        return x, y

    def data_generator(self, input_path, output_path,
                       word_dict=None, batch_size=64):

        with open(input_path, 'r') as fp:
            input_data = fp.readlines()

        with open(output_path, 'r') as fp:
            output_data = fp.readlines()

        for i in range(0, len(output_data), batch_size):
            batch_input = []
            for line in input_data[i: i+batch_size]:
                new_line = []
                for word in line.strip().split():
                    new_line.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                batch_input.append(new_line)
            batch_x = pad_sequences(batch_input)
            batch_output = [int(label.strip()) for label in output_data[i: i+batch_size]]
            batch_y = np.array(batch_output)
            yield batch_x, batch_y


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
