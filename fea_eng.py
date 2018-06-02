import csv
import random
import numpy as np
import time
import collections


def gen_fea_dict():
    fea_dict = {}
    index = 0
    with open('data/train.csv', 'r') as train_file:
        r = csv.reader(train_file)
        headers = r.next()
        for row in r:
            for i in range(2, 5):
                if len(row[i]) > 0:
                    fea_name = headers[i] + '-' + row[i]
                    if fea_name not in fea_dict:
                        fea_dict[fea_name] = index
                        index += 1
    with open('model/fea-dict.csv', 'w') as fea_dict_file:
        for key, value in fea_dict.iteritems():
            fea_dict_file.write('%s,%s\n' % (key, value))


def gen_inst():
    fea_dict = {}
    with open('model/fea-dict.csv', 'r') as fea_dict_file:
        r = csv.reader(fea_dict_file)
        for row in r:
            fea_dict[row[0]] = row[1]

    libsvm_file = open('model/inst.txt', 'w')
    with open('data/train.csv', 'r') as train_file:
        r = csv.reader(train_file)
        headers = r.next()
        for row in r:
            libsvm_file.write(row[5])
            libsvm_file.write(' ')
            for i in range(2, 5):
                if len(row[i]) > 0:
                    fea_name = headers[i] + '-' + row[i]
                    if fea_name in fea_dict:
                        libsvm_file.write('%s:%s ' % (fea_dict[fea_name], 1))

            libsvm_file.write('\n')


def split_train_valid():
    lines = open('model/inst.txt', 'r').readlines()
    random.shuffle(lines)
    valid_num = len(lines) / 10
    train_num = len(lines) - valid_num
    train_file = open('model/train.txt', 'w')
    valid_file = open('model/valid.txt', 'w')
    for i in range(train_num):
        train_file.write(lines[i])
    for i in range(train_num, len(lines)):
        valid_file.write(lines[i])


def main():
    # gen_fea_dict()
    # gen_inst()
    split_train_valid()

    pass


if __name__ == '__main__':
    main()
