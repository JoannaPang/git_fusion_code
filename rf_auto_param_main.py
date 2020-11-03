#! D:\Users\zhennanpang\Anaconda3\envs python
# -*- coding: utf-8 -*-
# @Time : 2020/8/20 16:47
# @Author : ZhennanpPang
# @File : aotu_param_main.py
# @Software: PyCharm

from utils import utils
from auto_param_gbdt_lr import compute_params
import scipy.io as scio
import numpy as np
import h5py
import os
from sklearn.ensemble import RandomForestClassifier
import math


def generate_x_train(hks, deep_feature):
    x_train = np.hstack((hks, deep_feature))
    return x_train

# utils1
def multi2onehot(y_train):
    """

    :param y_train:
    :return:
    """
    one_y = np.empty([0, 3])
    zero = np.array([1, 0, 0])
    one = np.array([0, 1, 0])
    two = np.array([0, 0, 1])
    for value in y_train:
        if value[0] == 0:
            one_y = np.vstack((one_y, zero))
        elif value[0] == 1:
            one_y = np.vstack((one_y, one))
        elif value[0] == 2:
            one_y = np.vstack((one_y, two))

    y_train = np.squeeze(y_train)
    # print('one y \n', one_y)
    # print('Success Load Data!', y_train.shape)
    # print('Success Load Data!', y_train)

    from sklearn.utils.multiclass import type_of_target

    print(type_of_target(y_train))
    print(type(y_train))
    print(type_of_target(one_y))
    print(type(one_y))

    return y_train, one_y

def normalize(x_list, max_list, min_list):
    """
    normalize the x_list
    :param x_list: the raw data of training sets
    :param max_list: the max list of raw data
    :param min_list: the min list of raw data
    :return: the normalized list
    """
    index = 0
    scalar_list = []
    for x in x_list:
        x_max = max_list[index]
        x_min = min_list[index]
        if x_max == x_min:
            x = 1.0
        else:
            x = np.round((x - x_min) / (x_max - x_min), 4)
        scalar_list.append(x)
        index += 1
    return scalar_list


def norm_data(x_train_nor):
    """
    normalize the train data and the test data
    :param x_train_nor: data of features for the normalization
    :param y_train_nor: data of tags for the normalization
    :return: the normalized features and normal labels
    """
    data_array = np.asmatrix(x_train_nor)
    max_list = np.max(data_array, axis=0)
    min_list = np.min(data_array, axis=0)

    scalar_data_mat = []
    for row_i in range(0, len(data_array)):
        if row_i == 0:
            row = data_array[row_i]
            row = row.tolist()
            scalar_data_mat.append(row)
        if row_i != 0:
            row = data_array[row_i]
            row = row.tolist()
            # print('row_i, row: ', row_i, row)
            scalar_row = normalize(row, max_list, min_list)
            scalar_data_mat.append(scalar_row)

    scalar_data_mat_np = np.array(scalar_data_mat)

    return scalar_data_mat_np



# load on mat file, return hks, deep_featrue
def load_x_file(filepath, is_norm=False):

    data =  h5py.File(filepath,'r')
    img_struct = data['img_struct'][0][0]

    hks = data[img_struct]['hks']
    hks = np.transpose(hks)

    deep_feature = data[img_struct]['deep_feature']
    deep_feature = np.transpose(deep_feature)

    x = data[img_struct]['X']
    x = np.transpose(x)

    y = data[img_struct]['Y']
    y = np.transpose(y)

    xy = np.hstack((x, y))

    r = data[img_struct]['R']
    r = np.transpose(r)

    g = data[img_struct]['G']
    g = np.transpose(g)

    b = data[img_struct]['B']
    b = np.transpose(b)

    rgb = np.hstack((r, g, b))

    if is_norm:
        print("normal ", xy[0:9])
        xy = norm_data(xy)
        print("normal ", xy[0:9])

    point = data[img_struct]['Point']
    len_p = len(point)
    point = np.transpose(point)
    point_array = []

    for i in range(0, len_p):
        temp = data[point[0][i]]
        temp = np.transpose(temp)
        point_array.append(temp)

    x_train = np.hstack((xy, rgb, hks, deep_feature))

    # generate_x_train(hks, deep_feature)
    return xy, rgb, hks, deep_feature, point_array, x_train


def load_y_file(filepath):

    data =  h5py.File(filepath,'r')
    img_struct = data['label_struct'][0][0]
    label = data[img_struct]['label']
    label = np.transpose(label)
    return label


def load_files(root_path_x, root_path_y, is_norm=False):
    xys = np.empty([0, 2])
    rgbs = np.empty([0, 3])
    hkss = np.empty([0, 23])
    deep_featrues = np.empty([0, 3])
    x_trains = np.empty([0, 31]) #need to fix!
    y_trains = np.empty([0, 1])
    for file in os.listdir(root_path_x):

        x_file_path = root_path_x + "/"+ file
        y_file_path = root_path_y + "/" + file

        # get hks deep_featrue x_train
        xy, rgb, hks, deep_featrue, point, x_train = load_x_file(x_file_path, is_norm)
        y_train = load_y_file(y_file_path)
        # x_train = generate_x_train(hks, deep_featrue)

        # col cat
        x_trains = np.vstack((x_trains, x_train))

        xys = np.vstack((xys, xy))
        rgbs = np.vstack((rgbs, rgb))
        hkss = np.vstack((hkss, hks))
        deep_featrues = np.vstack((deep_featrues, deep_featrue))
        y_trains = np.vstack((y_trains, y_train))


    return x_trains, y_trains


def train_test_get():
    # load train
    data_train_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/data_for_rf/cnn3'
    label_data_train_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/data_for_rf/label'
    x_train_, y_train_ = load_files(data_train_path, label_data_train_path, is_norm = False)
    y_train_, y_train_onhot_ = multi2onehot(y_train_)

    print('Success Load Data!', x_train_.shape)
    print('Success Load Data!', x_train_[0].shape)

    # load test
    data_test_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/data_for_rf/cnn3'
    lable_data_test_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/data_for_rf/label'
    x_test_, y_test_ = load_files(data_test_path, lable_data_test_path, is_norm = False)
    y_test_, y_test_onhot_ = multi2onehot(y_test_)

    print('Success Load Data!', x_train_.shape)
    print('Success Load Data!', x_train_[0].shape)

    return x_train_, y_train_, x_test_, y_test_

import pickle

def train_process():
    x_train_, y_train_, x_test_, y_test_ = train_test_get()
    # model = compute_params(x_train_, y_train_)
    # print('best model:', model)
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(x_train_,y_train_)
    print(rf.predict(x_train_))

    current_model_path = './model/'
    if not os.path.isdir(current_model_path):
        os.mkdir(current_model_path)
    current_model_name = 'rf_clf.pickle'

    with open(current_model_path+current_model_name, 'wb') as f:
        pickle.dump(rf, f)
        print('model saved successfully!')
        f.close()

from sklearn.metrics import accuracy_score


def test_process():
    x_train_, y_train_, x_test_, y_test_ = train_test_get()
    current_model_path = './model/'
    current_model_name = 'rf_clf.pickle'
    with open(current_model_path + current_model_name, 'rb') as f:
        rf = pickle.load(f)
        y_pred_ = rf.predict(x_test_)
        print(accuracy_score(y_test_, y_pred_))
        f.close()


import matplotlib.pyplot as pyplot

def test_process_one_image(root_path_x, root_path_y):
   # data = h5py.File(filepath, 'r')
    current_model_path = './model/'
    current_model_name = 'rf_clf.pickle'

    with open(current_model_path + current_model_name, 'rb') as f:
       rf = pickle.load(f)

    for file in os.listdir(root_path_x):

        x_file_path = root_path_x + "/"+ file
        y_file_path = root_path_y + "/" + file

        # get hks deep_featrue x_train
        xy, rgb, hks, deep_featrue, points, x_test = load_x_file(x_file_path, is_norm=False)

        result_image = np.zeros((700, 700))
        y_test = load_y_file(y_file_path)


        y_pred_ = rf.predict(x_test)

        for i in range(0, len(y_pred_)):
               # print(y_pred_[i])
            if(y_pred_[i]>0):
                for p in points[i]:
                    result_image[p[0]][p[1]]=y_pred_[i]


        #for i in range(0,len(result_image)):
        #    print(result_image[i])
        pyplot.imshow(result_image)
        pyplot.show()



if __name__ == '__main__':

    # train_process()
    # test_process()
    test_path_x_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/data_for_rf/cnn3'
    test_path_y_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/data_for_rf/label'
    test_process_one_image(test_path_x_, test_path_y_)
    # model = compute_params(x_train_, one_y)

