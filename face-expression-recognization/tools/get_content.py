#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from os.path import basename
from os.path import join


if __name__ == '__main__':

    path = '../database/jaffe/'
    type_list = os.listdir(path)
    name = "jaffe.txt"
    file_name = "../data_list/" + name
    f = open(file_name, 'a+')
    for i, type in enumerate(type_list):
        type_path = join(path, type)

        if os.path.isdir((type_path)):
            pic_list = os.listdir(type_path)

            for pic in pic_list:
                img_path = join(type_path,pic)
                # print(img_path)
                if img_path.endswith('.png') or img_path.endswith('.jpg'):
                    f.write(img_path + " " +str(i)+ "\n")
    f.close()

    print("Are you happy :)?")






