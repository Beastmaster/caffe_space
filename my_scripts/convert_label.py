#!/usr/bin/env python
# qin shuo  
# 2016/10/20 


import os
import sys
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import glob
import string

def resize_images(file_list):
    for i,ff in enumerate(file_list):
        if i%500 == 0:
            print 'Processing: ',i #  (float(i)/len(file_list))
        image = imread(ff)
        image = resize(image,(512,512))
        imsave(ff,image)
    pass


def convert_label(file_list,out_path=''):
    for i,file in enumerate(file_list):
        if i%500 == 0:
            print 'Processing: ',i #(float(i)/len(file_list))
        img = imread(file)
        if(len(img.shape)>2):
            img = convert_color_segmentation(img) 
            imsave(file,img)
        else:
            new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i,j] > 0:
                        new_img[i,j] = 1
            imsave(file,new_img)

def convert_label_parameter_help():
    print(
        "Input:",
        "Para1: label file list (.txt file)",
        "Para2: label file data root path (no / in the end)",
        "Para3: output path"
    )

def convert_color_segmentation(img):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    # slow!
    for i in range(0, arr_3d.shape[0]):
        for j in range(0, arr_3d.shape[1]):
            px = img[i,j,0] + img[i,j,1] +img[i,j,2]
            if (px>0):
                arr_2d[i, j] = 1
            else:
                arr_2d[i, j] = 0
    return arr_2d


if __name__ == '__main__':
    convert_label_parameter_help()
    #list_file = '/home/qinshuo/WorkPlace/deeplab2_2/train-DeepLab/exper/seg/data/label/*.*'
    list_file_dir = '/media/D/GE_qinshuo/train/label/*.png'
    
    file_list = glob.glob(list_file_dir)
    convert_label(file_list)

