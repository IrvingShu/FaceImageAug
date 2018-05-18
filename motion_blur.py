#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:39:04 2018

@author: queeny
使用指南：
指定范围随机size angle
python cnblog.py --random=yes --size=7,15 --angle=0,360
指定size angle
python cnblog.py --random=no --size=7 --angle=115

"""
import matplotlib.pyplot as plt
import pylab
import math
import numpy as np
import cv2
import os
import getopt
import random
import sys
import os.path as osp
#生成卷积核和锚点


def genaratePsf(length, angle):
    EPS = np.finfo(float).eps
    alpha = (angle-math.floor(angle / 180) * 180) / 180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1
    #模糊核大小
    sx = int(math.fabs(length*cosalpha + psfwdt*xsign - length*EPS))
    sy = int(math.fabs(length*sinalpha + psfwdt - length*EPS))
    psf1=np.zeros((sy, sx))
    half = length/2
    #psf1是左上角的权值较大，越往右下角权值越小的核。
    #这时运动像是从右下角到左上角移动
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i*math.fabs(cosalpha) - j*sinalpha
            rad = math.sqrt(i*i + j*j)
            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j]);
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    #运动方向是往左上运动，锚点在（0，0）
    anchor = (0, 0)
    #运动方向是往右上角移动，锚点一个在右上角
    #同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle < 90 and angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1]-1, 0)
    elif angle>-90 and angle<0:#同理：往右下角移动
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1]-1, psf1.shape[0]-1)
    elif anchor<-90:#同理：往左下角移动
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0]-1)
    psf1 = psf1/psf1.sum()
    return psf1, anchor

def gamma_trans(img, gamma):
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)


def func(filepath, savepath, filename, size, angle):
    img = plt.imread(filepath)    #在这里读取图片
    kernel, anchor = genaratePsf(size,angle)
    motion_blur = cv2.filter2D(img,-1,kernel,anchor=anchor)
#   plt.imshow(motion_blur)                      #显示卷积后的图片
    plt.imsave(os.path.join(savepath,"blur-"+filename), motion_blur)
#   pylab.show()


if "__main__" == __name__:
    img_path="/workspace/data"
    out_dir="/workspace/data/insightface_color_img_blur"
    img_list_path='/workspace/data/ms1m_insightface_112x112_rgb.txt'     # insightface_color_img/label/xx.jpg
    count=0
    with open(img_list_path,'r') as f:
        lines=f.readlines()
        print ('total num:%d' % (len(lines)))
        for line in lines:     # 每个图片
            count+=1
            if count%10000==0:
                print float(count)/len(lines)
            line=line.strip("\n")
            sub_dir_list=line.split("/")
            file_name=sub_dir_list[2]
            label=sub_dir_list[1]
            final_dir=os.path.join(out_dir,label)
            if not osp.exists(final_dir):
                os.makedirs(final_dir)
            size = random.randint(11, 17)
            angle = random.randint(0,360)
            while angle == 0 or angle == 90 or angle == 180 or angle == 270 or angle == 360:
                angle = random.randint(0, 360)
            func(os.path.join(img_path,line), final_dir, file_name, size, angle)

