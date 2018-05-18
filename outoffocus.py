#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:11:26 2018
@author: queeny
"""
import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np
import math
import fractions
import os
import os.path as osp
import random

R=5
# theta=2.7 #3.0
C=0.09

def func(filepath,savepath,filename,theta):
    img = plt.imread(filepath)    #在这里读取图片
    # plt.imshow(img)                                     #显示读取的图片
    # pylab.show()
    fil=[[0 for x in range (0,R)] for x in range (0,R)]
    center=[R/2,R/2]
    for i in range(0,R):
        for j in range(0,R):
            new_pos=[i-center[0],j-center[1]]
            if new_pos[0]<=R/2 and new_pos[1]<=R/2:
                fil[i][j]=C*math.exp(-(new_pos[0]**2+new_pos[1]**2)/(2*theta**2))
   # print fil
    kernel=np.array(fil)
    res = cv2.filter2D(img,-1,kernel)     #使用opencv的卷积函数
    # plt.imshow(res)                      #显示卷积后的图片
    plt.imsave(os.path.join(savepath,"OutOfFocus-"+filename),res)
    # pylab.show()




if "__main__" == __name__:
    img_path="/workspace/data"
    out_dir="/workspace/data/insightface_color_img_OutOfFocus"
    img_list_path='/workspace/data/ms1m_insightface_112x112_rgb.txt'     # insightface_color_img/label/xx.jpg
    count=0
    with open(img_list_path,'r') as f:
        lines=f.readlines()
        print ('total num:%d' % (len(lines)))
        for line in lines:     # 每个图片
            count+=1
 #           if count>30:
  #              break
            if count%10000==0:
                print float(count)/len(lines)
            line=line.strip("\n")
            count += 1
            if count > 30:
                break
            sub_dir_list=line.split("/")
            file_name=sub_dir_list[2]
            label=sub_dir_list[1]
            final_dir=os.path.join(out_dir,label)
            if not osp.exists(final_dir):
                os.makedirs(final_dir)
            theta=float(random.randint(15,27))/10
            func(os.path.join(img_path,line), final_dir, file_name, theta)
