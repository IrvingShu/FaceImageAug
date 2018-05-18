# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import cv2
import numpy as np
import os
import os.path as osp
import random
import sys


def gamma_trans(img, gamma):
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)


if "__main__" == __name__:
    img_path = "/workspace/data"
    out_dir = "/workspace/data/insightface_color_img_backlight_2.0_2.5"
    img_list_path = '/workspace/data/ms1m_img_list/ms1m_insightface_112x112_rgb.txt'  # insightface_color_img/label/xx.jpg
    count = 0
    with open(img_list_path, 'r') as f:
        lines = f.readlines()
        print ('total num:%d' % (len(lines)))
        for line in lines:  # 每个图片
            count += 1
            if count % 10000 == 0:
                print float(count) / len(lines)
            line = line.strip("\n")
            sub_dir_list = line.split("/")
            file_name = sub_dir_list[2]
            label = sub_dir_list[1]
            final_dir = os.path.join(out_dir, label)
            if not osp.exists(final_dir):
                os.makedirs(final_dir)
            img = cv2.imread(os.path.join(img_path, line))
           # gammaset = float(random.randint(20, 25)) / 10
            img_corrected = gamma_trans(img, 2.5)
            cv2.imwrite(os.path.join(final_dir, "backlight-" + file_name), img_corrected)
  


