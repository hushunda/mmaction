#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

def softmax(array):
    array = np.exp(array)
    return array/sum(array)

res_rgb_path = '../work_dirs/mydata/tsn_2d_rgb_resnet101_seg_3_f1s1_b32_g8/test.pkl'
res_flow_path = '../work_dirs/mydata/tsn_2d_flow_resnet101_seg_3_f1s1_b32_g8_lr_0.005/test.pkl'

res_rgb_data = pickle.load(open(res_rgb_path,'rb'))
res_flow_data = pickle.load(open(res_flow_path,'rb'))

print('data load ok')

ratio = 2
label = []
pre_rgb = []
pre_flow = []
pre = []

for k in res_rgb_data.keys():
    label.append(res_rgb_data[k][1])
    pre_rgb.append(np.argmax(res_rgb_data[k][0]))
    pre_flow.append(np.argmax(res_flow_data[k][0]))
    pre.append(np.argmax(ratio*softmax(res_flow_data[k][0])+softmax(res_rgb_data[k][0])))

label=np.array(label)
pre_rgb = np.array(pre_rgb)
pre_flow = np.array(pre_flow)

cls_conf = []
right = pre_rgb == label
for cls in np.unique(label):
    cls_conf.append(sum(right[label==cls])/sum(label==cls))
print(cls_conf)

cls_conf = []
right = pre_flow == label
for cls in np.unique(label):
    cls_conf.append(sum(right[label==cls])/sum(label==cls))
print(cls_conf)

cls_conf = []
right = pre == label
for cls in np.unique(label):
    cls_conf.append(sum(right[label==cls])/sum(label==cls))
print(cls_conf)

cls_conf = np.array(cls_conf)
print(cls_conf[cls_conf>0.5].mean())
print(confusion_matrix(label,pre))


