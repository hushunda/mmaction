#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
画出双流结果的每个类别的柱状
'''
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import mmcv
import argparse
import pickle
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)
def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('--rgb_config',default='configs/TSN/ucf101/tsn_rgb_sknet.py', help='test config file path')
    # parser.add_argument('--rgb_config',default='configs/TSN/hmdb51/tsn_rgb_sknet.py', help='test config file path')
    # parser.add_argument('--flow_config', default='configs/TSN/hmdb51/tsn_flow_sknet.py',help='checkpoint file')
    parser.add_argument('--flow_config', default='configs/TSN/ucf101/tsn_flow_sknet.py',help='checkpoint file')
    parser.add_argument('--out', help='output result file', default='default.pkl')
    args = parser.parse_args()
    return args

def main():
    '''画出没类的精度'''
    global args
    args = parse_args()

    rgb_cfg = mmcv.Config.fromfile(args.rgb_config)
    flow_cfg = mmcv.Config.fromfile(args.flow_config)

    rgb_info = pickle.load(open(os.path.join(rgb_cfg.work_dir,'test.pkl'),'rb'))
    flow_info = pickle.load(open(os.path.join(flow_cfg.work_dir,'test.pkl'),'rb'))

    path_dir = os.path.dirname(rgb_cfg.data_root)
    classInd = [x.split(' ') for x in open(path_dir+'/annotations/classInd.txt','r').readlines()]
    classInd = {int(x[0]):x[1]for x in classInd}

    # 验证数据一致
    for k in rgb_info.keys():
        assert k in flow_info

    ratio = 1.5
    gt_lable = []
    out_pred = []
    for k in rgb_info.keys():
        gt_lable.append(rgb_info[k][1])
        out_pred.append(np.argmax(softmax(rgb_info[k][0]+ratio*flow_info[k][0],dim=0)))

    plot_info = {}
    out_pred = np.array(out_pred)
    gt_lable = np.array(gt_lable)
    all_right = out_pred ==gt_lable
    for k,v in classInd.items():
        mask = gt_lable==k
        p = sum(all_right[mask])/sum(mask)
        plot_info[v.strip()]=p

    save_path = path_dir+'/two_stream_bar.png'
    plot(plot_info,save_path)

def plot(data,path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.figure(figsize=(6, 6.5))
    ax.scatter(list(data.keys()), list(data.values()))
    ax.plot(list(data.keys()), list(data.values()))
    plt.xticks(rotation=90,fontsize=12)
    ax.legend()
    if path!=None:
        plt.savefig(path)
    plt.show()

def plot_bar():
    '''画出迭代的精度'''
    sk_root = 'work_dirs/hmdb51/tsn_2d_flow_resnext/'
    re_root = 'work_dirs/hmdb51/tsn_2d_flow_sknet/'
    #
    sk_info = pickle.load(open(os.path.join(sk_root,'two_stream.pkl'),'rb'))
    re_info = pickle.load(open(os.path.join(re_root,'two_stream.pkl'),'rb'))

    labels = [x*10000 for x in range(len(sk_info))]
    width = 0.2
    x = np.arange(len(sk_info))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, sk_info, width, label='sknet101')
    # re_info = np.array(re_info)
    # re_info[1:] = re_info[1:]-0.05
    rects2 = ax.bar(x + width / 2, re_info, width, label='ResNeXt101')
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

def test():
    txt = 'asdfghjklqwertyuiopzxcvbnmrtghnsadqvwqw' \
          'qlopashnsajhcfjishidoncksmaolkhnyfgyhujimkfgyhujikjbvg'
    data = {txt[i*2:(i*2+2)]:i for i in range(51)}
    plot(data)

if __name__ == '__main__':
    main()
    # plot_bar()