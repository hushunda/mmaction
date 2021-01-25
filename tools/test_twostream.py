#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
测试双流代码
'''

import sys, os
import mmcv
import argparse
import pickle
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('rgb_config', help='test config file path')
    parser.add_argument('flow_config', default=None,help='checkpoint file')
    parser.add_argument('--out', help='output result file', default='default.pkl')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse_args()

    rgb_cfg = mmcv.Config.fromfile(args.rgb_config)
    flow_cfg = mmcv.Config.fromfile(args.flow_config)

    rgb_info = pickle.load(open(os.path.join(rgb_cfg.work_dir,'test.pkl'),'rb'))
    flow_info = pickle.load(open(os.path.join(flow_cfg.work_dir,'test.pkl'),'rb'))

    # 验证数据一致
    for k in rgb_info.keys():
        assert k in flow_info

    ratio = 1.5
    gt_lable = []
    out_pred = []
    for k in rgb_info.keys():
        gt_lable.append(rgb_info[k][1])
        # out_pred.append(rgb_info[k][0]+ratio*flow_info[k][0])
        if softmax(rgb_info[k][0]).max()>softmax(flow_info[k][0]).max():
            pre = rgb_info[k][0]
        else:
            pre = flow_info[k][0]
        out_pred.append(pre)

    top1, top5 = top_k_accuracy(out_pred, gt_lable, k=(1, 5))
    mean_acc = mean_class_accuracy(out_pred,gt_lable)
    print("Mean Class Accuracy = {:.04f}".format(mean_acc * 100))
    print("Top-1 Accuracy = {:.04f}".format(top1 * 100))
    print("Top-5 Accuracy = {:.04f}".format(top5 * 100))

if __name__ == '__main__':
    main()