#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
测试双流代码
'''

import glob
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
    parser.add_argument('--way', help='avg or max', default='avg')
    parser.add_argument('--ratio', help='flow/rgb', default=1.5)
    parser.add_argument('--multi', help='output result file', action='store_true')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse_args()

    rgb_cfg = mmcv.Config.fromfile(args.rgb_config)
    flow_cfg = mmcv.Config.fromfile(args.flow_config)
    if args.multi:
        '''存下所有的精度'''

        rgb_infoes = []
        rgb_file = glob.glob(os.path.join(rgb_cfg.work_dir, 'test_*.pkl'))
        idx_file = {int(os.path.basename(x).split('_')[1].split('.')[0]):x for x in rgb_file}
        sort_idx = sorted(idx_file.keys())
        for idx in sort_idx:
            file = idx_file[idx]
            rgb_infoes.append(pickle.load(open(file, 'rb')))

        flow_infoes = []
        flow_file = glob.glob(os.path.join(flow_cfg.work_dir, 'test_*.pkl'))
        idx_file = {int(os.path.basename(x).split('_')[1].split('.')[0]):x for x in flow_file}
        sort_idx = sorted(idx_file.keys())
        for idx in sort_idx:
            file = idx_file[idx]
            flow_infoes.append(pickle.load(open(file, 'br')))
        # rgb+ flow

        ratio = 1.5
        out_info = []
        for rgb_info,flow_info in zip(rgb_infoes,flow_infoes):

            gt_lable = []
            out_pred = []
            for k in rgb_info.keys():
                gt_lable.append(rgb_info[k][1])
                out_pred.append(rgb_info[k][0][0]+ratio*flow_info[k][0][0])
            # top1, top5 = top_k_accuracy(out_pred, gt_lable, k=(1, 5))
            # print(out_pred)
            mean_acc = mean_class_accuracy(out_pred,gt_lable)
            out_info.append(mean_acc)
        with open(os.path.join(flow_cfg.work_dir,'two_stream.pkl'),'wb') as f:
            pickle.dump(out_info,f)

    else:
        rgb_info = pickle.load(open(os.path.join(rgb_cfg.work_dir,'test.pkl'),'rb'))
        flow_info = pickle.load(open(os.path.join(flow_cfg.work_dir,'test.pkl'),'rb'))

        # 验证数据一致
        for k in rgb_info.keys():
            assert k in flow_info

        ratio = float(args.ratio)
        gt_lable = []
        out_pred = []
        for k in rgb_info.keys():
            gt_lable.append(rgb_info[k][1])
            if args.way == 'avg':
                #out_pred.append(softmax(rgb_info[k][0],dim=0)+ ratio* softmax(flow_info[k][0],dim=0))
                out_pred.append(rgb_info[k][0]+ ratio* flow_info[k][0])
            elif args.way == 'max':
                rgb = softmax(rgb_info[k][0],dim=0)
                flow = softmax(flow_info[k][0],dim=0)
                score = None
                if rgb.max()>flow.max():
                    score = rgb
                else:
                    score = flow
                out_pred.append(score)
            else:
                print('no the way: ',args.way)
                raise NotImplementedError

        top1, top5 = top_k_accuracy(out_pred, gt_lable, k=(1, 5))
        mean_acc = mean_class_accuracy(out_pred,gt_lable)
        print("Mean Class Accuracy = {:.04f}".format(mean_acc * 100))
        print("Top-1 Accuracy = {:.04f}".format(top1 * 100))
        print("Top-5 Accuracy = {:.04f}".format(top5 * 100))

if __name__ == '__main__':
    main()
