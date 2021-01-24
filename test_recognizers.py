#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
测试
'''

import torch
from mmaction.models import build_recognizer
import mmcv
from mmaction import datasets
import numpy as np
from mmaction.datasets import build_dataloader
from mmcv.runner import obj_from_dict


class Recognize():
    def __init__(self,rgb_model_path,flow_model_path,rgb_config_path=None,flow_config_path=None,classind_path=None):
        self.rgb_cfg = mmcv.Config.fromfile(rgb_config_path)
        self.flow_cfg = mmcv.Config.fromfile(flow_config_path)
        self.rgb_model = build_recognizer(rgb_model_path, train_cfg=None, test_cfg=self.rgb_cfg.test_cfg)
        self.flow_model = build_recognizer(flow_model_path, train_cfg=None, test_cfg=self.flow_cfg.test_cfg)
        #
        self.classind = None
        with open(classind_path,'r') as f:
            tmp = [x.strip().split(' ') for x in f.readlines()]
            self.classind = {int(i):x for i,x in tmp}
        if self.classind==None:
            raise ModuleNotFoundError("No classind file")

    def run(self,video_path,flow_path,algorithm='rgb'):
        assert algorithm in ['rgb','flow']

        if algorithm=='rgb':
            dataset = obj_from_dict(self.rgb_cfg.data.test, datasets, dict(test_mode=True))
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=1,
                dist=False,
                shuffle=False)
            with torch.no_grad():
                for data in enumerate(data_loader):
                    result = self.rgb_model(return_loss=False, **data)
        else:
            dataset = obj_from_dict(self.flow_cfg.data.test, datasets, dict(test_mode=True))
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=1,
                dist=False,
                shuffle=False)
            with torch.no_grad():
                for data in enumerate(data_loader):
                    result = self.flow_model(return_loss=False, **data)
        return self.classind[np.argmax(result.mean(axis=0),axis=1)]


def test():
    pass

if __name__ == '__main__':
    test()