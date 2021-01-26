#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
测试
'''

import os,sys,glob
import torch
from mmaction.models import build_recognizer
import mmcv
from mmaction import datasets
import numpy as np
from mmaction.datasets import build_dataloader
from mmcv.runner import obj_from_dict


py_file_root= os.path.dirname(__file__)

class Recognize():
    def __init__(self,rgb_model_path,flow_model_path=None,
                 rgb_config_path=py_file_root+'/configs/TSN/my_data/tsn_rgb_sknet.py',
                 flow_config_path=py_file_root+'/configs/TSN/my_data/tsn_flow_sknet.py',
                 classind_path=py_file_root+'/data/my_data/annotations/classInd.txt',):
        self.rgb_cfg = mmcv.Config.fromfile(rgb_config_path)
        self.flow_cfg = mmcv.Config.fromfile(flow_config_path)
        self.rgb_model = build_recognizer(rgb_model_path, train_cfg=None, test_cfg=self.rgb_cfg.test_cfg)
        '''TODO'''
        # self.flow_model = build_recognizer(flow_model_path, train_cfg=None, test_cfg=self.flow_cfg.test_cfg)
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
            data_root,test_split = self.rgb_pre(video_path)
            test_config = self.rgb_cfg.data.test
            test_config.ann_file = test_split
            test_config.img_prefix = data_root
            dataset = obj_from_dict(test_config, datasets, dict(test_mode=True))
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=1,
                dist=False,
                shuffle=False)
            with torch.no_grad():
                data=data_loader.__next__()
                result = self.rgb_model(return_loss=False, **data)
        else:
            ''' TODO '''
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

    def rgb_pre(self,video_path):
        video_name = os.path.basename(video_path)
        video_dir = os.path.dirname(video_path)
        img_root = os.path.join(video_dir,video_name.split('.')[0],'v_'+video_name.split('.')[0])
        os.makedirs(img_root)
        cmd = 'ffmpeg -i '+video_path +' '+img_root+'/' +'%img_05d.jpg'
        len_img = len(glob.glob(os.path.join(img_root,'img_*.png')))
        os.system(cmd)
        test_split = os.path.join(video_dir,'test.txt')
        test_info = [os.path.join(video_name.split('.')[0],'v_'+video_name.split('.')[0]),len_img,0]
        with open(test_split,'w') as f:
            f.writelines(' '.join(map(str,test_info)))
        return video_dir,test_split
def test():
    rgb_model_path = 'work_dirs/mydata/tsn_2d_rgb_sknet_seg_3_f1s1_b32_g8/latest.pth'
    video_path = 'data/my_data/test.mov'
    model = Recognize(rgb_model_path)
    pre = model.run(video_path)
    print(pre)

if __name__ == '__main__':
    test()
