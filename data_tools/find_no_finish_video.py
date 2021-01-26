#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
找到没有生成完光流的video
'''

import os

data_root = '../data/my_data/rawframes'
out_txt = './nofinishvideo.txt'

nofinish_video = []
for cls in os.listdir(data_root):
    cls_path = os.path.join(data_root,cls)
    for video_name in os.listdir(cls_path):
        all_data = os.listdir(os.path.join(cls_path,video_name))
        nun_frames = len([x for x in all_data if x.startswith('img')])
        num_flow_x = len([x for x in all_data if x.startswith('flow_x')])
        num_flow_y = len([x for x in all_data if x.startswith('flow_y')])
        if num_flow_x==num_flow_y:# and num_flow_x+2==nun_frames and num_flow_x>0:
            continue
        nofinish_video.append(os.path.join(cls,video_name))

print('num of no finish video:  ',len(nofinish_video))
with open(out_txt,'w') as f:
    for line in nofinish_video:
        f.writelines(line+'\n')
