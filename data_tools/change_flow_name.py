#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
修改光流文件名字
假定u为x,v为y
'''

import shutil
import os

src_root = '../data/hmdb51/tvl1_flow/'
out_root = '../data/hmdb51/rawframes/'
index_path = '../data/hmdb51/hmdb51_%s_split_1_rawframes.txt'

all_file = {}
# for phase in ['train','val']:
#     a = [x.strip().split(' ')[0] for x in open(index_path%phase,'r').readlines()]
#     all_file.extend(a)
for v in os.listdir(out_root):
    for n in os.listdir(os.path.join(out_root,v)):
        all_file[n] = v+'/'+n


def change_name():
    ## x
    i=0
    for file in os.listdir(os.path.join(src_root,'u')):
        if file.endswith('.bin'):continue
        if file not in all_file: continue
        for img_n in os.listdir(os.path.join(src_root,'u',file)):
            if img_n.endswith('.jpg'):
                idx = int(img_n.strip('frame').strip('.jpg'))
                in_path = os.path.join(src_root,'u',file,img_n)
                out_path = os.path.join(out_root,all_file[file])
                if not os.path.exists(out_path):
                    print(out_path)
                i+=1
                if os.path.exists(out_path+'/flow_x_%0.5d.jpg'%idx):
                    continue
                os.renames(in_path, os.path.join(out_path, 'flow_y_%0.5d.jpg' % idx))

    ## y
    for file in os.listdir(os.path.join(src_root,'v')):
        if file.endswith('.bin'):continue
        if file not in all_file: continue
        for img_n in os.listdir(os.path.join(src_root,'v',file)):
            if img_n.endswith('.jpg'):
                idx = int(img_n.strip('frame').strip('.jpg'))
                in_path = os.path.join(src_root,'v',file,img_n)
                out_path = os.path.join(out_root, all_file[file])
                # os.makedirs(out_path, exist_ok=True)
                os.renames(in_path, os.path.join(out_path, 'flow_y_%0.5d.jpg' % idx))

def change_HandstandPushups_name():
    ## HandstandPushups
    file = 'HandstandPushups'
    out_file = 'HandStandPushups'
    for v_n in os.listdir(os.path.join(out_root,file)):
        v_path = os.path.join(out_root,file,v_n)
        if not os.path.isdir(v_path): continue
        for img_name in os.listdir(v_path):
            in_path = os.path.join(v_path, img_name)
            out_path =os.path.join(v_path.replace(file,out_file))#, img_name.replace('u','y'))
            # os.makedirs(out_path, exist_ok=True)
            os.renames(in_path,os.path.join(out_path,img_name))

if __name__ == '__main__':
    change_name()
    # change_HandstandPushups_name()




