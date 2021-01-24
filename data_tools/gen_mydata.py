#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
处理自己的数据集,

    my_data/my_data_train_split_1_rawframes.txt
    my_data/my_data_val_split_1_rawframes.txt
    my_data/my_data_test_split_1_rawframes.txt
    annotations/classInd.txt
    annotations/trainlist01.txt
    annotations/testlist01.txt
'''

import os,glob


def main():
    src_root = '../data/my_data/rawframes'
    out_root = '../data/my_data'
    os.makedirs(os.path.join(out_root,'annotations'),exist_ok=True)

    all_cls = [x for x in os.listdir(src_root) if os.path.isdir(os.path.join(src_root,x))]
    all_cls.sort()
    with open(os.path.join(out_root,'annotations','classInd.txt'),'w') as f:
        for i,x in enumerate(all_cls):
            f.writelines(' '.join(map(str,[i,x]))+'\n')

    train = []
    test = []
    ratio = 0.8
    for idx,cls in enumerate(all_cls):
        cls_path = os.path.join(src_root,cls)
        v_allname = [x for x in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path,x))]
        v_allname.sort()
        len_data = [len(glob.glob(os.path.join(cls_path,x,'img_0*'))) for x in v_allname]
        split_idx = int(len(v_allname)*ratio)

        v_len_name = [[l,os.path.join(cls,x)] for l,x in zip(v_allname,len_data)]

        train.extend([[idx,l,x] for l,x in v_len_name[:split_idx]])
        test.extend([[idx,l,x] for l,x in v_len_name[split_idx:]])

    with open(os.path.join(out_root,'my_data_train_split_1_rawframes.txt'),'w') as f:
        for i,l,x in train:
            f.writelines(' '.join(map(str,[l,x,i]))+'\n')

    with open(os.path.join(out_root,'my_data_val_split_1_rawframes.txt'),'w') as f:
        for i,l,x in test:
            f.writelines(' '.join(map(str,[l,x,i]))+'\n')

    with open(os.path.join(out_root,'my_data_test_split_1_rawframes.txt'),'w') as f:
        for i,l,x in test:
            f.writelines(' '.join(map(str,[l,x,i]))+'\n')




if __name__ == '__main__':
    main()
