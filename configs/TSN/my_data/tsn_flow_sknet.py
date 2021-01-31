# model settings
model = dict(
    type='TSN2D',
    modality='Flow',
    in_channels=10,
    backbone=dict(
        type='SKnet101',
        pretrained='pretrain_model/sknet101.pth',
        bn_eval=True,
        partial_bn=True),
    spatial_temporal_module=dict(
        type='SimpleSpatialModule',
        spatial_type='avg',
        spatial_size=7),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.3,
        in_channels=2048,
        num_classes=10))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDataset'
data_root = 'data/my_data/rawframes'
img_norm_cfg = dict(
    mean=[116.3], std=[1], to_rgb=False)
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='data/my_data/my_data_train_split_1_rawframes.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        num_segments=3,
        new_length=5,
        new_step=1,
        random_shift=True,
        modality='Flow',
        image_tmpl='flow_{}_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0.5,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=True,
        more_fix_crop=False,
        multiscale_crop=True,
        scales=[1, 0.875, 0.75, 0.66],
        max_distort=1,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file='data/my_data/my_data_val_split_1_rawframes.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        num_segments=3,
        new_length=5,
        new_step=1,
        random_shift=False,
        modality='Flow',
        image_tmpl='flow_{}_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=False),
    test=dict(
        type=dataset_type,
        ann_file='data/my_data/my_data_val_split_1_rawframes.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        num_segments=25,
        new_length=5,
        new_step=1,
        random_shift=False,
        modality='Flow',
        image_tmpl='flow_{}_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='ten_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[190, 300])
checkpoint_config = dict(interval=1)
# workflow = [('train', 5), ('val', 1)]
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 340
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/mydata/tsn_2d_flow_sknet_seg_3_f1s1_b32_g8_lr_0.005'
load_from = 'work_dirs/tsn_2d_flow_sknet_seg_3_f1s1_b32_g8_lr_0.005_std1/latest.pth'
resume_from = None



