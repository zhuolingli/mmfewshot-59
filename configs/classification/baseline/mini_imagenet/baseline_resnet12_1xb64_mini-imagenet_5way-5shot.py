_base_ = [
    '../../_base_/meta_test/mini-imagenet_meta-test_5way-5shot.py',
    '../../_base_/runtime/epoch_based_runtime.py',
    '../../_base_/schedules/sgd_200epoch.py'
]

img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=img_size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

meta_finetune_cfg = dict(
    num_steps=600,
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001))

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='MiniImageNetDataset',
        data_prefix='data/mini_imagenet',
        subset='train',
        pipeline=train_pipeline),
    val=dict(
        meta_test_cfg=dict(
            fast_test=True,
            support=dict(
                batch_size=4, drop_last=True, train=meta_finetune_cfg))),
    test=dict(
        meta_test_cfg=dict(
            fast_test=True,
            support=dict(
                batch_size=4, drop_last=False, train=meta_finetune_cfg)))) # 原本是 drop=True 的

model = dict(
    type='Baseline',
    backbone=dict(type='ResNet12'), 
    head=dict(type='LinearHead', num_classes=64, in_channels=640),  
    # 对于finetune方法，meta test 阶段要新建一个分类头，所以要传入cfg
    meta_test_head=dict(type='LinearHead', num_classes=5, in_channels=640)) 






evaluation = dict(by_epoch=True, metric='accuracy', interval=1)
