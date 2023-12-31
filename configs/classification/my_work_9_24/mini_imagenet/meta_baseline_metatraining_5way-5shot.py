_base_ = [
    '../../_base_/meta_test/mini-imagenet_meta-test_5way-5shot.py',
    '../../_base_/runtime/iter_based_runtime.py',
    '../../_base_/schedules/sgd_100k_iter.py'
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='EpisodicDataset',
        num_episodes=100000,
        num_ways=10, # 为啥原始设置为 10 way， query为 5？
        num_shots=5,
        num_queries=5,
        dataset=dict(
            type='MiniImageNetDataset',
            data_prefix='data/mini_imagenet',
            subset='train',
            pipeline=train_pipeline)),
    test=dict(meta_test_cfg=dict(fast_test=True)))

model = dict(
    type='MetaBaseline',
    backbone=dict(type='ResNet12_512'),
    head=dict(type='MetaBaselineHead'))
load_from = ('./my_output/pretrain_resnet12_1xb100_mini-imagenet_5way-5shot/best_accuracy_mean.pth')
