_base_ = [
    '../../_base_/meta_test/tiered-imagenet_meta-test_5way-5shot.py',
    '../../_base_/runtime/iter_based_runtime.py',
    '../../_base_/schedules/sgd_100k_iter.py'
]
runner = dict(type='IterBasedRunner', max_iters=100000)
optimPizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
evaluation = dict(by_epoch=False, interval=1000, metric='accuracy')
num_val_episodes = 300

img_size = 84
# img_norm_cfg = dict(
#     mean=[120.39586422,115.59361427, 104.54012653]
# , std=[70.68188272,   68.27635443,  72.54505529]
# , to_rgb=True)
img_norm_cfg = dict(
mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadImageFromBytes'),
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
        num_ways=10,
        num_shots=5,
        num_queries=5,
        dataset=dict(
            type='TieredImageNetDataset',  # 原始代码里这列误写为 mini
            data_prefix='data/tiered_imagenet',
            subset='train',
            pipeline=train_pipeline)))

model = dict(
    type='MetaBaseline',
    backbone=dict(type='ResNet12'),
    head=dict(type='MetaBaselineHead'))
load_from = ('work_dirs/baseline_resnet12_1xb64_tiered-imagenet_5way-5shot/best_accuracy_mean.pth')
# load_from = ('web_best_models/meta-baseline/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot_20211120_230843-6f3a6e7e.pth')

# load_from= ('web_best_models/baseline/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot_20211120_095923-17c5cf85.pth')