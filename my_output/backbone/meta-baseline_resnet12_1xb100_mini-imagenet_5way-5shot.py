img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(96, -1)),
    dict(type='CenterCrop', crop_size=84),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
num_ways = 5
num_shots = 5
num_queries = 15
num_val_episodes = 100
num_test_episodes = 2000
data = dict(
    val=dict(
        type='MetaTestDataset',
        num_episodes=100,
        num_ways=5,
        num_shots=5,
        num_queries=15,
        dataset=dict(
            type='MiniImageNetDataset',
            subset='val',
            data_prefix='data/mini_imagenet',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', size=(96, -1)),
                dict(type='CenterCrop', crop_size=84),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        meta_test_cfg=dict(
            num_episodes=100,
            num_ways=5,
            fast_test=False,
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(batch_size=25, num_workers=0),
            query=dict(batch_size=75, num_workers=0))),
    test=dict(
        type='MetaTestDataset',
        num_episodes=2000,
        num_ways=5,
        num_shots=5,
        num_queries=15,
        episodes_seed=0,
        dataset=dict(
            type='MiniImageNetDataset',
            subset='test',
            data_prefix='data/mini_imagenet',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', size=(96, -1)),
                dict(type='CenterCrop', crop_size=84),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        meta_test_cfg=dict(
            num_episodes=2000,
            num_ways=5,
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(batch_size=25, num_workers=0),
            query=dict(batch_size=75, num_workers=0))),
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='EpisodicDataset',
        num_episodes=100000,
        num_ways=10,
        num_shots=5,
        num_queries=5,
        dataset=dict(
            type='MiniImageNetDataset',
            data_prefix='data/mini_imagenet',
            subset='train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='RandomResizedCrop', size=84),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='ColorJitter',
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='ToTensor', keys=['gt_label']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ])))
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=4000)
evaluation = dict(by_epoch=False, interval=2000, metric='accuracy')
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'web_best_models/baseline/baseline_resnet12_1xb64_mini-imagenet_5way-1shot_20211120_095923-26863e78.pth'
resume_from = None
workflow = [('train', 1)]
pin_memory = True
use_infinite_sampler = True
seed = 0
runner = dict(type='IterBasedRunner', max_iters=100000)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.25,
    step=[20000, 40000])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=84),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
model = dict(
    type='MetaBaseline',
    backbone=dict(type='ResNet12'),
    head=dict(type='MetaBaselineHead', cal_acc=True))
work_dir = 'my_output/metapretraining'
gpu_ids = range(0, 4)
