_base_ = [
    '../../_base_/meta_test/mini-imagenet_meta-test_5way-5shot.py',
    '../../_base_/runtime/iter_based_runtime.py',
    '../../_base_/schedules/sgd_100k_iter.py'
]

runner = dict(type='IterBasedRunner', max_iters=100000)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
evaluation = dict(by_epoch=False, interval=1000, metric='accuracy')
num_val_episodes = 300



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
    workers_per_gpu=8, # 8
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
    test=dict(meta_test_cfg=dict(fast_test=True)),
    val=dict(num_episodes=num_val_episodes,
             meta_test_cfg=dict(num_episodes=num_val_episodes)),
)

model = dict(
    type='MetaBaseline',
    backbone=dict(type='ResNet12'),
    head=dict(type='MetaBaselineHead',
              cal_acc=True))
# load_from = ('./work_dirs/baseline_resnet12_1xb64_mini-imagenet_5way-5shot/best_accuracy_mean.pth')
# openmmlab布的最佳权重
load_from = ('web_best_models/baseline/baseline_resnet12_1xb64_mini-imagenet_5way-1shot_20211120_095923-26863e78.pth') 
# load_from = ('my_output/pretrain/best_accuracy_mean.pth') # 自己训练的权重
