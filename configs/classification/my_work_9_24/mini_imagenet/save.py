# 12/5 号开始使用这个。

_base_ = [
    '../../_base_/meta_test/mini-imagenet_meta-test_5way-5shot.py',
    '../../_base_/runtime/epoch_based_runtime.py',
    '../../_base_/schedules/sgd_200epoch.py'
]

img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = {{_base_.test_pipeline}}

# ban infinite sampler 
use_infinite_sampler = False

test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=16,
    train=dict(
        # _delete_ = True,
        type='MiniImageNetDataset',
        data_prefix='data/mini_imagenet',
        subset='test',
        pipeline=test_pipeline),
    )


# model = dict(
#     type='MetaBaseline',
#     backbone=dict(type='ResNet12'),
#     head=dict(type='MetaBaselineHead',cal_acc=True))


model = dict(
    type='ProtoNet',
    backbone=dict(type='ResNet12'),
    head=dict(type='PrototypeHead'))
