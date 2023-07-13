_base_ = [
    # '../../_base_/meta_test/mini-imagenet_meta-test_5way-1shot.py',
    '../../_base_/meta_test/mini-imagenet_meta-test_5way-5shot.py',
    '../../_base_/runtime/iter_based_runtime.py',
    '../../_base_/schedules/sgd_100k_iter.py'
]

runner = dict(type='IterBasedRunner', max_iters=200000)
optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.0005) # 0.01 0.9 0.0001
evaluation = dict(by_epoch=False, interval=500, metric='accuracy')

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
        num_ways=10, 
        num_shots=5,
        num_queries=15, # MSE loss， 不需要引入query，将 原型作为标签。
        dataset=dict(
            type='MiniImageNetDataset',
            data_prefix='data/mini_imagenet',
            subset='train',
            pipeline=train_pipeline)),
    test=dict(meta_test_cfg=dict(fast_test=True)), # 
    val=dict(meta_test_cfg=dict(fast_test=True)),#  模型验证采用一次性计算全部验证样本的emb
    )

model = dict(
    type='MyClassifier',
    backbone=dict(type='ResNet12'),
    head=dict(type='VAEHead',
        vaecfg=dict(
            class_feature_file='my_output/save_embeddings_10_21/train/class_centroid.pickle',
            semantic_emb_file='semantic_embeddings/clip_semantic_emb_name.pkl',
            # semantic_emb_file='semantic_embeddings/GPT2-768-raw.pkl',
            # semantic_emb_file='semantic_embeddings/GPT2-xl-1600-raw.pkl',
            kg_emb_file='semantic_embeddings/KG',
            in_dim=640,
            semantic_dim=512,
            # semantic_dim=1600, # 语义嵌入是定死的 512 维度
            kg_entity_dim=1000,
            hidden_size_rule=dict(img=[1600,1600],
                                  semantic=[1200,1200], # [400, 400]
                                  kg=[800,600]),
            latent_size=160,
            loss=dict(type='MSELoss', loss_weight=1.0),
            ),
        ),
    )
load_from = ('my_output/backbone/81_18.pth')



