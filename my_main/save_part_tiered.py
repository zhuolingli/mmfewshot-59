# 在model 在trainset pretrain后，计算 trainingset 的 类别质心并保存。
# 用在 training set上泛化最好的用作计算？还是在val 上泛化最好的model 用于计算？

import os
import os.path as osp
import argparse

import torch
import mmcv
from functools import partial

from mmfewshot.classification.apis import init_classifier
from mmfewshot.classification.datasets import (build_dataset, build_dataloader,)
from mmfewshot.utils import compat_cfg
from mmfewshot.utils import multi_pipeline_collate_fn as collate
from torch.utils.data import DataLoader, Dataset

from mmfewshot.classification.datasets.tiered_imagenet import TRAIN_CLASSES, VAL_CLASSES, TEST_CLASSES

Tiered_ALL = TRAIN_CLASSES + VAL_CLASSES + TEST_CLASSES



import pickle as pkl

def parse_args():
    parser = argparse.ArgumentParser(description='save training set class centroids')
    # parser.add_argument('--config', default='configs/classification/my_work_9_24/mini_imagenet/save_class_centroid.py', help='train config file path')
    # parser.add_argument('--config', default='configs/classification/my_work_9_24/mini_imagenet/save.py', help='train config file path')
    parser.add_argument('--config', default='configs/classification/my_work_9_24/tiered_imagenet/save.py', help='train config file path')
    
    
    # parser.add_argument('--checkpoint', default='my_output/backbone/81_18.pth') # meta baseline 后的model权重
    # protoNet
    # parser.add_argument('--checkpoint', default='web_best_models/protonet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot_20211120_134319-73173bee.pth')
    # parser.add_argument('--checkpoint', default='web_best_models/protonet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot_20211120_153230-eb72884e.pth')
    parser.add_argument('--checkpoint', default='web_best_models/meta-baseline/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot_20211120_230843-6f3a6e7e.pth')
    
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main(output_file):
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    
    # build the model from a config file and a checkpoint file
    model = init_classifier(args.config, args.checkpoint, device=args.device)
    # dataset = build_dataset(cfg.data.train)
    dataset = build_dataset(cfg.data.train)
    
    train_dataloader_default_args = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=None,
        round_up=False,
        seed=cfg.get('seed'),
        pin_memory=False,
        use_infinite_sampler=cfg.use_infinite_sampler,
        shuffle=False,)
    
    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    
    data_loader = build_dataloader(dataset, **train_loader_cfg)
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    model.cuda()
    model.eval()
    feats_list, img_metas_list, gt_labels_list = [], [], []
    
    with open('data/tiered_imagenet/class_names.txt', 'r') as f:
        classnames= f.readlines()

    with open('data/tiered_imagenet/synsets.txt', 'r') as f:
        wnids = f.readlines()
    
    name2wnid_dict = {}
    count = 0
    for idx, (name, wnid) in enumerate(zip(classnames, wnids), 1):
        name = name.split('\n')[0].split(',')[0]
        wnid = wnid.split('\n')[0]
        # print(idx, name, wnid)
        count = count + 1
        if count != idx:
            print(idx, name, wnid)
        
        name2wnid_dict[name]= wnid
    # label2wnid_dict =  {label:name2wnid_dict[name[0].split(',')[0]] for label, name in enumerate(Tiered_ALL)}
    # with open('label2wnid.pkl', 'wb') as f:
    #     pkl.dump(label2wnid_dict, f,protocol=pkl.HIGHEST_PROTOCOL)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img_metas_list.extend(data['img_metas'].data[0])
            feats = model(img=data['img'].cuda(), mode='extract_feat')
            feats_list.append(feats.cpu())
            gt_labels_list.append(data['gt_label'].cpu())
            prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    feats = torch.cat(feats_list, dim=0)
    gt_labels = torch.cat(gt_labels_list, dim=0)
    label2name_dict =  {label:name for name, label in data_loader.dataset.class_to_idx.items()}

   

    wnid2Embidx = {name2wnid_dict[name.split(',')[0]]:[] for name in data_loader.dataset.CLASSES}
    img_metas_list = [] # 重建metas信息，保存每个样本对应的wnid值
    
    
    for idx, label in enumerate(gt_labels.tolist()):
        name =  label2name_dict[label].split(',')[0]  
        wnid = name2wnid_dict[name]
        img_metas_list.append(wnid)
        wnid2Embidx[wnid].append(idx)
    

    
    
    all_features = {}
    all_features['feats'] = feats
    all_features['gt_labels'] = gt_labels
    all_features['img_metas'] = img_metas_list 
    # with open(os.path.join(output_file, 'all_img_feats.pickle'), 'wb') as f:
    #     pkl.dump(all_features, f,protocol=pkl.HIGHEST_PROTOCOL)

    
    class_feature = {}
    for wnid, embidx in wnid2Embidx.items():
        emb = feats[embidx,:]
        mean = torch.mean(emb, dim=0).unsqueeze(dim=0)
        std = torch.std(emb, dim=0).unsqueeze(dim=0)
        class_feature[wnid] = {'mean':mean, 'std':std}
    # with open(os.path.join(output_file, 'class_centroid.pickle'), 'wb') as f:
    #     pkl.dump(class_feature, f,protocol=pkl.HIGHEST_PROTOCOL)
    
        
        

if __name__ == '__main__':    
    DATA_FILE = 'my_output/metabase_emb/tiered'
    
    # 记得要在 conifg save_class_centroid 文件中同步更改数据集
    # set = 'train'  
    set = 'test'
    output_file = os.path.join(DATA_FILE, set)
    main(output_file)
