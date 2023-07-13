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

import pickle as pkl

def parse_args():
    parser = argparse.ArgumentParser(description='save training set class centroids')
  
    parser.add_argument('--config', default='configs/classification/my_work_9_24/cub/save.py', help='train config file path')
    
    # protoNet
    parser.add_argument('--checkpoint', default='web_best_models/protonet/proto-net_resnet12_1xb105_cub_5way-1shot_20211120_101211-da5bfb99.pth')
    
    # metabaseline
    # parser.add_argument('--checkpoint', default='web_best_models/meta-baseline/meta-baseline_resnet12_1xb100_cub_5way-1shot_20211120_191622-8978c781.pth')
    
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
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            metas= [int(i['filename'].split('/')[3].split('.')[0]) for i in data['img_metas'].data[0]]
            img_metas_list.extend(metas)
            feats = model(img=data['img'].cuda(), mode='extract_feat')
            feats_list.append(feats.cpu())
            gt_labels_list.append(data['gt_label'].cpu())
            prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    feats = torch.cat(feats_list, dim=0)
    gt_labels = torch.cat(gt_labels_list, dim=0)
    
    wnids = [ int(item.split('.')[0]) for item in data_loader.dataset.CLASSES] # 数据集中类名对应的序号
    wnid2Embidx = {wnid:[] for wnid in wnids}
    
    for idx, img_metas in enumerate(img_metas_list):
        wnid = img_metas
        wnid2Embidx[wnid].append(idx)
    
    
    all_features = {}
    all_features['feats'] = feats
    all_features['gt_labels'] = gt_labels
    all_features['img_metas'] = img_metas_list 
    
    if not osp.exists(output_file):
        os.makedirs(output_file)
        print('make dir:{}'.format(output_file))
    
    with open(os.path.join(output_file, 'all_img_feats.pickle'), 'wb') as f:
        pkl.dump(all_features, f,protocol=pkl.HIGHEST_PROTOCOL)
    
    class_feature = {}
    for wnid, embidx in wnid2Embidx.items():
        emb = feats[embidx,:]
        mean = torch.mean(emb, dim=0).unsqueeze(dim=0)
        std = torch.std(emb, dim=0).unsqueeze(dim=0)
        class_feature[wnid] = {'mean':mean, 'std':std}
    
    class_feat_pth = os.path.join(output_file, 'class_centroid.pickle')
 
    with open(class_feat_pth, 'wb') as f:
        pkl.dump(class_feature, f,protocol=pkl.HIGHEST_PROTOCOL)
    
        
        

if __name__ == '__main__':    
    DATA_FILE = 'my_output/proto_emb/cub/'
    
    # 记得要在 conifg save 文件中同步更改数据集 !!!!
   
   
    # set = 'train'  
    set = 'test'
    # set = 'val'

    output_file = os.path.join(DATA_FILE, set)
    main(output_file)
