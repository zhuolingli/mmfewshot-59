# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from torch import Tensor

from mmfewshot.classification.datasets import label_wrapper
from .base_head import BaseFewShotHead



import torch.nn as nn
import math
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import random

from mmcls.models.builder import build_loss

class ProtoComNet(nn.Module):
    def __init__(self, 
                clip_file: str = None,
                in_dim: int = 512,
                clip_semantic_dim: int = 512,
                out_dim: int = 512,
                class_feature_file: str = None,
                loss: Dict = dict(type='MSELoss', loss_weight=1.0),
                clip_mode: str = None,
                **kwargs,
                ):
        super(ProtoComNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim+clip_semantic_dim, out_features=(in_dim+clip_semantic_dim)//4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=(in_dim+clip_semantic_dim)//4, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_dim)
        )
        self.compute_loss = build_loss(loss)
        if clip_file is not None:
            try:
                with open(clip_file, 'rb') as handle:
                    self.clip_semantic_embs = pickle.load(handle)
            except:
                print('failed to load clip emb from {}'.format(clip_file))
        if class_feature_file is not None:
            try:
                with open(class_feature_file, 'rb') as handle:
                    self.class_feature = pickle.load(handle)
            except:
                print('failed to load clip emb from {}'.format(clip_file))
    def forward(self, feats, wnid_list,  mode= 'boost_supports'):
        if mode == 'train_ComNet':
            # get clip semantic embedding  and ground turth feature for each support sample  
            clip_emb_list = []
            class_feature_list = []
            for wnid in wnid_list:
                clip_emb = self.clip_semantic_embs[wnid]['semantic_emb']
                class_feature_mean = self.class_feature[wnid]['mean']
                class_feature_std = self.class_feature[wnid]['std']
                reparam_feature = self.reparameterize(class_feature_mean, class_feature_std)
                
                clip_emb_list.append(clip_emb.reshape(1,-1))
                class_feature_list.append(reparam_feature)
            device = feats[0].device
            clip_embs = torch.cat(clip_emb_list, dim=0).to(device)
            repara_class_features = torch.cat(class_feature_list, dim=0).to(device)
            
            feats_and_semantic = torch.cat((feats, clip_embs), dim=1 )
            
            z = self.encoder(feats_and_semantic)
            out = self.decoder(z)
            boosted_feats = out 
            loss = self.compute_loss(boosted_feats, repara_class_features)
            return loss  
        if mode == 'boost_supports':
            clip_emb_list = []
            for wnid in wnid_list:
                clip_emb = self.clip_semantic_embs[wnid]['semantic_emb']
                clip_emb_list.append(clip_emb.reshape(1,-1))
            device = feats[0].device
            clip_embs = torch.cat(clip_emb_list, dim=0).to(device)
            feats_and_semantic = torch.cat((feats, clip_embs), dim=1 )
            
            z = self.encoder(feats_and_semantic)
            out = self.decoder(z)
            boosted_feats = out
            return boosted_feats
        else:
            assert False, 'mode:{} confict'.format(mode)

    def reparameterize(self, mu, var):
        std = var
        eps = torch.randn_like(std) / 10
        return mu + eps*std

@HEADS.register_module() 
class ClipHead(BaseFewShotHead):
    """Classification head with clip semantic embedding
    Args:
        temperature (float): Scaling factor of `cls_score`. Default: 10.0.
        learnable_temperature (bool): Whether to use learnable scale factor
            or not. Default: True.
    """
    def __init__(self,
                 comNet:Dict = None,
                 temperature: float = 1.0,
                 learnable_temperature: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(cal_acc=True, *args, **kwargs)
        self.ComNet = ProtoComNet(**comNet)
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

        # used in meta testing
        self.support_feats = []
        # self.support_unwrapped_labels = [] # 真实标签
        self.support_labels = []    # episodic 化的 label
        self.support_wnids = []
        # self.label_to_unwrapped_label = None    # task 包含的类别真实标签
        self.mean_support_feats = None
        self.class_ids = None
        
    def forward_train_ComNet(self, support_feats: Tensor, support_labels: Tensor, support_wnids:List, **kwargs):
        class_ids, _ = torch.unique(support_labels).sort()
        mean_support_feats = torch.cat([
            support_feats[support_labels == class_id].mean(0, keepdim=True)
            for class_id in class_ids
        ], dim=0)
        
        wnids = [support_wnids[torch.nonzero(support_labels==class_id)[0].item()] for class_id in class_ids]
        loss = self.ComNet(mean_support_feats, wnids, 'train_ComNet')
        return loss


    def forward_train(self, support_feats: Tensor, support_labels: Tensor, support_wnids: List,
                      query_feats: Tensor, query_labels: Tensor,
                      **kwargs) -> Dict:
        """Forward training data.
        
        Args:
            support_feats (Tensor): Features of support data with shape (N, C).
            support_labels (Tensor): Labels of support data with shape (N).
            query_feats (Tensor): Features of query data with shape (N, C).
            query_labels (Tensor): Labels of query data with shape (N).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 这里接收的标签 是 真实标签
        class_ids, _ = torch.unique(support_labels).cpu().sort()
        class_ids = class_ids.tolist()
        wnids=  [support_wnids[torch.nonzero(support_labels==class_id)[0].item()] for class_id in class_ids]
        assert len(wnids) == len(set(wnids))       
        episodic_support_labels = label_wrapper(support_labels, class_ids)
        
        mean_support_feats = torch.cat([
            support_feats[support_labels == class_id].mean(0, keepdim=True)
            for class_id in class_ids
        ],dim=0)
        cosine_distance = torch.mm(
            F.normalize(query_feats),
            F.normalize(mean_support_feats).transpose(0, 1))
        
        boosted_prototypes = self.ComNet(mean_support_feats, wnids, 'boost_supports')    
        
        n_way = len(class_ids)
        scale = 10
        
        support_labels_one_hot = F.one_hot(episodic_support_labels, n_way).float() # [25, 5]
        assign_1 = F.softmax(cosine_distance * scale, dim=-1) # [75, 5]
        assign_1 = torch.cat([support_labels_one_hot, assign_1], dim=0)  # [100,5 ]
        assign_1_transposed = assign_1.transpose(0, 1) # [5,100]
        emb = torch.cat([support_feats, query_feats], dim=0)
        mean_1 = torch.mm(assign_1_transposed, emb)  #[5,100]*[100, 512]
        mean_1 = mean_1.div(assign_1_transposed.sum(dim=1, keepdim=True).expand_as(mean_1))
        
        diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_1.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)
        std_1 = (assign_1_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_1_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)
        
        logits = torch.mm(F.normalize(query_feats), F.normalize(boosted_prototypes).transpose(0,1))
        # logits = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(1).expand(-1, boosted_prototypes.shape[0], -1),
                                                    # boosted_prototypes.unsqueeze(0).expand(query_feats.shape[0], -1, -1), dim=-1)
        assign_2 = F.softmax(logits * scale, dim=-1)
        assign_2 = torch.cat([support_labels_one_hot, assign_2], dim=0)
        assign_2_transposed = assign_2.transpose(0, 1)
        emb = torch.cat([support_feats, query_feats], dim=0)
        mean_2 = torch.mm(assign_2_transposed, emb)
        mean_2 = mean_2.div(assign_2_transposed.sum(dim=1, keepdim=True).expand_as(mean_2))
        diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_2.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)        
        std_2 = (assign_2_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_2_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)

        prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
        cosine_distance = torch.mm(F.normalize(query_feats), F.normalize(prototypes).transpose(0,1))
       
        scores = cosine_distance * self.temperature

        query_labels = label_wrapper(query_labels, class_ids)
        losses = self.loss(scores, query_labels)
        return losses

    def forward_support(self, x: Tensor, gt_label: Tensor,support_wnids:List, **kwargs) -> None:
        """Forward support data in meta testing."""
        self.support_feats.append(x)
        self.support_labels.append(gt_label)
        self.support_wnids.extend(support_wnids) # support_wnids 为 list 类型。

    def forward_query(self, query_feats: Tensor,**kwargs) -> List:
        """Forward query data in meta testing."""
        cosine_distance = torch.mm(
            F.normalize(query_feats),
            F.normalize(self.mean_support_feats).transpose(0, 1))
        support_labels = torch.cat(self.support_labels,dim=0)
        support_feats = torch.cat(self.support_feats, dim=0)
        
        wnids=  [self.support_wnids[torch.nonzero(support_labels==class_id)[0].item()] for class_id in self.class_ids]
        
        boosted_prototypes = self.ComNet(self.mean_support_feats, wnids, 'boost_supports')    
        
        n_way = self.class_ids.shape[0]
        scale = 10
        
        support_labels_one_hot = F.one_hot(support_labels, n_way).float() # [25, 5]
        assign_1 = F.softmax(cosine_distance * scale, dim=-1) # [75, 5]
        assign_1 = torch.cat([support_labels_one_hot, assign_1], dim=0)  # [100,5 ]
        assign_1_transposed = assign_1.transpose(0, 1) # [5,100]
        emb = torch.cat([support_feats, query_feats], dim=0)
        mean_1 = torch.mm(assign_1_transposed, emb)  #[5,100]*[100, 512]
        mean_1 = mean_1.div(assign_1_transposed.sum(dim=1, keepdim=True).expand_as(mean_1))
        
        diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_1.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)
        std_1 = (assign_1_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_1_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)
        
        logits = torch.mm(F.normalize(query_feats), F.normalize(boosted_prototypes).transpose(0,1))
       
        assign_2 = F.softmax(logits * scale, dim=-1)
        assign_2 = torch.cat([support_labels_one_hot, assign_2], dim=0)
        assign_2_transposed = assign_2.transpose(0, 1)
        emb = torch.cat([support_feats, query_feats], dim=0)
        mean_2 = torch.mm(assign_2_transposed, emb)
        mean_2 = mean_2.div(assign_2_transposed.sum(dim=1, keepdim=True).expand_as(mean_2))
        diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_2.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)        
        std_2 = (assign_2_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_2_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)

        prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
        cosine_distance = torch.mm(F.normalize(query_feats), F.normalize(prototypes).transpose(0,1))
       
        scores = cosine_distance * self.temperature
        pred = F.softmax(scores, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self) -> None:
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        # reset saved features for testing new task
        self.support_feats.clear()
        self.support_labels.clear()
        self.support_wnids.clear()
        self.class_ids = None
        self.mean_support_feats = None


    def before_forward_query(self) -> None:
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        support_feats = torch.cat(self.support_feats, dim=0)
        support_labels = torch.cat(self.support_labels, dim=0)
        self.class_ids, _ = torch.unique(support_labels).sort()
        self.mean_support_feats = torch.cat([
            support_feats[support_labels == class_id].mean(0, keepdim=True)
            for class_id in self.class_ids
        ],dim=0)

        if max(self.class_ids) + 1 != len(self.class_ids):
            warnings.warn(f'the max class id is {max(self.class_ids)}, while '
                          f'the number of different number of classes is '
                          f'{len(self.class_ids)}, it will cause label '
                          f'mismatch problem.')



