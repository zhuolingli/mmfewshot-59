# # Copyright (c) OpenMMLab. All rights reserved.
# import warnings
# from typing import Dict, List

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcls.models.builder import HEADS
# from torch import Tensor
# from demo.demo_attention_rpn_detector_inference import main

# from mmfewshot.classification.datasets import label_wrapper
# from .base_head import BaseFewShotHead

# import torch.nn as nn
# import math
# import pickle
# import numpy as np
# import scipy.sparse as sp
# import torch
# import torch.nn.functional as F
# import os
# import random

# from mmcls.models.builder import build_loss

# # from model.encoder_decoder import encoder_template, decoder_template, weights_init

# class VAE(nn.Module):
#     def __init__(self, 
#                 semantic_emb_file: str = None,
#                 class_feature_file: str = None,
#                 kg_emb_file: str = None,
#                 in_dim: int = 640,
#                 semantic_dim: int = 1600,
#                 kg_entity_dim = 1000,
#                 hidden_size_rule: dict = {},
#                 latent_size: int = 128,
#                 loss: Dict = dict(type='MSELoss', loss_weight=1.0),
#                 **kwargs,
#                 ):
#         super(VAE, self).__init__()
        
#         modules = [nn.Linear(2*latent_size, latent_size), nn.ReLU(), nn.Linear(latent_size, latent_size)]
#         self.fuse_latent_vectors = nn.Sequential(*modules)
#         # built VAE, 在encoder_template 内部会对 linear model作初始化。
#         self.img_encoder = encoder_template(in_dim, latent_size, hidden_size_rule['img'])
#         self.semantic_encoder = encoder_template(semantic_dim, latent_size, hidden_size_rule['semantic'], 'semantic')
#         self.kg_encoder = encoder_template(kg_entity_dim, latent_size, hidden_size_rule['kg'])
        
#         self.img_decoder = decoder_template(latent_size, in_dim, hidden_size_rule['img'])
#         self.semantic_decoder = decoder_template(latent_size,semantic_dim, hidden_size_rule['semantic'])
#         self.kg_decoder = decoder_template( latent_size, kg_entity_dim, hidden_size_rule['kg'])
        
#         self.compute_loss = build_loss(loss) # 原文使用的是 L1 都试试
#         # semantic, kg, class_feature
#         self.wnid2data = self.load_sematic_data(class_feature_file,semantic_emb_file, kg_emb_file)

#     def load_vae_weight(self, weight_file):
#         ckpt = torch.load(weight_file, map_location='cpu')
        
#         self.img_encoder.load_state_dict(ckpt['encoder']['img'])
#         self.img_decoder.load_state_dict(ckpt['decoder']['img'])
        
#         self.semantic_encoder.load_state_dict(ckpt['encoder']['semantic'])
#         self.semantic_decoder.load_state_dict(ckpt['decoder']['semantic'])
        
#         self.kg_encoder.load_state_dict(ckpt['encoder']['kg'])
#         self.kg_decoder.load_state_dict(ckpt['decoder']['kg'])


#     def load_sematic_data(self, class_feature_file, semantic_file, kg_file, **kwargs):
#         kg = KG(kg_file)
#         with open(class_feature_file, 'rb') as handle:
#             class_features = pickle.load(handle)
#         with open(semantic_file, 'rb') as handle:
#             semantic_embs = pickle.load(handle)
        
#         wnid2data = {}
#         for wnid in semantic_embs.keys():
#             # try:
#             #     semantic_emb = semantic_embs[wnid].float() # 使用glove进行测试
#             #     # print("loaded glove embeddings!!")
#             # except:
#             semantic_emb = semantic_embs[wnid]['semantic_emb'].float() # 使用 clip进行测试
#                 # print("loaded clip embeddings!!")
                
#             kg_entity = kg.wnid_to_embedding.get(wnid, torch.zeros(1000))
#             class_feature = class_features.get(wnid, None)
            
#             wnid2data[wnid] = (semantic_emb, kg_entity, class_feature)
#         return wnid2data
    
#     def train_vae(self,):
#         pass
    
#     def forward(self, feats, wnid_list,  mode= 'boost_supports'):
#         if mode == 'train_head':
#             # get semantic embedding  and ground turth feature for each support sample  
#             semantic_emb_list = []
#             kg_emb_list = []
#             class_feature_list = []
#             for wnid in wnid_list:
#                 semantic_emb, kg_emb, class_feature = self.wnid2data[wnid]
#                 class_feature_mean = class_feature['mean']
#                 class_feature_std = class_feature['std']
#                 reparam_feature = self.reparameterize(class_feature_mean, class_feature_std)
                
#                 semantic_emb_list.append(semantic_emb.unsqueeze(0))
#                 kg_emb_list.append(kg_emb.reshape(1,-1))
#                 class_feature_list.append(reparam_feature)
            
#             device = feats[0].device
#             semantic_embs = torch.cat(semantic_emb_list, dim=0).to(device)
#             repara_class_features = torch.cat(class_feature_list, dim=0).to(device)
#             # kg_embs = torch.cat(kg_emb_list, dim=0).to(device)
            
#             Z_from_img, _ = self.img_encoder(feats)
#             Z_from_semantic, _, _ = self.semantic_encoder(semantic_embs)
            
#             Z = torch.cat((Z_from_img, Z_from_semantic), dim=1)
#             Z = self.fuse_latent_vectors(Z)
            
#             feats_from_Z = self.img_decoder(Z)
#             loss = self.compute_loss(feats_from_Z, repara_class_features)
#             return loss  
        
#         if mode == 'boost_supports':
#             Z_from_img, Z_from_semantic, img_from_semantic = self.boost_prototype(feats, wnid_list)
#             return Z_from_img, Z_from_semantic, img_from_semantic
#             assert False, 'mode:{} confict'.format(mode)

#     def boost_prototype(self,feats, wnid_list):
#         # get semantic embedding  and ground turth feature for each support sample  
#         semantic_emb_list = []
#         kg_emb_list = []
#         for wnid in wnid_list:
#             semantic_emb, kg_emb, _  = self.wnid2data[wnid]
#             semantic_emb_list.append(semantic_emb.unsqueeze(0))
#             kg_emb_list.append(kg_emb.reshape(1,-1))

#         device = feats[0].device
#         semantic_embs = torch.cat(semantic_emb_list, dim=0).to(device)
#         # kg_embs = torch.cat(kg_emb_list, dim=0).to(device)
            
#         Z_from_img, _ = self.img_encoder(feats)
#         if self.semantic_encoder.training:
#             Z_from_semantic, _, _ = self.semantic_encoder(semantic_embs)
#         else:
#             Z_from_semantic, _= self.semantic_encoder(semantic_embs)
            
#         Z = Z_from_semantic
#         # Z = torch.cat((Z_from_img, Z_from_semantic), dim=1)
#         # Z = self.fuse_latent_vectors(Z)
#         img_from_semantic = self.img_decoder(Z)
#         return Z_from_img, Z_from_semantic, img_from_semantic

#     def reparameterize(self, mu, var):
#         std = var
#         # eps = torch.randn_like(std) / 1 # 10
#         eps = 0 # 屏蔽噪声
#         return mu + eps*std

# @HEADS.register_module() 
# class VAEHead(BaseFewShotHead):
#     def __init__(self,
#                  vaecfg:Dict = None,
#                  temperature: float = 1.0,
#                  learnable_temperature: bool = True,
#                  *args,
#                  **kwargs) -> None:
#         super().__init__(cal_acc=True, *args, **kwargs)
#         self.VAE = VAE(**vaecfg)
#         if learnable_temperature:
#             self.temperature = nn.Parameter(torch.tensor(temperature))
#         else:
#             self.temperature = temperature

#         # used in meta testing
#         self.support_feats = []
#         # self.support_unwrapped_labels = [] # 真实标签
#         self.support_labels = []    # episodic 化的 label
#         self.support_wnids = []
#         # self.label_to_unwrapped_label = None    # task 包含的类别真实标签
#         self.mean_support_feats = None
#         self.class_ids = None
#         self.logits2logit = nn.Sequential(nn.Linear(4,1),nn.Sigmoid())

#     def forward_train_head(self, support_feats: Tensor, support_labels: Tensor, support_wnids:List, **kwargs):
#         class_ids, _ = torch.unique(support_labels).sort()
#         mean_support_feats = torch.cat([
#             support_feats[support_labels == class_id].mean(0, keepdim=True)
#             for class_id in class_ids
#         ], dim=0)
        
#         wnids = [support_wnids[torch.nonzero(support_labels==class_id)[0].item()] for class_id in class_ids]
#         loss = self.VAE(mean_support_feats, wnids, 'train_head')
#         return loss


#     def forward_train(self, support_feats: Tensor, support_labels: Tensor, support_wnids: List,
#                       query_feats: Tensor, query_labels: Tensor,
#                       **kwargs) -> Dict:
#         """Forward training data.
        
#         Args:
#             support_feats (Tensor): Features of support data with shape (N, C).
#             support_labels (Tensor): Labels of support data with shape (N).
#             query_feats (Tensor): Features of query data with shape (N, C).
#             query_labels (Tensor): Labels of query data with shape (N).

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         # 这里接收的标签 是 真实标签
#         class_ids, _ = torch.unique(support_labels).cpu().sort()
#         class_ids = class_ids.tolist()
#         wnids=  [support_wnids[torch.nonzero(support_labels==class_id)[0].item()] for class_id in class_ids]
#         assert len(wnids) == len(set(wnids))       
#         episodic_support_labels = label_wrapper(support_labels, class_ids)
        
#         mean_support_feats = torch.cat([
#             support_feats[support_labels == class_id].mean(0, keepdim=True)
#             for class_id in class_ids
#         ],dim=0)
        
        
#         Z_from_img, Z_from_semantic, img_from_semantic = self.VAE(mean_support_feats, wnids, 'boost_supports')    
#         Z_query, _ = self.VAE.img_encoder(query_feats)
        
#         cosine_distance_ori = torch.mm(F.normalize(query_feats), F.normalize(mean_support_feats).transpose(0,1))
#         cosine_distance_latent_img = torch.mm(F.normalize(Z_query), F.normalize(Z_from_img).transpose(0,1))
#         cosine_distance_latent_sem = torch.mm(F.normalize(Z_query), F.normalize(Z_from_semantic).transpose(0,1))
#         cosine_distance_recons_img = torch.mm(F.normalize(query_feats), F.normalize(img_from_semantic).transpose(0,1))
        
        
#         n_query = query_feats.shape[0]
#         n_way = cosine_distance_ori.shape[1]
        
#         logits = 0.5 * cosine_distance_ori + 0.25 * cosine_distance_latent_sem + 0.25 * cosine_distance_latent_img
    
        
#         # cosine_distance = torch.cat([cosine_distance_ori, cosine_distance_latent_img, cosine_distance_latent_sem, cosine_distance_recons_img], dim=1)
#         # cosine_distance = cosine_distance.reshape(n_query,-1, n_way).transpose(2,1)
        
#         # logits = self.logits2logit(cosine_distance.reshape(-1, 4)).reshape(n_query, n_way)
        
#         scores = logits * self.temperature
#         query_labels = label_wrapper(query_labels, class_ids)
#         losses = self.loss(scores, query_labels)
#         return losses
        
#         boosted_prototypes = self.VAE(mean_support_feats, wnids, 'boost_supports')    
        
#         prototypes = 0.25 * boosted_prototypes + 0.75 * mean_support_feats
#         cosine_distance = torch.mm(F.normalize(query_feats), F.normalize(prototypes).transpose(0,1))
        
#         scores = cosine_distance * self.temperature
#         query_labels = label_wrapper(query_labels, class_ids)
#         losses = self.loss(scores, query_labels)
#         return losses
        
#         cosine_distance = torch.mm(
#             F.normalize(query_feats),
#             F.normalize(mean_support_feats).transpose(0, 1))
        
        
#         n_way = len(class_ids)
#         scale = 10
        
#         support_labels_one_hot = F.one_hot(episodic_support_labels, n_way).float() # [25, 5]
#         assign_1 = F.softmax(cosine_distance * scale, dim=-1) # [75, 5]
#         assign_1 = torch.cat([support_labels_one_hot, assign_1], dim=0)  # [100,5 ]
#         assign_1_transposed = assign_1.transpose(0, 1) # [5,100]
#         emb = torch.cat([support_feats, query_feats], dim=0)
#         mean_1 = torch.mm(assign_1_transposed, emb)  #[5,100]*[100, 512]
#         mean_1 = mean_1.div(assign_1_transposed.sum(dim=1, keepdim=True).expand_as(mean_1))
        
#         diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_1.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)
#         std_1 = (assign_1_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_1_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)
        
#         logits = torch.mm(F.normalize(query_feats), F.normalize(boosted_prototypes).transpose(0,1))
#         # logits = torch.nn.functional.cosine_similarity(query_feats.unsqueeze(1).expand(-1, boosted_prototypes.shape[0], -1),
#                                                     # boosted_prototypes.unsqueeze(0).expand(query_feats.shape[0], -1, -1), dim=-1)
#         assign_2 = F.softmax(logits * scale, dim=-1)
#         assign_2 = torch.cat([support_labels_one_hot, assign_2], dim=0)
#         assign_2_transposed = assign_2.transpose(0, 1)
#         emb = torch.cat([support_feats, query_feats], dim=0)
#         mean_2 = torch.mm(assign_2_transposed, emb)
#         mean_2 = mean_2.div(assign_2_transposed.sum(dim=1, keepdim=True).expand_as(mean_2))
#         diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_2.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)        
#         std_2 = (assign_2_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_2_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)

#         prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
#         cosine_distance = torch.mm(F.normalize(query_feats), F.normalize(prototypes).transpose(0,1))

#         scores = cosine_distance * self.temperature

#         query_labels = label_wrapper(query_labels, class_ids)
#         losses = self.loss(scores, query_labels)
#         return losses

#     def forward_support(self, x: Tensor, gt_label: Tensor,support_wnids:List, **kwargs) -> None:
#         """Forward support data in meta testing."""
#         self.support_feats.append(x)
#         self.support_labels.append(gt_label)
#         self.support_wnids.extend(support_wnids) # support_wnids 为 list 类型。

#     def forward_query(self, query_feats: Tensor,**kwargs) -> List:
#         """Forward query data in meta testing."""

#         support_labels = torch.cat(self.support_labels,dim=0)
#         support_feats = torch.cat(self.support_feats, dim=0)
        
#         wnids=  [self.support_wnids[torch.nonzero(support_labels==class_id)[0].item()] for class_id in self.class_ids]
        
#         Z_from_img, Z_from_semantic, img_from_semantic = self.VAE(self.mean_support_feats, wnids, 'boost_supports')    
#         Z_query, _ = self.VAE.img_encoder(query_feats)
        
#         cosine_distance_ori = torch.mm(F.normalize(query_feats), F.normalize(self.mean_support_feats).transpose(0,1))
#         cosine_distance_latent_img = torch.mm(F.normalize(Z_query), F.normalize(Z_from_img).transpose(0,1))
#         cosine_distance_latent_sem = torch.mm(F.normalize(Z_query), F.normalize(Z_from_semantic).transpose(0,1))
#         cosine_distance_recons_img = torch.mm(F.normalize(query_feats), F.normalize(img_from_semantic).transpose(0,1))
#         n_query = query_feats.shape[0]
#         n_way = cosine_distance_ori.shape[1]
#         cosine_distance = torch.cat([cosine_distance_ori, cosine_distance_latent_img, cosine_distance_latent_sem, cosine_distance_recons_img], dim=1)
#         cosine_distance = cosine_distance.reshape(n_query,-1, n_way).transpose(2,1)
        
#         factor = [ 1.0, 0., 0., 0.0]       
#         logits = factor[0] * cosine_distance_ori + factor[1] * cosine_distance_latent_img + factor[2] * cosine_distance_latent_sem +  factor[3] * cosine_distance_recons_img

#         # logits = self.logits2logit(cosine_distance.reshape(-1, 4)).reshape(n_query, n_way)
        
#         # scores = logits * self.temperature
#         scores = logits
#         pred = F.softmax(scores, dim=1)
#         pred = list(pred.detach().cpu().numpy())
#         return pred
        
#         # transductive
#         cosine_distance = torch.mm(
#         F.normalize(query_feats),
#         F.normalize(self.mean_support_feats).transpose(0, 1))
#         n_way = self.class_ids.shape[0]
#         scale = 10
        
#         support_labels_one_hot = F.one_hot(support_labels, n_way).float() # [25, 5]
#         assign_1 = F.softmax(cosine_distance * scale, dim=-1) # [75, 5]
#         assign_1 = torch.cat([support_labels_one_hot, assign_1], dim=0)  # [100,5 ]
#         assign_1_transposed = assign_1.transpose(0, 1) # [5,100]
#         emb = torch.cat([support_feats, query_feats], dim=0)
#         mean_1 = torch.mm(assign_1_transposed, emb)  #[5,100]*[100, 512]
#         mean_1 = mean_1.div(assign_1_transposed.sum(dim=1, keepdim=True).expand_as(mean_1))
        
#         diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_1.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)
#         std_1 = (assign_1_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_1_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)
        
#         logits = torch.mm(F.normalize(query_feats), F.normalize(boosted_prototypes).transpose(0,1))
       
#         assign_2 = F.softmax(logits * scale, dim=-1)
#         assign_2 = torch.cat([support_labels_one_hot, assign_2], dim=0)
#         assign_2_transposed = assign_2.transpose(0, 1)
#         emb = torch.cat([support_feats, query_feats], dim=0)
#         mean_2 = torch.mm(assign_2_transposed, emb)
#         mean_2 = mean_2.div(assign_2_transposed.sum(dim=1, keepdim=True).expand_as(mean_2))
#         diff = torch.pow(emb.unsqueeze(0).expand(n_way, -1, -1) - mean_2.unsqueeze(1).expand(-1, emb.shape[0], -1), 2)        
#         std_2 = (assign_2_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=1) / assign_2_transposed.unsqueeze(-1).expand_as(diff).sum(dim=1)

#         prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
#         cosine_distance = torch.mm(F.normalize(query_feats), F.normalize(prototypes).transpose(0,1))
       
#         scores = cosine_distance * self.temperature
#         pred = F.softmax(scores, dim=1)
#         pred = list(pred.detach().cpu().numpy())
#         return pred

#     def before_forward_support(self) -> None:
#         """Used in meta testing.

#         This function will be called before model forward support data during
#         meta testing.
#         """
#         # reset saved features for testing new task
#         self.support_feats.clear()
#         self.support_labels.clear()
#         self.support_wnids.clear()
#         self.class_ids = None
#         self.mean_support_feats = None


#     def before_forward_query(self) -> None:
#         """Used in meta testing.

#         This function will be called before model forward query data during
#         meta testing.
#         """
#         support_feats = torch.cat(self.support_feats, dim=0)
#         support_labels = torch.cat(self.support_labels, dim=0)
#         self.class_ids, _ = torch.unique(support_labels).sort()
#         self.mean_support_feats = torch.cat([
#             support_feats[support_labels == class_id].mean(0, keepdim=True)
#             for class_id in self.class_ids
#         ],dim=0)

#         if max(self.class_ids) + 1 != len(self.class_ids):
#             warnings.warn(f'the max class id is {max(self.class_ids)}, while '
#                           f'the number of different number of classes is '
#                           f'{len(self.class_ids)}, it will cause label '
#                           f'mismatch problem.')





