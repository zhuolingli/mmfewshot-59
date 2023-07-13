import torch.nn as nn
import math
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import random

class ProtoComNet(nn.Module):
    def __init__(self, cfg):
        super(ProtoComNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=cfg.in_dim + cfg.semantic_dim, out_features=(cfg.in_dim + cfg.semantic_dim)//2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features= (cfg.in_dim + cfg.semantic_dim)//2, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=cfg.out_dim)
        )
        self.aggregator = nn.Sequential(
            nn.Linear(in_features=cfg.semantic_dim+cfg.indim, out_features=300),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features=1)
        )
        try:
            with open('my_output/save_class_feature_centroid/class_feautre.pickle', 'rb') as handle:
                self.class_feature = pickle.load(handle)
        except:
            print('no found ' + 'my_output/save_class_feature_centroid/class_feautre.pickle')
        
        try:
            with open('/run/media/cv/d/lzl/mmd/python文件处理脚本/label_ocr_recog/0527/train/clip/clip_semantic_emb.pkl', 'rb') as handle:
                self.clip_semantic = pickle.load(handle)
        except:
            print('no found ' + '/run/media/cv/d/lzl/mmd/python文件处理脚本/label_ocr_recog/0527/train/clip/clip_semantic_emb.pkl')
        
        
    def forward(self, x, y, use_scale=False, is_infer=False):
        if is_infer == False:
            nb = x.shape[0]
            outputs = []
            targets = []
            for i in range(nb):
                input_feature = torch.zeros(self.n, self.in_dim).cuda()
                for k, v in self.metapart_feature.items():
                    input_feature[k:k+1, :] = self.reparameterize(v['mean'], v['std'])
                input_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                   self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] = x[i:i+1, :]

                semantic_feature = self.semantic_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                               self.part_prior['wnids2id'][self.label2catname[y[i].item()]]+1, :, :]

                semantic_feature = torch.cat([semantic_feature, x[i:i+1, :].unsqueeze(0).expand(-1, self.n, -1)], dim=-1)
                fuse_adj = self.aggregator(semantic_feature).squeeze(dim=-1)

                fuse_adj = self.adj[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                     self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] * fuse_adj

                eye = 1 - torch.eye(self.adj.shape[0]).type_as(fuse_adj)
                adj = fuse_adj * eye[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                     self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] + torch.eye(
                    self.adj.shape[0]).type_as(fuse_adj)[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                                         self.part_prior['wnids2id'][
                                                             self.label2catname[y[i].item()]] + 1, :]

                z = self.encoder(input_feature)
                g = torch.mm(adj, z)
                out = self.decoder(g)
                outputs.append(out)
                targets.append(self.class_feature[y[i].item()]['mean'])
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            return outputs, targets
        else:
            nb = x.shape[0]
            outputs = []
            for i in range(nb):
                input_feature = torch.zeros(self.n, self.in_dim).cuda()
                for k, v in self.metapart_feature.items():
                    input_feature[k:k + 1, :] = v['mean']
                input_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                              self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] = x[i:i + 1, :]

                semantic_feature = self.semantic_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                                         self.part_prior['wnids2id'][
                                                             self.label2catname[y[i].item()]] + 1, :, :]
                semantic_feature = torch.cat([semantic_feature, x[i:i + 1, :].unsqueeze(0).expand(-1, self.n, -1)],
                                             dim=-1)
                fuse_adj = self.aggregator(semantic_feature).squeeze(dim=-1)
                fuse_adj = self.adj[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                    self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] * fuse_adj
                eye = 1 - torch.eye(self.adj.shape[0]).type_as(fuse_adj)
                adj = fuse_adj * eye[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                     self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] + torch.eye(
                    self.adj.shape[0]).type_as(fuse_adj)[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                                         self.part_prior['wnids2id'][
                                                             self.label2catname[y[i].item()]] + 1, :]
                z = self.encoder(input_feature)
                out = torch.mm(adj, z)
                out = self.decoder(out)
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)

            return outputs, None

    def reparameterize(self, mu, var):
        std = var
        eps = torch.randn_like(std)
        return mu + eps*std
