import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from nltk.corpus import wordnet as wn
from labels_and_prompts import imagenet_templates, imagenet_classes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import pickle as pkl
from glove import GloVe
import gensim

def get_clip_semantic_emb(classname, templates, model):
    with torch.no_grad():
        texts = [template.format(classname) for template in templates] #format with class
        texts = clip.tokenize(texts).cuda() #tokenize
        class_embeddings = model.encode_text(texts) #embed with text encoder
        # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        # class_embedding = class_embeddings.mean(dim=0)
        # class_embedding /= class_embedding.norm()
    return class_embeddings
    # return class_embedding

if __name__ == '__main__':
    data_file = 'data/mini_imagenet'
    output_file = 'semantic_embeddings/glove_4_18' # miniImageNet 存放点
    
    
    class_names = os.listdir(os.path.join(data_file, 'images'))

    # wnid2synset_dict = {}
    # for class_name in class_names:
    #     synset = wn.synset_from_pos_and_offset('n', int(class_name[1:]))
    #     name = synset.name().split('.')[0].replace('_', ' ')
    #     definition = synset.definition()
    #     # semantic_emb = get_clip_semantic_emb(definition, imagenet_templates, model)
    #     semantic_emb = get_clip_semantic_emb(name, imagenet_templates, model)
        
    #     wnid2synset_dict[class_name] = {
    #         'name':name, 'definition':definition, 'semantic_emb':semantic_emb.cpu().float()
    #     }
        
    # with open(output_file, 'wb') as f:
    #     pkl.dump(wnid2synset_dict, f)
    
    
    wnid2synset_dict = {}
    glove = GloVe()
    template = 'An image of the {}'
    for class_name in class_names:
        synset = wn.synset_from_pos_and_offset('n', int(class_name[1:]))
        name = synset.lemma_names()
        name = [template.format(item)  for item in name]
        definition = synset.definition()
        # semantic_emb = get_clip_semantic_emb(definition, imagenet_templates, model)
        semantic_emb = glove[name]
        semantic_emb = semantic_emb.unsqueeze(0).expand(80, -1)
        wnid2synset_dict[class_name] = {
            'name':name, 'definition':definition, 'semantic_emb':semantic_emb.cpu().float()
        }
        
    with open(output_file, 'wb') as f:
        pkl.dump(wnid2synset_dict, f)

