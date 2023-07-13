import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import clip
from nltk.corpus import wordnet as wn
from my_main.clip.labels_and_prompts import imagenet_templates, imagenet_classes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import pickle as pkl

def load_clip(mode):
    model_zoo = clip.available_models()
    assert mode in model_zoo, "mode error!"
    model, preprocess = clip.load(mode)
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    return model, preprocess

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
    mode = 'ViT-B/32'
    data_file = 'data/tiered'
    output_file = 'semantic_embeddings/clip_semantic_emb_name_80_tiredImagnet.pkl' # tiredImageNet 存放点
    
    
    with open(data_file+'/class_names.txt', 'r') as f:
        class_names =  f.read().splitlines()
 
    with open(data_file+'/synsets.txt', 'r') as f:
        wnids =  f.read().splitlines()
    
    
    tokenize = clip.tokenize
    model, preprocess = load_clip(mode)

    wnid2synset_dict = {}
    for wnid, class_name in zip(wnids, class_names):
        synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
        name = synset.name().split('.')[0].replace('_', ' ')
        definition = synset.definition()
        # semantic_emb = get_clip_semantic_emb(definition, imagenet_templates, model)
        semantic_emb = get_clip_semantic_emb(name, imagenet_templates, model)
        
        wnid2synset_dict[wnid] = {
            'name':name, 'definition':definition, 'semantic_emb':semantic_emb.cpu()
        }
        
    with open(output_file, 'wb') as f:
        pkl.dump(wnid2synset_dict, f)
    
    # with open(output_file, 'rb') as f:
    #     obj = pkl.load(f)