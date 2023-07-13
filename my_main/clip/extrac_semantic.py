import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import clip
from nltk.corpus import wordnet as wn
from labels_and_prompts import imagenet_templates, imagenet_classes
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
    # data_file = '/run/media/cv/d/lzl/mmd/python文件处理脚本/label_ocr_recog/0527/train/mmfewshot/data/mini_imagenet'
    data_file = '/run/media/cv/d/lzl/mmd/python文件处理脚本/label_ocr_recog/0527/train/mmfewshot/data/tiered_imagenet'
    # output_file = 'semantic_embeddings/clip_semantic_emb_name_80.pkl' # miniImageNet 存放点
    output_file = 'semantic_embeddings/clip_semantic_emb_name_80_tiredImagnet.pkl' # tiredImageNet 存放点
    
    
    class_names = os.listdir(os.path.join(data_file, 'images'))
    tokenize = clip.tokenize
    model, preprocess = load_clip(mode)

    wnid2synset_dict = {}
    for class_name in class_names:
        synset = wn.synset_from_pos_and_offset('n', int(class_name[1:]))
        name = synset.name().split('.')[0].replace('_', ' ')
        definition = synset.definition()
        # semantic_emb = get_clip_semantic_emb(definition, imagenet_templates, model)
        semantic_emb = get_clip_semantic_emb(name, imagenet_templates, model)
        
        wnid2synset_dict[class_name] = {
            'name':name, 'definition':definition, 'semantic_emb':semantic_emb.cpu().float()
        }
        
    with open(output_file, 'wb') as f:
        pkl.dump(wnid2synset_dict, f)
    
    # with open(output_file, 'rb') as f:
    #     obj = pkl.load(f)