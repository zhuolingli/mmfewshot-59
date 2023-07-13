from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec





glove_input_file = 'glove_pre/glove.840B.300d.txt'
word2vec_output_file = 'glove_pre/glove.840B.300d.word2vec.txt'
(count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
print(count, '\n', dimensions)

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# 获得单词cat的词向量 
cat_vec = glove_model['cat']
print(cat_vec)    
# 获得单词frog的最相似向量的词汇
print(glove_model.most_similar('frog'))


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from nltk.corpus import wordnet as wn
from my_main.clip.labels_and_prompts import imagenet_templates, imagenet_classes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import pickle as pkl
from my_main.clip.glove import GloVe
import gensim

if __name__ == '__main__':
    data_file = 'data/mini_imagenet'
    # output_file = 'semantic_embeddings/glove_4_18' # miniImageNet 存放点
    
    
    class_names = os.listdir(os.path.join(data_file, 'images'))

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

