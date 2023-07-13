import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import pickle as pkl
import os.path as osp
import os
import random 
import time

t = time.gmtime()
FILE_NAME = time.strftime("%Y-%m-%d %H:%M:%s",t)

COLOR_NAMES= {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
# 'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
# 'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
# 'white':                '#FFFFFF',
# 'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

COLOR_NAMES = list(COLOR_NAMES.keys())

random.shuffle(COLOR_NAMES)


def get_mydata_MSVAE(path, split, class_num=64, num_img_per_class=300): 
    with open(osp.join(path, split,'all_img_feats.pickle'), 'rb') as f:
        data = pkl.load(f)
    
    
    labels =  np.array(data['gt_labels'])
    assert max(labels) in (63, 15, 19), 'label misassignment! The max label is{}'.format(max(labels))
    feats = np.array(data['feats'])
    
    class_num = len(set(labels)) if (len(set(labels)) < class_num) else class_num
    labelids = random.sample(set(labels), class_num)
    sub_feats = np.empty((0,640),np.float32)
    sub_labels = []

    for i in labelids:
        
        choose_id = random.sample(range(600 * i,  600 * (i+1)), num_img_per_class )
        
        sub_feats = np.concatenate((sub_feats, feats[choose_id]), axis=0)
        sub_labels.extend([i] * num_img_per_class)

    return sub_feats, sub_labels

# 对样本进行预处理并画图
def plot_embedding(data, labels, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    
    colors = np.random.rand(len(set(labels)), 3)
    
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    fig = plt.figure()		# 创建图形实例
    ax = plt.subplot(111)		# 创建子图
    plt.scatter(data[:, 0], data[:, 1], c=colors[labels], marker='o', s=1,cmap=CMAP )
    
    plt.axis('off') # 坐标轴不可见
    # plt.xticks()		# 指定坐标的刻度
    # plt.yticks()
    # plt.title(title, fontsize=14)
    # 返回值pytho
    return fig

# 主函数，执行t-SNE降维
def main(class_num_to_vis, num_img_per_class):
    # data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
    my_allvector, my_alllabel = get_mydata_MSVAE(PATH, SPLIT, class_num_to_vis, num_img_per_class)
    print('Starting compute t-SNE Embedding...')
    # print(label.shape)
    ts = TSNE(n_components=2, init='pca', random_state=0, perplexity=PREP, early_exaggeration=EXA)
    # t-SNE降维
    reslut = ts.fit_transform(my_allvector)
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, my_alllabel, 't-SNE Embedding of digits')
    
    plt.savefig(osp.join('visulization', FILE_NAME), dpi=3600)
    # 显示图像
    plt.show()

# 主函数
if __name__ == '__main__':
    
    PATH = 'my_output/proto_emb/mini'
    SPLIT = 'train'
    CMAP= 'Set2'
    PREP = 8 # 30 # 越大同类点簇越密集
    EXA = 1 # 12  # 越大异类簇间隔越远
    class_num_to_vis = 64
    num_img_per_class = 500
    
    
    main(class_num_to_vis, num_img_per_class)

