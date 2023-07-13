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
def get_mydata_MSVAE(path): 
    with open(path, 'rb') as f:
        data = pkl.load(f)
    
    support, proto,query, latent_support, latent_proto, latent_query, latent_sem = data

    return  support, proto,query, latent_support, latent_proto, latent_query, latent_sem

# 对样本进行预处理并画图
def plot_embedding(data, labels, latent_data, latent_labels):
    
    colors = np.array(['red', 'g',  'b' ])
    light_colors =  np.array(['salmon', 'lightgreen', 'lightsteelblue'])
    # np.random.seed(0)
    # colors = np.random.rand(len(set(labels)), 3)
    
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    
    
    x_min, x_max = np.min(latent_data, 0), np.max(latent_data, 0)
    
    latent_data = (latent_data - x_min) / (x_max - x_min)	
    
    
    plt.figure(figsize=(20,8))		# 创建图形实例
    plt.subplot(121)		# 创建子图
    plt.scatter(data[:3, 0], data[:3, 1], c=colors[labels[:3]], marker='*',cmap=CMAP, s=160 ) # 画prototypes
    plt.scatter(data[3:, 0], data[3:, 1], c=light_colors[labels[3:]], marker='.',cmap=CMAP,alpha=0.5 )
    plt.xticks([])		# 关闭坐标刻度
    plt.yticks([])
    plt.title('Image Feature Space', fontsize=14)
    
    plt.subplot(122)
    plt.scatter(latent_data[:3, 0], latent_data[:3, 1], c=colors[latent_labels[:3]], marker='*',cmap=CMAP, s=160 )
    plt.scatter(latent_data[3:6, 0], latent_data[3:6, 1], c=colors[latent_labels[3:6]], marker='+',cmap=CMAP, s=160)
    plt.scatter(latent_data[6:, 0], latent_data[6:, 1], c=light_colors[latent_labels[6:]], marker='.',cmap=CMAP, alpha=0.5)
    

    # plt.axis('off') # 坐标轴不可见
    plt.xticks([])		# 指定坐标的刻度
    plt.yticks([])
    plt.title('Shared Latent Space', fontsize=14)
    # 返回值pytho


# 主函数，执行t-SNE降维
def main():
    # data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
    support, proto,query, latent_support, latent_proto, latent_query, latent_sem = get_mydata_MSVAE(PATH)
    print('Starting compute t-SNE Embedding...')
    # print(label.shape)
    ts = TSNE(n_components=2, init='pca', random_state=0, perplexity=PREP, early_exaggeration=EXA)
    # t-SNE降维
    
  
    # index = 40 # 5
    for index in range(100):
  
        basic_data = np.concatenate((proto[index][0],query[index][0]), axis=0)
        basic_label =  proto[index][1] + query[index][1]
        basic_reslut = ts.fit_transform( basic_data)

        latent_data = np.concatenate((latent_proto[index][0] ,latent_sem[index][0],  latent_query[index][0] , ), axis=0)
        latent_label =  latent_proto[index][1] +  latent_sem[index][1] + latent_query[index][1] 
        latent_result = ts.fit_transform(latent_data)
    
    
    
        # 调用函数，绘制图像
        fig = plot_embedding(basic_reslut, basic_label,latent_result,latent_label)
        
        plt.savefig(osp.join('visulization/imgs', FILE_NAME + '_' + str(index) + '.pdf'), dpi=600)
        # 显示图像
        # plt.show()

# 主函数
if __name__ == '__main__':
    
    PATH = 'visualization_metatest.pkl'
    CMAP= 'Set2'
    PREP = 50 # 30 # 越大同类点簇越密集
    EXA = 80 # 12  # 越大异类簇间隔越远
    
    
    main()

