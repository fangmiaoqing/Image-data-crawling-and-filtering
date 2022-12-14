import os
import re
import cv2
import umap
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        # 冻结模型
        for p in self.features.parameters():
            p.requires_grad = False
        # 检测是否有GPU
        self.device = device
        self.to(self.device)

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


# 提取图像特征
def get_img_feature(model, img_path):
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = torch.from_numpy(img)
    img = img.to(model.device).float()
    img = torch.unsqueeze(img, 0)  # batch size 1
    img = img.permute(0, 3, 1, 2)
    feature = model(img)
    return feature


# UMAP降维
def do_umap(features, channel=2, random_state=None):
    model = umap.UMAP(n_components=channel, random_state=random_state)
    return model.fit_transform(features), model


# t-SNE降维
def do_tsne(data, random_state=0):
    tsne = TSNE(n_components=2, init='pca', random_state=random_state)
    return tsne.fit_transform(data), tsne


# 绘制数据图像
def plot_embedding(data, type=None, text=None, title="", colors=None):
    if type is None:
        type = np.zeros_like(data[:, 0])
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if text is not None:
            plt.text(data[i, 0], data[i, 1], str(text[i]),
                     color=plt.cm.Set1((type[i] + 1) / 10.) if colors is None else colors[type[i]],
                     fontdict={'weight': 'bold', 'size': 8})
        else:
            plt.scatter(data[i, 0], data[i, 1], s=3,
                        color=plt.cm.Set1((type[i] + 1) / 10.) if colors is None else colors[type[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig("./"+title+'.png')
    # plt.show()
    plt.close(fig)
    return fig


if __name__ == '__main__':
    root_dir = r"E:\download_imgs\download_images\yiren"
    # root_dir = r"E:\download_imgs\download_images\dog"
    file_suffix = "jpeg|jpg|png"
    remove_dir = root_dir + "/remove"
    if not os.path.exists(remove_dir):
        os.makedirs(remove_dir)

    # 模型初始化
    model = ResNet50()
    # 提取图像特征
    feature_list = []
    name_list = []
    for img_name in os.listdir(root_dir)[:]:
        # 对处理文件的类型进行过滤
        if re.search(file_suffix, img_name) is None:
            continue
        img_path = os.path.join(root_dir,img_name)
        mean, std = get_img_feature(model, img_path)
        mean = mean.to(device).numpy().reshape(-1)
        std = std.to(device).numpy().reshape(-1)
        feature = np.concatenate((mean, std), 0)
        print(feature.shape)
        feature_list.append(feature)
        name_list.append(img_name[8:10])

    # 特征绘图
    feature_list = np.array(feature_list)
    name_list = np.array(name_list)
    feature_list_tsne, _ = do_tsne(feature_list)
    plot_embedding(feature_list_tsne, text=name_list,title="tsne")
    feature_list_umap, _ = do_umap(feature_list)
    plot_embedding(feature_list_umap, text=name_list,title="umap")
    cv2.waitKey()
