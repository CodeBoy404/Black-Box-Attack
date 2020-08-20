import torch
import argparse
from better_dnn import DDN
import random
import torchvision.transforms as transforms

from utils import NormalizedModel
from fgm_l2 import FGM_L2
from attack import attack
from utils import NormalizedModel
from models import resnet18, resnext50_32x4d
from tensorflow.keras.applications import ResNet101, ResNet50
import torchvision.models as cleanmodels
from models import cifar_resnet32
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from imageio import imwrite, imread
import imageio
import dataset
import numpy as np
from time import time
import tensorflow as tf
from boundaryattack import untargeted_boundary_attack, preprocess
from typing import Callable, Iterable, Optional
import os
from AHBA_attack import untagetattack, bound_search, binary_search

start_time = time()


image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)

all_test_num = 0
all_l2_norm = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
attacks = [
    DDN(100, device=device),
    # FGM_L2(1)
]
# device = torch.device('cuda:0')
# Load model under attack:
model= cifar_resnet32(pretrained="cifar100")
# model = torch.load(r'model&dataset/cifar-10-batches-py/cifar10-resnet32-e96f90cf.pth')
# m = cleanmodels.resnet50(pretrained=True)
# model = NormalizedModel(m, image_mean, image_std)
# # state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
state_dict = torch.load(r'model&dataset/cifar-10-batches-py/cifar100-resnet32-6568a0a0.pth')
model.load_state_dict(state_dict)
model.eval().to(device)
# model = ResNet50(weights='imagenet')


# Simulate a black-box model (i.e. returns only predictions, no gradient):
# def black_box_model(img):
#     t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
#     t_img = t_img.unsqueeze(0).to(device)
#     t_img = t_img.squeeze(0)
#     t_img = preprocess(t_img)
#     with torch.no_grad():
#         return torch.Tensor(model.predict(t_img)).to(device).argmax()

def black_box_model(img):
    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
    t_img = t_img.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t_img).argmax()

# Load surrogate model
# smodel = cleanmodels.resnet50(pretrained=True)
smodel = resnext50_32x4d()
smodel = NormalizedModel(smodel, image_mean, image_std)
state_dict = torch.load("resnext50_32x4d_ddn.pt")
smodel.load_state_dict(state_dict)
smodel.eval().to(device)
#
# smodel.eval().to(device)

def load_file(cifar):
    '''加载cifar数据集'''

    import pickle
    with open(cifar, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


def convert_cifar10pix(X, pix):
    '''
    将cifar-10数据集的原始矩阵[10000,3072]转化为[10000,227,227,3]
    用于Alexnet训练
    X - 原始矩阵 shape = [10000, 3072]
    m - 10000代表图片张数
    n_H - 图片高度
    n_W - 图片宽度
    channel - RGB三色通道
    '''

    from PIL import Image
    import numpy as np
    X = np.reshape(X, (10000, 3, 32, 32))  # 将图片转换成(m, channel,n_H,n_W)
    X = X.transpose(0, 2, 3, 1)  # 转换成标准的矩阵(m, n_H,n_W, channel)
    X_resized = np.zeros((10000, pix, pix, 3))  # 创建一个存储修改过图片分辨率的矩阵

    for i in range(0, 10000):
        img = X[i]
        img = Image.fromarray(img)
        img = np.array(img.resize((pix, pix), Image.BICUBIC))  # 修改分辨率，再转为array类
        X_resized[i, :, :, :] = img

    # X_resized /= 255
    return X_resized

# 这里加载了第一个data_batch
data = load_file('model&dataset/cifar-10-batches-py/test_batch')
test_dataset = data['data']
test_dataset = convert_cifar10pix(test_dataset, 32)
Y = data['labels']

#

surrogate_models = [smodel]
attacks = [
    DDN(100, device=device),
    # FGM_L2(1)
]
error_num = 0
f = open('l2.txt', 'w', encoding='utf-8')
max_norm = -65535
min_norm = 65536
max_call = -65535
min_call = 65536
times = 1000
#39, 0.10500727716859516
#136, 0.014139416766525447
#288, 0.05376984000314544
#
dataset_index = random.sample(range(0, 10000), times)
for i in dataset_index:
    img = test_dataset[i]
    label = black_box_model(img)

    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1).unsqueeze(0)
    t_img = t_img.to(device)

    # print("替身模型:", smodel(t_img).argmax(), end=" ")
    # print("被攻击:", black_box_model(img))
    l2_norm = 0

    if smodel(t_img).argmax() != black_box_model(img):
        # label = smodel(t_img).argmax()
        # print('-------i:', i)
        adv, test_num = untagetattack(model=black_box_model, s_models=surrogate_models, attacks=attacks,
                                                 image=img, label=label, targeted=False, device=device)
        # print(adv)
        # flag2, test_num, adv = boundaryinitial(image=img, label=label, targeted=False, device=device)
        # adv = adv.permute(1, 2, 0).cpu().numpy() * 255
        # if flag2:
        #     print("初始类型:", black_box_model(img), end=" ")
        #     adv = torch.tensor(adv)
        #     adv = adv.squeeze(0).to(device)
        #
        #     adv = adv.cpu().numpy()
        #     print(adv.shape)
        #     print("攻击图片类型:", black_box_model(adv))
        #     all_test_num = all_test_num + test_num
        #     l2_norm = np.linalg.norm(((adv - img) / 255))
        #     print(l2_norm)
        #     all_l2_norm = all_l2_norm + l2_norm
        # else:
        #     error_num = error_num + 1

        if adv is None or black_box_model(img) == black_box_model(adv):
            error_num = error_num + 1
            continue
        else:
            # print("初始类型:", black_box_model(img), end=" ")
            # print("攻击图片类型:", black_box_model(adv))
            all_test_num = all_test_num + test_num
            l2_norm = np.linalg.norm(((adv - img) / 255))
            print(i, " --- ", l2_norm)
            all_l2_norm = all_l2_norm + l2_norm
        # continue


    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Sanity check: image correctly labeled:
    # t_img = t_img.to(device)
    else:
        adv, count, test_num = attack(black_box_model, surrogate_models, attacks,
                                  img, label, targeted=False, device=device)

        # print("初始类型:", black_box_model(img), end=" ")
        print('-------i:', i)
        # print("攻击图片类型:", black_box_model(adv))
    # if adv is None:
    #     # print("被攻击模型，不成功：" + str(black_box_model(img)))
    #     # print("替身模型，不成功：" + str(smodel(t_img).argmax()))
    #     error_num = error_num + 1
    #     continue
    # print("被攻击模型，成功：" + str(black_box_model(img)))
    # print("替身模型，成功：" + str(smodel(t_img).argmax()))

        if adv is None or black_box_model(img) == black_box_model(adv):
            error_num = error_num + 1
            continue
        else:
            all_test_num = all_test_num + test_num
            l2_norm = np.linalg.norm(((adv - img) / 255))
            # print(l2_norm)
            all_l2_norm = all_l2_norm + l2_norm

    # if i == 60:
    #
    #     print('adv', adv)
    #     print('img', img)
    # if l2_norm < 0.1:
    #     imageio.imsave('results/ori_img' + str(i) + '.png', img)
    #     imageio.imsave('results/adv_img' + str(i) + '.png', adv)
    #     f.write(str(i) + '\t' + str(l2_norm) + '\n')

    if l2_norm > max_norm:
        max_norm = l2_norm
    if l2_norm < min_norm:
        min_norm = l2_norm
    if test_num > max_call:
        max_call = test_num
    if test_num < min_call:
        min_call = test_num


    # diff_norm = img - adv
    # imageio.imsave('diff_norm.png', diff_norm)
    # break


end_time = time()
imageio.imsave('delta_cifar10/ori_img' + str(i) + '.png', img)
imageio.imsave('delta_cifar10/adv_img' + str(i) + '.png', adv)
# f.write(str(i) + '\t' + str(l2_norm) + '\n')
f.close()


print("Total time is " + str(end_time - start_time) + " s.")

print("error:" + str(error_num))
print("all_test_time:" + str(all_test_num))

# pred_on_adv = black_box_model(adv)
#
# print('True label: {}; Prediction on the adversarial: {}'.format(label,
#                                                                  pred_on_adv))

# Compute l2 norm in range [0, 1]
# l2_norm = np.linalg.norm(((adv - img) / 255))
print('average L2 norm of the attack: {:.4f}'.format(all_l2_norm / (times - error_num)))
# print('Saving adversarial image to "data/adv.png"')

# imwrite('data/adv.png', adv)

print('最大范数:', max_norm)
print('最小范数:', min_norm)
print('最大调用次数:', max_call)
print('最小调用次数:', min_call)
