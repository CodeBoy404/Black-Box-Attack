import torch
from better_dnn import DDN
import random

from attack import attack
from utils import NormalizedModel
from models import resnet18, resnext50_32x4d
from models import cifar_resnet32
import imageio
import numpy as np
from time import time
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

# Load model under attack:
model= cifar_resnet32()
state_dict = torch.load(r'pretrained_model/cifar-10/cifar10-resnet32-e96f90cf.pth')
model.load_state_dict(state_dict)
model.eval().to(device)


# Simulate a black-box model (i.e. returns only predictions, no gradient):
def black_box_model(img):
    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
    t_img = t_img.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t_img).argmax()

# Load surrogate model
smodel = resnext50_32x4d()
smodel = NormalizedModel(smodel, image_mean, image_std)
state_dict = torch.load("resnext50_32x4d_ddn.pt")
smodel.load_state_dict(state_dict)
smodel.eval().to(device)


def load_file(cifar):
    '''加载cifar数据集'''

    import pickle
    with open(cifar, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


def convert_cifar10pix(X, pix):
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

    return X_resized

# load data_batch--cifar-10
data = load_file('pretrained_model/cifar-10/test_batch')
test_dataset = data['data']
test_dataset = convert_cifar10pix(test_dataset, 32)
Y = data['labels']

surrogate_models = [smodel]
attacks = [
    DDN(100, device=device),
    # FGM_L2(1)
]
error_num = 0
max_norm = -65535
min_norm = 65536
max_call = -65535
min_call = 65536

# the number of test example
times = 1000

# random test example
dataset_index = random.sample(range(0, 10000), times)
for i in dataset_index:
    img = test_dataset[i]
    label = black_box_model(img)

    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1).unsqueeze(0)
    t_img = t_img.to(device)

    l2_norm = 0

    if smodel(t_img).argmax() != black_box_model(img):
        adv, test_num = untagetattack(model=black_box_model, s_models=surrogate_models, attacks=attacks,
                                                 image=img, label=label, targeted=False, device=device)

        if adv is None or black_box_model(img) == black_box_model(adv):
            error_num = error_num + 1
            continue
        else:
            all_test_num = all_test_num + test_num
            l2_norm = np.linalg.norm(((adv - img) / 255))
            # print(l2_norm)
            all_l2_norm = all_l2_norm + l2_norm

    else:
        adv, count, test_num = attack(black_box_model, surrogate_models, attacks,
                                  img, label, targeted=False, device=device)

        if adv is None or black_box_model(img) == black_box_model(adv):
            error_num = error_num + 1
            continue
        else:
            all_test_num = all_test_num + test_num
            l2_norm = np.linalg.norm(((adv - img) / 255))
            # print(l2_norm)
            all_l2_norm = all_l2_norm + l2_norm
    if l2_norm > max_norm:
        max_norm = l2_norm
    if l2_norm < min_norm:
        min_norm = l2_norm
    if test_num > max_call:
        max_call = test_num
    if test_num < min_call:
        min_call = test_num
end_time = time()
imageio.imsave('delta_cifar10/ori_img' + str(i) + '.png', img)
imageio.imsave('delta_cifar10/adv_img' + str(i) + '.png', adv)

print("Total time is " + str(end_time - start_time) + " s.")
print("error:" + str(error_num))
print("all_test_time:" + str(all_test_num))
print('average L2 norm of the attack: {:.4f}'.format(all_l2_norm / (times - error_num)))
print('最大范数:', max_norm)
print('最小范数:', min_norm)
print('最大调用次数:', max_call)
print('最小调用次数:', min_call)
