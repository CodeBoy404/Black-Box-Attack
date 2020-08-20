import torch
import argparse
from better_dnn import DDN
import torchvision.transforms as transforms

from utils import NormalizedModel
from fgm_l2 import FGM_L2
from attack import attack
from utils import NormalizedModel
from models import resnet18, resnext50_32x4d
from tensorflow.keras.applications import ResNet101, ResNet50
import torchvision.models as cleanmodels
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
from untagetattack import untagetattack, bound_search, binary_search

start_time = time()

parser = argparse.ArgumentParser('1000time Attack example')
parser.add_argument('data', help='path to dataset')
parser.add_argument('--model-path', '--m', required=True)
parser.add_argument('--surrogate-model-path', '--sm', required=True)
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', '-b', default=64, type=int, help='mini-batch size')

args = parser.parse_args()

image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)

# Data loading code
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = dataset.TinyImageNet(args.data, mode='test', transform=test_transform)
all_test_num = 0
all_l2_norm = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
attacks = [
    DDN(100, device=device),
    # FGM_L2(1)
]
# device = torch.device('cuda:0')
# Load model under attack:
m = resnet18()
# model = torch.load(r'model&dataset/mnist.pth')
# m = cleanmodels.resnet50(pretrained=True)
model = NormalizedModel(m, image_mean, image_std)
state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)
model.eval().to(device)
# model = ResNet50(weights='imagenet')


# Simulate a black-box model (i.e. returns only predictions, no gradient):


def black_box_model(img):
    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
    t_img = t_img.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t_img).argmax()

# Load surrogate model
# smodel = cleanmodels.resnet101(pretrained=True)
smodel = resnext50_32x4d()
smodel = NormalizedModel(smodel, image_mean, image_std)
state_dict = torch.load(args.surrogate_model_path)
smodel.load_state_dict(state_dict)
smodel.eval().to(device)
#
# smodel.eval().to(device)


def boundaryinitial(
        image: np.ndarray,
        label: int,
        targeted: bool,
        device: Optional[torch.device]):

    init_img = image  # 还没有进行初始化的原始图片
    t_image = torch.tensor(image).float().div(255).permute(2, 0, 1)
    t_image = t_image.unsqueeze(0).to(device)
    t_label = torch.tensor(label, device=device).unsqueeze(0)
    adv_img = DDN(100, device=device).attack(model=smodel, inputs=t_image, labels=t_label, targeted=targeted).squeeze(0)

    delta = adv_img.permute(1, 2, 0).cpu().numpy() * 255 - image
    delta = np.round(delta)
    norm = np.linalg.norm(delta)
    found = False
    label2 = black_box_model(image)
    print(black_box_model(delta+image))
    if norm > 0:
        print("边界搜索")
        lower, upper, found, num1 = bound_search(model=black_box_model, image=image,
                                                 label=label2, delta=delta,
                                                 targeted=targeted)
    print(upper)
    if found:
        adv_img = image+upper
    else:
        return False, 0, adv_img

    # t_image和adv_img都已经进行归一化了

    print("初始l2范数：", end=" ")
    print(np.linalg.norm(upper/255))
    num = num1
    flag1, num2, adv1 =untargeted_boundary_attack(adv_img, image, black_box_model)
    num = num1 + num2
    return True, num, adv1
    # return adv_img, count, test_num


#

surrogate_models = [smodel]
attacks = [
    DDN(100, device=device),
    # FGM_L2(1)
]
error_num = 0
print(len(test_dataset))
f = open('l2.txt', 'w', encoding='utf-8')
max_norm = -65535
min_norm = 65536
max_call = -65535
min_call = 65536

times = 200

for i in range(0, times):
    img = imageio.core.util.Array(test_dataset.__getitem__(i)[0].permute(1, 2, 0).cpu().numpy() * 255)

    label = black_box_model(img)

    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1).unsqueeze(0)
    t_img = t_img.to(device)

    # print("替身模型:", smodel(t_img).argmax(), end=" ")
    # print("被攻击:", black_box_model(img))

    if smodel(t_img).argmax() != black_box_model(img):
        # label = smodel(t_img).argmax()
        adv, test_num = untagetattack(model=black_box_model, s_models=surrogate_models, attacks=attacks,
                                                 image=img, label=label, targeted=False, device=device)
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

        if black_box_model(img) == black_box_model(adv):
            error_num = error_num + 1
        else:
            # print("初始类型:", black_box_model(img), end=" ")
            # print("攻击图片类型:", black_box_model(adv))
            all_test_num = all_test_num + test_num
            l2_norm = np.linalg.norm(((adv - img) / 255))
            # print(l2_norm)
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

        if black_box_model(img) == black_box_model(adv):
            error_num = error_num + 1
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
print('average L2 norm of the attack: {:.4f}'.format(all_l2_norm / times))
# print('Saving adversarial image to "data/adv.png"')

# imwrite('data/adv.png', adv)

print('最大范数:', max_norm)
print('最小范数:', min_norm)
print('最大调用次数:', max_call)
print('最小调用次数:', min_call)
