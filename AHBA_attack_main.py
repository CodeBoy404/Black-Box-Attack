import AHBA_targeted_attack

import imageio
import torch
import torchvision.transforms as transforms
import numpy as np
from utils import NormalizedModel
from imageio import imread,imwrite
import time
import argparse
import datetime
import os
from PIL import Image
from models import resnet18, resnext50_32x4d
from better_dnn import DDN
from attack import attack
import dataset
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from AHBA_attack import untagetattack

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Attack example')
    parser.add_argument('--data', '--f', help='path to save image', default='data')
    parser.add_argument('--model-path', '--m', default='resnet18_clean.pt', help='path to AttackModel')
    parser.add_argument('--surrogate-model-path', '--sm', default="resnext50_32x4d_ddn.pt", help='path to surrogateModel')
    parser.add_argument('--targeted', '--t', default=0, type=int, help='0 is untargeted, 1 is targeted')
    parser.add_argument('--attack-image', '--i', default="data/img.png", help='path to attack_image')
    parser.add_argument('--attack-Class', '--c', default="data/img1.png", help='path to the image of attack Class')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Data loading code
    args = parser.parse_args()
    start_time = time.time()
    img = imread(args.attack_image)
    path = args.data
    print(args.targeted)
    if args.targeted == 0:
        image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)
        m = resnet18()
        attacks = [
            DDN(100, device=device),
        ]
        model = NormalizedModel(m, image_mean, image_std)
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        def black_box_model(img):
            t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
            t_img = t_img.unsqueeze(0).to(device)
            with torch.no_grad():
                return model(t_img).argmax()

        smodel = resnext50_32x4d()
        smodel = NormalizedModel(smodel, image_mean, image_std)
        state_dict = torch.load(args.surrogate_model_path)
        smodel.load_state_dict(state_dict)
        smodel.eval().to(device)
        surrogate_models = [smodel]
        label = black_box_model(img)
        t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1).unsqueeze(0)
        t_img = t_img.to(device)

        # When the label predicted by the avatar model is not equal to that predicted by the attack model
        if smodel(t_img).argmax() != black_box_model(img):
            adv, test_num = untagetattack(model=black_box_model, s_models=surrogate_models, attacks=attacks,
                                          image=img, label=label, targeted=False, device=device)

            if adv is None or black_box_model(img) == black_box_model(adv):
                # attack failure
                print("攻击失败")
            else:
                # attack success
                print("原始图片类别：",black_box_model(img))
                print("对抗性图片类别：", black_box_model(adv))
                print("call次数：", test_num)
                path = path + "/adv.png"
                imwrite(path, adv)
                l2_norm = np.linalg.norm(((adv.astype(np.float32) - img.astype(np.float32)) / 255))
                print("norm is", l2_norm)

        else:
            # The label predicted by the avatar model is the same as that predicted by the attack model
            adv, count, test_num = attack(black_box_model, surrogate_models, attacks,
                                          img, label, targeted=False, device=device)
            if adv is None or black_box_model(img) == black_box_model(adv):
                print("攻击失败")
            else:
                print("原始图片类别：",black_box_model(img))
                print("对抗性图片类别：", black_box_model(adv))
                print("call次数：", test_num)
                path = path + "/adv.png"
                imwrite(path, adv)
                l2_norm = np.linalg.norm(((adv.astype(np.float32) - img.astype(np.float32)) / 255))
                print("norm is", l2_norm)
    elif args.targeted == 1:
        initial_sample = imread(args.attack_Class)
        target_sample = img
        AHBA_targeted_attack.boundary_attack(initial_sample, target_sample, paths=path)
    end_time = time.time()
    print("执行时间: ", end_time - start_time)
