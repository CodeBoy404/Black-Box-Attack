import torch
import argparse
from better_dnn import DDN
import torchvision.transforms as transforms

from attack import attack
from utils import NormalizedModel
from models import resnet18, resnext50_32x4d
import imageio
import dataset
import numpy as np
from time import time
from AHBA_attack import untagetattack, bound_search, binary_search

start_time = time()

parser = argparse.ArgumentParser('1000time Attack example')
parser.add_argument('data', default='data', help='path to dataset')
parser.add_argument('--model-path', '--m', required=True)
parser.add_argument('--surrogate-model-path', '--sm', default="resnext50_32x4d_ddn.pt")
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

# load dataset -- tinyImagenet
test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = dataset.TinyImageNet(args.data, mode='test', transform=test_transform)
all_test_num = 0
all_l2_norm = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
attacks = [
    DDN(100, device=device),
    # FGM_L2(1)
]

# load attack model
m = resnet18()
model = NormalizedModel(m, image_mean, image_std)
state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)
model.eval().to(device)

# Attack model only return predict class
def black_box_model(img):
    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
    t_img = t_img.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t_img).argmax()


# Load surrogate model

smodel = resnext50_32x4d()
smodel = NormalizedModel(smodel, image_mean, image_std)
state_dict = torch.load(args.surrogate_model_path)
smodel.load_state_dict(state_dict)
smodel.eval().to(device)
surrogate_models = [smodel]

# choose attack way
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

for i in range(100, 200):
    img = imageio.core.util.Array(test_dataset.__getitem__(i)[0].permute(1, 2, 0).cpu().numpy() * 255)

    label = black_box_model(img)

    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1).unsqueeze(0)
    t_img = t_img.to(device)


    # When the label predicted by the avatar model is not equal to that predicted by the attack model
    if smodel(t_img).argmax() != black_box_model(img):
        adv, test_num = untagetattack(model=black_box_model, s_models=surrogate_models, attacks=attacks,
                                      image=img, label=label, targeted=False, device=device)

        if adv is None or black_box_model(img) == black_box_model(adv):
            # attack failure
            error_num = error_num + 1
        else:
            # attack success
            all_test_num = all_test_num + test_num
            l2_norm = np.linalg.norm(((adv.astype(np.float32) - img.astype(np.float32)) / 255))
            print(i, " ", l2_norm)
            all_l2_norm = all_l2_norm + l2_norm

    else:
        # The label predicted by the avatar model is the same as that predicted by the attack model
        adv, count, test_num = attack(black_box_model, surrogate_models, attacks,
                                      img, label, targeted=False, device=device)
        if adv is None or black_box_model(img) == black_box_model(adv):
            error_num = error_num + 1
        else:
            all_test_num = all_test_num + test_num
            l2_norm = np.linalg.norm(((adv.astype(np.float32) - img.astype(np.float32)) / 255))
            print(i, " ", l2_norm)
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

print("Total time is " + str(end_time - start_time) + " s.")
print("error:" + str(error_num))
print("all_test_time:" + str(all_test_num))
print('average L2 norm of the attack: {:.4f}'.format(all_l2_norm / times))
print('最大范数:', max_norm)
print('最小范数:', min_norm)
print('最大调用次数:', max_call)
print('最小调用次数:', min_call)
