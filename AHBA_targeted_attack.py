from __future__ import print_function

import imageio
import torch
import torchvision.transforms as transforms
import numpy as np

from imageio import imread
import time
import argparse
import datetime
import os
from PIL import Image
import dataset
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def orthogonal_perturbation(delta, prev_sample, target_sample):
    prev_sample = prev_sample.reshape(64, 64, 3)
    # Generate perturbation
    perturb = np.random.randn(64, 64, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
    # Project perturbation onto sphere around target
    diff = (target_sample - prev_sample).astype(np.float32)
    diff /= get_diff(target_sample, prev_sample)
    diff = diff.reshape(3, 64, 64)
    perturb = perturb.reshape(3, 64, 64)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    # Check overflow and underflow
    mean = [103.939, 116.779, 123.68]
    perturb = perturb.reshape(64, 64, 3)
    overflow = (prev_sample + perturb) - np.concatenate((np.ones((64, 64, 1)) * (255. - mean[0]),
                                                         np.ones((64, 64, 1)) * (255. - mean[1]),
                                                         np.ones((64, 64, 1)) * (255. - mean[2])), axis=2)
    overflow = overflow.reshape(64, 64, 3)
    perturb -= overflow * (overflow > 0)
    underflow = np.concatenate((np.ones((64, 64, 1)) * (0. - mean[0]), np.ones((64, 64, 1)) * (0. - mean[1]),
                                np.ones((64, 64, 1)) * (0. - mean[2])), axis=2) - (prev_sample + perturb)
    underflow = underflow.reshape(64, 64, 3)
    perturb += underflow * (underflow > 0)
    return perturb


def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb


def get_converted_prediction(sample, classifier):
    sample = sample.reshape(64, 64, 3)
    mean = [103.939, 116.779, 123.68]
    sample[..., 0] += mean[0]
    sample[..., 1] += mean[1]
    sample[..., 2] += mean[2]
    sample = sample[..., ::-1].astype(np.uint8)
    sample = sample.astype(np.float32).reshape(1, 64, 64, 3)
    sample = sample[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    sample[..., 0] -= mean[0]
    sample[..., 1] -= mean[1]
    sample[..., 2] -= mean[2]
    label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
    return label


def draw(sample, classifier, folder, n_call=0, flag=False, norm=0, path="images"):
    label = get_converted_prediction(np.copy(sample), classifier)
    sample = sample.reshape(64, 64, 3)
    # Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
    mean = [103.939, 116.779, 123.68]
    sample[..., 0] += mean[0]
    sample[..., 1] += mean[1]
    sample[..., 2] += mean[2]
    sample = sample[..., ::-1].astype(np.uint8)
    # Convert array to image and save
    sample = Image.fromarray(sample)
    id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())

    # Save with predicted label for image (may not be adversarial due to uint8 conversion)
    if flag:
        sample.save(os.path.join(path, folder, "{}_{}_original.png".format(id_no, label)))
    else:
        sample.save(os.path.join(path, folder, "{}_{}_{}_{}.png".format(id_no, label, n_call, norm)))


def sampletoarray(sample):
    sample = sample.reshape(64, 64, 3)
    # Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
    mean = [103.939, 116.779, 123.68]
    sample[..., 0] += mean[0]
    sample[..., 1] += mean[1]
    sample[..., 2] += mean[2]
    sample = sample[..., ::-1].astype(np.uint8)
    return sample


def preprocess(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, 64, 64)
    sample_2 = sample_2.reshape(3, 64, 64)
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)


def binary_search(adversarial_sample, target_sample, attack_class, classifier, folder, paths, n_calls):
    adversarial_sample = sampletoarray(adversarial_sample)
    target_sample = sampletoarray(target_sample)

    upper = np.copy(target_sample).astype(np.float32) - np.copy(adversarial_sample).astype(np.float32)
    lower = np.copy(target_sample).astype(np.float32) - np.copy(target_sample).astype(np.float32)
    trial_sample = adversarial_sample
    for _ in range(100):
        middle = np.round((upper + lower) / 2)
        middle = np.clip(trial_sample + middle, 0, 255) - trial_sample
        trial_class = np.argmax(classifier.predict(preprocess(trial_sample + middle)))
        if trial_class == attack_class:
            lower = middle
        else:
            upper = middle
        diff_bin = upper - lower

        if (abs(diff_bin.max()) <= 1 and abs(diff_bin.min()) <= 1) or _ == 99:
            trial_sample = trial_sample + lower
            trial_sample = preprocess(trial_sample)
            draw(np.copy(trial_sample), classifier, folder, n_calls + _,
                 norm=np.linalg.norm((target_sample.astype(np.float32)
                                      - sampletoarray(np.copy(trial_sample)).astype(np.float32)) / 255), path=paths)
            n_calls = n_calls + _
            break
    return trial_sample


def boundary_attack(initial_sample, target_sample, paths="images", classifier=ResNet50(weights='imagenet'), img_i=-1):
    initial_sample = preprocess(initial_sample)
    target_sample = preprocess(target_sample)
    if img_i > -1:
        folder = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple()) + "_" + str(img_i)
    else:
        folder = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
    os.mkdir(os.path.join(paths, folder))
    draw(np.copy(initial_sample), classifier, folder, flag=True, path=paths)
    draw(np.copy(target_sample), classifier, folder, path=paths)

    attack_class = np.argmax(classifier.predict(initial_sample))

    n_steps = 0
    n_calls = 0
    epsilon = 1.
    delta = 0.1
    min_diff = 656556
    min_norm_1 = 255
    min_norm_adv = initial_sample
    init_norm = np.linalg.norm((sampletoarray(np.copy(target_sample)).astype(np.float32) -
                           sampletoarray(np.copy(initial_sample)).astype(np.float32)) / 255)
    print("初始norm：", init_norm)
    # Move first step to the boundary
    adversarial_sample = binary_search(np.copy(initial_sample), np.copy(target_sample), attack_class, classifier, folder
                                       , paths, n_calls)
    init_norm_2 = np.linalg.norm((sampletoarray(np.copy(target_sample)).astype(np.float32) -
                           sampletoarray(np.copy(adversarial_sample)).astype(np.float32)) / 255)
    print("二分搜索norm：", init_norm_2)
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample),
                                                                 adversarial_sample, target_sample)
        prediction = classifier.predict(trial_sample.reshape(1, 64, 64, 3))
        n_calls += 1
        if np.argmax(prediction) == attack_class:
            adversarial_sample = trial_sample
            break
        elif delta <= 1e-1:
            break
        else:
            epsilon *= 0.9
    while True:
        d_step = 0
        while True:
            d_step += 1
            trial_samples = []
            for i in np.arange(10):
                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
                trial_samples.append(trial_sample)
            predictions = classifier.predict(np.array(trial_samples).reshape(-1, 64, 64, 3))
            n_calls += 10
            predictions = np.argmax(predictions, axis=1)
            d_score = np.mean(predictions == attack_class)
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.9
                elif d_score > 0.7:
                    delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]
                break
            else:
                delta *= 0.9
        e_step = 0
        while True:
            e_step += 1
            trial_sample = adversarial_sample + forward_perturbation(
                epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
            prediction = classifier.predict(trial_sample.reshape(1, 64, 64, 3))
            n_calls += 1
            if np.argmax(prediction) == attack_class:
                adversarial_sample = trial_sample
                epsilon /= 0.9
                break
            elif e_step > 500:
                break
            else:
                epsilon *= 0.9
        n_steps += 1
        chkpts = [1, 5, 10, 50, 100, 500]
        if (n_steps in chkpts) or (n_steps % 500 == 0):
            draw(np.copy(adversarial_sample), classifier, folder, n_calls,
                 norm=np.linalg.norm((sampletoarray(np.copy(target_sample)).astype(np.float32) -
                                      sampletoarray(np.copy(adversarial_sample)).astype(np.float32)) / 255), path=paths)
        diff = np.mean(get_diff(adversarial_sample, target_sample))
        tmp_norm = np.linalg.norm((sampletoarray(np.copy(target_sample)).astype(np.float32) -
                                   sampletoarray(np.copy(adversarial_sample)).astype(np.float32)) / 255)
        # print("step: ", n_steps, ", norm is ", tmp_norm)
        if tmp_norm < min_norm_1 and np.argmax(classifier.predict(adversarial_sample)) == attack_class:
            min_norm_1 = tmp_norm
            min_norm_adv = np.copy(adversarial_sample)
        if diff <= 1e-3 or e_step > 500 or (n_steps > 1000 and tmp_norm - min_norm_1 > 0.1) \
                or (min_norm_1 < 5 and tmp_norm - min_norm_1 > 0.5):
            tmp_norm2 = np.linalg.norm(
                ((sampletoarray(np.copy(target_sample)).astype(np.float32) -
                  sampletoarray(np.copy(adversarial_sample)).astype(np.float32)) / 255))
            draw(np.copy(adversarial_sample), classifier, folder, n_calls,
                 norm=tmp_norm2, path=paths)
            if min_norm_1 < tmp_norm2:
                adversarial_sample = np.copy(min_norm_adv)
            else:
                adversarial_sample = adversarial_sample
            adversarial_sample = binary_search(np.copy(adversarial_sample).astype(np.float32),
                                               np.copy(target_sample).astype(np.float32), attack_class,
                                               classifier, folder
                                               , paths, n_calls)
            # upper = adversarial_sample - target_sample
            # lower = target_sample - target_sample

            # for _ in range(100):
            #     trial_sample = adversarial_sample
            #     middle = np.round((upper + lower) / 2)
            #     middle = np.clip(trial_sample + middle, 0, 255) - trial_sample
            #     trial_class = np.argmax(classifier.predict(preprocess(trial_sample + middle)))
            #     if trial_class == attack_class:
            #         lower = middle
            #     else:
            #         upper = middle
            #     diff_bin = upper - lower
            #
            #     if abs(diff_bin.max()) <= 1 and abs(diff_bin.min()) <= 1:
            #         trial_sample = trial_sample + lower
            #         trial_sample = preprocess(trial_sample)
            #         draw(np.copy(trial_sample), classifier, folder, n_calls + _,
            #              norm=np.linalg.norm((target_sample.astype(np.float32)
            #                                   - adversarial_sample.astype(np.float32)) / 255), path=paths)
            #         n_calls = n_calls + _
            #         break
            # print("min_norm:", min_norm_1)
            draw(np.copy(min_norm_adv), classifier, folder, 0,
                 norm=np.linalg.norm((sampletoarray(np.copy(target_sample)).astype(np.float32)
                                      - sampletoarray(min_norm_adv).astype(np.float32)) / 255), path=paths)
            break
        if min_diff > diff:
            min_diff = diff
    global all_test_norm
    global all_test_num
    global max_call
    global min_norm
    global min_call
    global max_norm
    norm = np.linalg.norm(((target_sample.astype(np.float32) - adversarial_sample.astype(np.float32)) / 255))
    if img_i > -1:
        f.write(str(img_i) + '\t' + str(norm) + '\n')
    print("norm is ", norm)
    all_test_norm += norm
    all_test_num += n_calls
    if max_call < n_calls:
        max_call = n_calls
    if min_call > n_calls:
        min_call = n_calls
    if max_norm < norm:
        max_norm = norm
    if min_norm > norm:
        min_norm = norm


all_test_num = 0
all_test_norm = 0
max_norm = -65535
min_norm = 65536
max_call = -65535
min_call = 65536
image_count = 400  # the number of test example

if __name__ == "__main__":
    f = open('norm.txt', 'w', encoding='utf-8')
    parser = argparse.ArgumentParser('1000time Attack example')
    parser.add_argument('data', help='path to dataset')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Data loading code
    args = parser.parse_args()
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = dataset.TinyImageNet(args.data, mode='test', transform=test_transform)
    start_time = time.time()
    for i in range(200, 200 + image_count):
        img = imread('data/img.png')
        initial_sample = imageio.core.util.Array(test_dataset.__getitem__(i)[0].permute(1, 2, 0).cpu().numpy() * 255)
        target_sample = img

        boundary_attack(initial_sample, target_sample, img_i=i)
        if (i - 199) % 10 == 0:
            end_time = time.time()
            print("样本数量：", i - 199)
            print("执行时间: ", end_time - start_time)
            print("最大范数: ", max_norm)
            print("最小范数: ", min_norm)
            print("平均范数: ", all_test_norm / (i - 199))
            print("最大call: ", max_call)
            print("最小call: ", min_call)
            print("平均calls: ", all_test_num / (i - 199))
    f.close()
