import argparse
import random
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm
import sys
import lpips
from torchvision import transforms, utils
from torch.utils import data
import os
from PIL import Image
import numpy as np

def calc_lpips(args):
    device = args.device
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        preprocess = transforms.Compose([
            transforms.Resize([args.size, args.size]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        filelist = os.listdir(args.path)
        sourcelist = os.listdir(args.source_path)
        random.shuffle(filelist)
        filelist = filelist[:args.n_sample]
        N = len(filelist)
        M = len(sourcelist)
        indices = []
        for i in tqdm(range(N)):
            temp_dists = torch.zeros(M).to(device)
            for j in range(M):
                input1_path = os.path.join(args.path, filelist[i])
                input2_path = os.path.join(args.source_path, sourcelist[j])

                input_image1 = Image.open(input1_path)
                input_image2 = Image.open(input2_path)

                input_tensor1 = preprocess(input_image1)
                input_tensor2 = preprocess(input_image2)

                input_tensor1 = input_tensor1.to(device)
                input_tensor2 = input_tensor2.to(device)

                dist = lpips_fn(input_tensor1, input_tensor2)

                temp_dists[j] = dist
            indices.append(torch.argmax(temp_dists).item())
        print("Number of Modes: ", len(set(indices)))
        # print(indices)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_sample", type=int, default=500)

    args = parser.parse_args()

    calc_lpips(args)
