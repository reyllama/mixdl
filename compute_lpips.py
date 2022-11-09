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
from statistics import stdev


def calc_lpips(args, fn, ckpt):
    device = args.device
    with torch.no_grad():
        lpips_fn = fn.to(device)
        preprocess = transforms.Compose([
            transforms.Resize([args.size, args.size]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        filelist = os.listdir(os.path.join(args.path, str(ckpt)))
        random.shuffle(filelist)
        if args.n_sample == -1:
            args.n_sample = len(filelist)
        filelist = filelist[:args.n_sample]
        N = len(filelist)
        dists = torch.zeros(size=(N, N), device=device)
        for i in tqdm(range(N)):
            for j in range(i + 1, N):
                input1_path = os.path.join(args.path, str(ckpt), filelist[i])
                input2_path = os.path.join(args.path, str(ckpt), filelist[j])

                input_image1 = Image.open(input1_path)
                input_image2 = Image.open(input2_path)

                input_tensor1 = preprocess(input_image1)
                input_tensor2 = preprocess(input_image2)

                input_tensor1 = input_tensor1.to(device)
                input_tensor2 = input_tensor2.to(device)

                dist = lpips_fn(input_tensor1, input_tensor2)

                dists[i,j] = dist
        return 2*dists.sum()/(N*(N-1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--start", type=int, default=20000)
    parser.add_argument("--end", type=int, default=80000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_sample", type=int, default=-1)
    parser.add_argument("--n_run", type=int, default=5)

    args = parser.parse_args()

    fn = lpips.LPIPS(net='vgg')

    for ckpt in range(args.start, args.end+10000, 10000):
        res = []
        for _ in range(args.n_run):
            res += [calc_lpips(args, fn, ckpt).item()]
            # print(f"{ckpt}: {_+1} / {args.n_run}: {sum(res) / (_+1)}")
        mean = sum(res) / args.n_run
        if args.n_run < 2:
            std = 0
        else:
            std = stdev(res)
        print(f"{ckpt} Final LPIPS Mean: {mean:.4f} / Std: {std:.4f}")