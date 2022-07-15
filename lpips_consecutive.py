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
import pandas as pd
from collections import defaultdict

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
        random.shuffle(filelist)
        filelist = filelist[:args.n_sample]
        N = len(filelist)
        dists = torch.zeros(size=(N, N), device=device)
        for i in tqdm(range(N)):
            for j in range(i + 1, N):
                input1_path = os.path.join(args.path, filelist[i])
                input2_path = os.path.join(args.path, filelist[j])

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
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--start_ckpt", type=int, required=True)
    parser.add_argument("--end_ckpt", type=int, required=True)
    parser.add_argument("--ckpt_step", type=int, default=500)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--n_run", type=int, default=5)

    args = parser.parse_args()

    paths = [args.root + f"/sample/individual/{i}" for i in range(args.start_ckpt, args.end_ckpt+500, 500)]
    base_dic = defaultdict(list)
    for i, path in enumerate(paths):
        print(f"{i+1} / {len(paths)} in process")
        args.path = path
        res = []
        for _ in range(args.n_run):
            res += [calc_lpips(args).item()]
            # print(f"{_+1} / args.n_run: {sum(res) / (_+1)}")
        mean = sum(res) / args.n_run
        std = stdev(res)
        base_dic["Average"].append(mean)
        base_dic["Std_Dev"].append(std)
        print(f"Final LPIPS Mean: {mean:.4f} / Std: {std:.4f}")
    output_df = pd.DataFrame(base_dic, index=list(range(args.start_ckpt, args.end_ckpt+500, 500)))
    print(output_df)
    output_df.to_csv(args.root+f"/LPIPS_output.csv")