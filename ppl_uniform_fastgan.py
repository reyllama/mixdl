import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import lpips
from fastgan_models import Generator
from statistics import stdev
from collections import defaultdict
import pandas as pd

def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Perceptual Path Length calculator")

    parser.add_argument(
        "--space", choices=["z", "w"], default="w", help="space that PPL calculated with"
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the models"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5000,
        help="number of the samples for calculating PPL",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--step", type=float, default=0.1, help="step size for interpolation"
    )
    parser.add_argument(
        "--crop", action="store_true", help="apply center crop to the images"
    )
    parser.add_argument(
        "--sampling",
        default="end",
        choices=["end", "full"],
        help="set endpoint sampling method",
    )
    parser.add_argument(
        "--n_mlp",
        type=int,
        default=8,
        help="Number of FC layers in mapping network"
    )
    parser.add_argument(
        "ckpt", metavar="CHECKPOINT", help="path to the model checkpoints"
    )

    args = parser.parse_args()

    noise_dim = 256
    # device = torch.device('cuda:%d' % (args.cuda))

    net_ig = Generator(ngf=64, nz=noise_dim, nc=3, im_size=256)  # , big=args.big )
    net_ig.to(device)

    ckpt = torch.load(args.ckpt, map_location=lambda a,b: a)

    # g = Generator(args.size, latent_dim, args.n_mlp).to(device)
    net_ig.load_state_dict(ckpt["g"])
    net_ig.eval()

    percept = lpips.LPIPS(net='vgg').to(device)

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]

    steps = np.arange(0, 1+args.step, args.step)
    output_dists = defaultdict(list)


    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            # noise = g.make_noise()

            inputs = torch.randn([batch * 2, noise_dim], device=device)
            if args.sampling == "full":
                lerp_t = torch.rand(batch, device=device)
            else:
                lerp_t = torch.zeros(batch, device=device)

            for i in range(len(steps) - 1):
                prev_step, step = steps[i], steps[i + 1]

                # if args.space == "w":
                #     latent = g.get_latent(inputs)
                #     latent_t0, latent_t1 = latent[::2], latent[1::2]
                #     latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None] + prev_step)
                #     latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + step)
                #     latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

                latent_t0, latent_t1 = inputs[::2], inputs[1::2]
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None] + prev_step)
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + step)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*inputs.shape)

                # image, _ = g([latent_e], input_is_latent=True, noise=noise)
                image = net_ig(latent_e)[0]

                if args.crop:
                    c = image.shape[2] // 8
                    image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

                factor = image.shape[2] // 256

                if i==0:
                    start = image[::2]
                if i==len(steps)-2:
                    end = image[1::2]
                    endpoint_dist = percept(start, end).view(image.shape[0] // 2) / (
                            args.step ** 2
                    )
                    output_dists[str(i+1)].append(endpoint_dist.to("cpu").numpy())

                if factor > 1:
                    image = F.interpolate(
                        image, size=(256, 256), mode="bilinear", align_corners=False
                    )

                dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                    args.step ** 2
                )
                output_dists[str(i)].append(dist.to("cpu").numpy())

    distances = dict()
    for i in range(len(steps)):
        temp = np.concatenate(output_dists[str(i)], 0)

        # lo = np.percentile(temp, 1, interpolation="lower")
        # hi = np.percentile(temp, 99, interpolation="higher")
        # filtered_dist = np.extract(
        #     np.logical_and(lo <= temp, temp <= hi), temp
        # )
        distances[str(i)] = temp
    # print(distances)
    distances['endpoint'] = distances[f"{len(steps)-1}"]
    del distances[f"{len(steps)-1}"]
    # output_dists.append(filtered_dist.mean())
    dist_df = pd.DataFrame(distances, index=list(range(len(distances['0']))))
    # print(f"ppls: {dist_df}")
    means = dist_df.drop(columns=['endpoint']).mean(axis=1)
    stds = dist_df.drop(columns=['endpoint']).std(axis=1)
    assert len(stds) == len(distances['0'])
    # print(sum(stds)/len(stds))
    print("Mean: ", means.mean())
    print("Std.Dev: ", stds.mean())
    print("Endpoint Mean: ", dist_df['endpoint'].mean())
    # ckpt_num = int(args.ckpt.split("/")[-1][:-3])
    # save_dir = "/".join(args.ckpt.split("/")[:-2])+f"/ppl_uniform_at_{ckpt_num}.csv"
    # dist_df.to_csv(temp.csv)
    # output_dists = [dist.astype(np.float64) for dist in distances.items()]

    # print("ppl:", filtered_dist.mean())
