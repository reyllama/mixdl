import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from model import Generator
from fastgan_models import Generator as FastGenerator
from calc_inception import load_patched_inception_v3
from dataset import MultiResolutionDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception3
from torchvision.datasets import ImageFolder
from collections import defaultdict


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img[0].to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to("cpu"))

    features = torch.cat(feature_list, 0)

    return features

@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features

@torch.no_grad()
def extract_feature_from_fastgan(
        generator, inception, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 256, device=device)
        img, _ = generator(latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features

def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

def main(args):
    if not args.input_is_image:
        ckpt = torch.load(args.ckpt)

        g = Generator(args.size, 512, 8).to(device)
        g.load_state_dict(ckpt["g_ema"])
        # g = nn.DataParallel(g)
        g.eval()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    if args.sfid:
        inception = load_patched_inception_v3(2)
    else:
        inception = load_patched_inception_v3(3)
    # inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception = inception.to(device)
    inception.eval()

    if args.input_is_image:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dset = ImageFolder(args.image_path, transform=transform)
        loader = DataLoader(dset, batch_size=args.batch, num_workers=4)

        features = extract_features(loader, inception, device).numpy()
    else:
        features = extract_feature_from_samples(
            g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
        ).numpy()
    print(f"extracted {features.shape[0]} features")

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    if not args.sfid:
        print("fid:", fid)
    else:
        print(f"sfid: {fid:.3f}")

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Calculate FID scores")

    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate mean for truncation",
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the generator"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5000,
        help="number of the samples for calculating FID",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for generator"
    )
    parser.add_argument(
        "--inception",
        type=str,
        required=False,
        help="path to precomputed inception embedding",
    )
    parser.add_argument(
        "--sfid",
        action='store_true',
        help='whether to calculate sfid'
    )
    parser.add_argument(
        "--input_is_image",
        action='store_true',
        help='whether input is image'
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=False
    )
    parser.add_argument(
        "--in_chunk",
        action="store_true",
        help="whether to compute metrics in chunk"
    )
    parser.add_argument(
        "--method",
        type=str,
        default='cvx_global2'
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=False,
        help='list all the datasets to evaluate FIDs, separated by ","'
    )
    parser.add_argument(
        "--interval",
        type=str,
        required=False,
        help='interval of checkpoints at which to evaluate FIDs, separated by ","'
    )
    parser.add_argument(
        "--fastgan",
        action="store_true",
        help="if the generator is FastGAN"
    )
    parser.add_argument(
        "--n_mlp",
        type=int,
        default=8
    )
    parser.add_argument(
        "ckpt", metavar="CHECKPOINT", help="path to generator checkpoint"
    )

    args = parser.parse_args()

    if args.in_chunk:
        dsets = args.datasets.split(",")
        ckpts = args.interval.split(",")
        ckpts = [i for i in range(int(ckpts[0]), int(ckpts[1])+10000, 10000)]
        ckpts = [str(c).zfill(6) for c in ckpts]
        print("dsets: ", dsets)
        print("ckpts: ", ckpts)
        fids = defaultdict(list)
        failures = defaultdict(list)

        if args.fastgan:
            print("Watch Out: Folder Names are DIFFERENT from STYLEGAN2!!")

        for dset in dsets:

            # print(f"{dset} in progress!")

            inception_dset = dset

            if dset[-2:] == "GD":
                inception_dset = dset[:-8]

            if args.sfid:
                args.inception = f"inception_features/s_inception/inception_{inception_dset}-full.pkl"
            else:
                args.inception = f"inception_features/inception/inception_{inception_dset}-full.pkl"

            for c in ckpts:

                if  args.fastgan:
                    g = FastGenerator(ngf=64, nz=256, nc=3, im_size=args.size).to(device) # TODO: altered to import FastGAN generator
                    c = c[1:]
                    checkpoint = f"../FastGAN-pytorch/train_results/{dset}/models/{c}.pth"

                    try:
                        ckpt = torch.load(checkpoint, map_location=lambda a,b: a)
                    except:
                        print(f"checkpoint '{checkpoint}' does not exist!")
                        failures[dset].append(c)
                        continue
                    g.load_state_dict(ckpt["g"])

                else:
                    try:
                        g = Generator(args.size, 512, 8).to(device)
                        checkpoint = f"experiments/{args.method}/{dset}/checkpoint/{c}.pt"

                        try:
                            ckpt = torch.load(checkpoint)
                        except:
                            print(f"checkpoint '{checkpoint}' does not exist!")
                            failures[dset].append(c)
                            continue
                        g.load_state_dict(ckpt["g_ema"])
                        print("n_mlp=8")
                    except:
                        g = Generator(args.size, 512, 2).to(device)
                        checkpoint = f"experiments/{args.method}/{dset}/checkpoint/{c}.pt"

                        try:
                            ckpt = torch.load(checkpoint)
                        except:
                            print(f"checkpoint '{checkpoint}' does not exist!")
                            failures[dset].append(c)
                            continue
                        g.load_state_dict(ckpt["g_ema"])
                        print("n_mlp=2")


                g.eval()


                if not args.fastgan:
                    if args.truncation < 1:
                        with torch.no_grad():
                            mean_latent = g.mean_latent(args.truncation_mean)
                    else:
                        mean_latent = None

                if args.sfid:
                    inception = load_patched_inception_v3(2)
                else:
                    inception = load_patched_inception_v3(3)

                inception = inception.to(device)
                inception.eval()
                if args.fastgan:
                    features = extract_feature_from_fastgan(g, inception, args.batch, args.n_sample, device).numpy()
                else:
                    features = extract_feature_from_samples(
                        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
                    ).numpy()
                print(f"extracted {features.shape[0]} features from {dset}-{c}")

                sample_mean = np.mean(features, 0)
                sample_cov = np.cov(features, rowvar=False)

                with open(args.inception, "rb") as f:
                    embeds = pickle.load(f)
                    real_mean = embeds["mean"]
                    real_cov = embeds["cov"]

                fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                fids[dset].append(fid)
        print("="*40)
        for k in fids:
            print("(Failure cases)", k, failures[k])
        for k in fids:
            print(k, [f"{comp:.2f}" for comp in fids[k]])
        print("="*40)

    else:

        if not args.input_is_image:
            ckpt = torch.load(args.ckpt)

            g = Generator(args.size, 512, 8).to(device)
            g.load_state_dict(ckpt["g_ema"])
            # g = nn.DataParallel(g)
            g.eval()

        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g.mean_latent(args.truncation_mean)

        else:
            mean_latent = None

        if args.sfid:
            inception = load_patched_inception_v3(2)
        else:
            inception = load_patched_inception_v3(3)
        # inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception = inception.to(device)
        inception.eval()

        if args.input_is_image:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            dset = ImageFolder(args.image_path, transform=transform)
            loader = DataLoader(dset, batch_size=args.batch, num_workers=4)

            features = extract_features(loader, inception, device).numpy()
        else:
            features = extract_feature_from_samples(
                g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
            ).numpy()
        print(f"extracted {features.shape[0]} features")

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        with open(args.inception, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]

        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        if not args.sfid:
            print("fid:", fid)
        else:
            print(f"sfid: {fid:.3f}")