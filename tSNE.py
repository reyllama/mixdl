import argparse

import numpy as np
from sklearn.manifold import TSNE

import torch
from model import Generator
from tqdm import tqdm
import matplotlib.pyplot as plt

def visualize_latent(args, g_ema, device):
    latents = []
    dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(args.n_interp))
    with torch.no_grad():
        for _ in tqdm(range(args.sample//args.n_interp)):
            latent = torch.randn(args.n_interp, args.latent, device=device)
            alpha = dirichlet.sample((args.n_interp,)).to(device)  # (Batch, Batch)
            if args.interpolate == "Z":
                latent = torch.matmul(alpha, latent)
            elif args.interpolate == "X":
                latent = g_ema.get_latent(latent)
            elif args.interpolate == "W":
                latent = g_ema.get_latent(latent)
                latent = torch.matmul(alpha, latent)
            else:
                raise NotImplementedError
            try:
                latents.append(latent.detach().cpu().numpy())
            except:
                latents.append(latent.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    latents = latents.astype(np.float32)
    X = TSNE(n_components=2).fit_transform(latents)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c="b", s=0.5)
    title_str = f"latent_{args.interpolate}"
    plt.axis('tight')

    plt.title(title_str)
    fig.savefig(args.root + "/" + title_str + '.png')


if __name__ == "__main__":
    device = "cpu"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--interpolate", type=str, default="W", help="Whether to interpolate in latent space"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=4096,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--n_interp",
        type=int,
        default=8,
        help="number of samples to inteprolate"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="experiments/",
        help="Root dir"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    args.size = 256

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.root+"/"+args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    mean_latent = None

    visualize_latent(args, g_ema, device)