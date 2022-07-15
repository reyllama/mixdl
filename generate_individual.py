import argparse
import os
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(args.sample)/args.dir_div)
        j = 0
        n_ckpt = str(int(args.ckpt.split("/")[-1][:-3]))
        if not os.path.exists(args.root + f"/sample/individual/{n_ckpt}"):
            os.makedirs(args.root + f"/sample/individual/{n_ckpt}")
        print(n_ckpt)
        for i in tqdm(range(args.pics//args.sample)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            if args.interpolate=="True":
                w = g_ema.get_latent(sample_z)
            # sample1, _ = g_ema(
            #     [w], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
            # )
                alpha = dirichlet.sample((1,))
                alpha = alpha.to(device)
                w = torch.matmul(alpha, w)
                samples, _ = g_ema(
                    [w], truncation=args.truncation, truncation_latent=None, input_is_latent=True
                )
                for sample in samples:
                    utils.save_image(
                        sample,
                        args.root + f"/sample/individual/interp_{n_ckpt}/{str(j + 1).zfill(6)}.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                    j += 1
            else:
                samples, _ = g_ema(
                    [sample_z], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=False
                )
                for sample in samples:
                    utils.save_image(
                        sample,
                        args.root+f"/sample/individual/{n_ckpt}/{str(j+1).zfill(6)}.png", # TODO: Alter file path
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                    j += 1




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--root", type=str, help="root directory"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=8,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=5000, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
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
    parser.add_argument(
        "--interpolate",
        type=str,
        default='False'
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3
    )
    parser.add_argument(
        "--dir_div",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--use_ema",
        type=str,
        default="True"
    )

    parser.add_argument(
        "--n_mlp",
        type=int,
        default=8
    )

    args = parser.parse_args()

    args.latent = 512
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.root+"/"+args.ckpt)
    if args.use_ema == "True":
        g_ema.load_state_dict(checkpoint["g_ema"])
        print("loading ema Generator")
    else:
        g_ema.load_state_dict(checkpoint["g"])
        print("loading normal Generator")
    if args.truncation < 1:
        print("Truncation Applied")
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        print("No truncation applied")
        mean_latent = None

    generate(args, g_ema, device, mean_latent)