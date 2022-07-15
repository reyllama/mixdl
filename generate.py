import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

def interpolate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(100*args.pics)):
            sample_z = torch.randn(2, args.latent, device=device)
            sample, w = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent, return_latents=True
            )
            w1, w2 = w[0].unsqueeze(0), w[1].unsqueeze(0)
            sample_w = torch.cat([torch.lerp(w1, w2, v) for v in torch.linspace(0, 1, 8, device=device)], dim=0)
            sample, _ = g_ema(
                [sample_w], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
            )


            utils.save_image(
                sample,
                args.root+f"/sample/interp/{str(i+1).zfill(6)}.png",
                nrow=8,
                normalize=True,
                range=(-1,1)
            )

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(args.sample)/4)
        # a = None
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            alpha = dirichlet.sample((args.sample, ))
            alpha[0] = torch.ones(args.sample)/4
            # print(alpha)
            alpha = alpha.to(device)
            w = g_ema.get_latent(sample_z)
            sample1, _ = g_ema(
                [w], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
            )
            w = torch.matmul(alpha, w)
            sample2, _ = g_ema(
                [w], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
            )
            utils.save_image(
                torch.cat([sample1, sample2], dim=0),
                args.root+f"/sample/interpolation_final/{str(i+1).zfill(6)}.png",
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )
            # if a is None:
            #     a = alpha
            # else:
            #     a = torch.cat([a, alpha], dim=0)
        # print(a.mean(dim=0), a.var(dim=0))
        # print(alpha.detach())


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
        default=4,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
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
        "--twosample",
        type=str,
        default="True"
    )

    parser.add_argument(
        "--extrapolate",
        type=str,
        default='False'
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.root+"/"+args.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])
    g_ema.to(device)
    print("args: ", args.twosample)
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    if args.twosample != "True":
        print("Generate")
        generate(args, g_ema, device, mean_latent)
    else:
        print("Interpolate")
        interpolate(args, g_ema, device, mean_latent)