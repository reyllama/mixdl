import argparse
import math
import random
import os

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import numpy

try:
    import wandb

except ImportError:
    wandb = None


from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def equidis_reg():
    # TODO!
    pass

def unif_norm(tens):
    assert len(tens.size()) == 2
    return tens / tens.sum(dim=1, keepdim=True)

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def generate(args, g_ema, device, mean_latent, idx):
    with torch.no_grad():
        g_ema.eval()
        j = 0
        n_ckpt = idx
        if os.path.exists(args.root+f"/sample/individual/{n_ckpt}/000001.png"):
            return
        if not os.path.exists(args.root+f"/sample/individual/{n_ckpt}"):
            os.makedirs(args.root+f"/sample/individual/{n_ckpt}")
        for i in tqdm(range(5000//8)):
            sample_z = torch.randn(8, 512, device=device)

            samples, _ = g_ema(
                [sample_z], truncation=1, truncation_latent=mean_latent, input_is_latent=False
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

def train(args, loader, generator, discriminator, extra, g_optim, d_optim, e_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    sfm = nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss()
    sim = nn.CosineSimilarity()
    l2 = nn.MSELoss()
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    # if args.augment and args.augment_p == 0:
    #     ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    # k = 1
    # d_ratio_mean = 0
    lowp, highp = 0, args.highp

    # equidistance_reg = True
    # adaptive = False
    # evaluate_interp = True

    distr = torch.distributions.dirichlet.Dirichlet(torch.ones(args.batch)/args.dir_div, validate_args=None)
    if args.gaussian:
        distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.batch), torch.eye(args.batch))
    if args.uniform:
        distr = torch.distributions.uniform.Uniform(0, 1)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)

        # Flags for interpolation and dynamic adv training
        which = i % args.interp_freq
        is_dynamic = (args.dynamic and i % args.dynamic_every==0)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        if which == 0:
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred, _ = discriminator(fake_img, extra=extra, flag=which, p_ind=numpy.random.randint(lowp, highp))  # TODO: __main__ should input extra and e_optim

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
        else:
            real_img_aug = real_img

        real_pred, _ = discriminator(real_img_aug, extra=extra, flag=which, p_ind=numpy.random.randint(lowp, highp), real=True)

        # w = generator.get_latent(z)  # (Batch, 512)
        # fake_img, _ = generator([w], return_feats=False, input_is_latent=True)

        if which > 0:
            if args.uniform:
                alpha = distr.sample((args.batch,args.batch)).to(device)
                alpha = unif_norm(alpha)
            else:
                alpha = distr.sample((args.batch,)).to(device)  # (Batch, Batch)
                if args.gaussian:
                    alpha = sfm(alpha)
            z = torch.randn(args.batch, args.latent, device=device)
            if args.workstation:
                w = generator.get_latent(z)  # (Batch, 512)
                fake_img, _ = generator([w], return_feats=False, input_is_latent=True)
            else:
                fake_img, w = generator([z], return_latents=True)  # (Batch, 512)
            w = torch.matmul(alpha, w)
            if args.workstation:
                interp_img, _ = generator([w], return_feats=False, input_is_latent=True)
            else:
                interp_img, _ = generator(w, return_feats=False, input_is_latent=True)
            if args.augment:
                interp_img, _ = augment(interp_img, ada_aug_p)
                fake_img, _ = augment(fake_img, ada_aug_p)
            inp_imgs = torch.cat([fake_img, interp_img], dim=0)
            fake_pred, _ = discriminator(interp_img, extra=extra, flag=which, p_ind=numpy.random.randint(lowp, highp), sim=False)
            _, sim_pred = discriminator(inp_imgs, extra=extra, flag=1-which, p_ind=numpy.random.randint(lowp, highp), sim=True)
            rel_loss = args.d_kl_wt * kl_loss(torch.log(sim_pred), sfm(alpha))

        d_loss = d_logistic_loss(real_pred, fake_pred)

        if which > 0:
            d_loss += rel_loss

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        extra.zero_grad()
        d_loss.backward()
        d_optim.step()
        e_optim.step()

        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(
                real_img, extra=extra, flag=which, p_ind=numpy.random.randint(lowp, highp))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(extra, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        if which == 0:
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred, _ = discriminator(fake_img, extra=extra, flag=which, p_ind=numpy.random.randint(lowp, highp))

        elif which > 0:
            # alpha = distr.sample((args.batch,)).to(device)  # (Batch, Batch)
            if args.uniform:
                alpha = distr.sample((args.batch,args.batch)).to(device)
                alpha = unif_norm(alpha)
            else:
                alpha = distr.sample((args.batch,)).to(device)  # (Batch, Batch)
                if args.gaussian:
                    alpha = sfm(alpha)
            z = torch.randn(args.batch, args.latent, device=device)
            w = generator.get_latent(z)  # (Batch, 512)
            w = torch.matmul(alpha, w)
            interp_img, _ = generator([w], return_feats=False, input_is_latent=True)
            if args.augment:
                interp_img, _ = augment(interp_img, ada_aug_p)
            fake_pred, _ = discriminator(interp_img, extra=extra, flag=which, p_ind=numpy.random.randint(lowp, highp))

        g_loss = g_nonsaturating_loss(fake_pred)


        #############################
        # TODO: Reset Var Name

        if which > 0:

            z = torch.randn(args.batch, args.latent, device=device)

            dist_source = sfm(alpha) # (2, K)

            feat_ind = numpy.random.randint(1, generator.num_layers - 1, size=args.batch) # (Select layer idx to extract activation from)

            # computing distances among target generations
            w = generator.get_latent(z) # (Batch, 512)
            sample_w = torch.matmul(alpha, w) # (Batch, 512)

            img_source, feat_source = generator([w], return_feats=True, input_is_latent=True)        # Edge
            img_target, feat_target = generator([sample_w], return_feats=True, input_is_latent=True) # Interpolated
            # with torch.no_grad():
            #     d1, d2 = discriminator(img_target[0].unsqueeze(0)), discriminator(img_target[1].unsqueeze(0))
            #     d_ratio = d1.mean() / (d2.mean()+1e-8)
            #     d_ratio_mean = ((d_ratio+4*d_ratio_mean)/5).item()
            dist_target = torch.zeros([args.batch, args.batch]).cuda()

            # iterating over different elements in the batch
            for pair1 in range(args.batch):
                for pair2 in range(args.batch):
                    anchor_feat = torch.unsqueeze(
                        feat_target[feat_ind[pair1]][pair1].reshape(-1), 0)
                    compare_feat = torch.unsqueeze(
                        feat_source[feat_ind[pair1]][pair2].reshape(-1), 0)
                    dist_target[pair1, pair2] = sim(anchor_feat, compare_feat)
            dist_target = sfm(dist_target)
            rel_loss = args.kl_wt * \
                kl_loss(torch.log(dist_target), dist_source) # distance consistency loss
            g_loss = g_loss + rel_loss

        ###############################


        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # if adaptive:
        #     if d_ratio_mean > 2 and k<5:
        #         print(f"Increase k to {k+1} at iter {i}, d_ratio_mean {d_ratio_mean:.2f}")
        #         k += 1
        #         d_ratio_mean = 0

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 500 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z], return_latents=False)
                    z = torch.randn(2, args.latent, device=device)
                    w = g_ema.get_latent(z)
                    w1, w2 = w[0].unsqueeze(0), w[1].unsqueeze(0)
                    sample_w = torch.cat(
                        [torch.lerp(w1, w2, v) for v in torch.linspace(0, 1, 8, device=device)], dim=0)
                    interp, _ = g_ema(
                        [sample_w], input_is_latent=True
                    )
                    utils.save_image(
                        sample,
                        args.root+f"/sample/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        interp,
                        args.root + f"/sample/interpolationW/{str(i).zfill(6)}.png",
                        nrow=int(8),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i > 0 and i % args.save_every == 0:
                generate(args, g_ema, device, mean_latent, i)
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    args.root + f"/checkpoint/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--root", type=str, help="root directory of the experiment")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=150001, help="total training iterations"
    )
    parser.add_argument("--save_every", default=10000, type=int, help="when to save model checkpoints")
    parser.add_argument(
        "--batch", type=int, default=4, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--interp_freq", type=int, default=2, help='Interval of interpolation during training'
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=25,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument("--highp", type=int, default=4)
    parser.add_argument("--dir_div", type=float, default=1.0)
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation"
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--kl_wt",
        type=float,
        default=1000.0,
        help="Weight of KL loss term"
    )
    parser.add_argument(
        "--dynamic",
        type=str,
        default="False",
        help="Whether to use dynamic adversarial training scheme"
    )
    parser.add_argument(
        "--dynamic_every",
        type=int,
        default=2,
        help="interval of applying dynamic adv training"
    )
    parser.add_argument(
        "--n_mlp",
        type=int,
        default=8,
        help="number of fc layers in mapping network"
    )
    parser.add_argument(
        "--d_kl_wt",
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--gaussian',
        action='store_true',
        help='whether to sample interpolation coefficient from standard gaussian distribution'
    )
    parser.add_argument(
        '--uniform',
        action='store_true',
        help='whether to sample interpolation coefficient from uniform distribution'
    )
    parser.add_argument(
        '--workstation',
        action='store_true',
        help='whether we are working on the workstation'
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model_cdc import Generator, Extra
        from model_cdc import Patch_Discriminator as Discriminator
        # from model import Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    extra = Extra().to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    a_reg_ratio = 0.1

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    e_optim = optim.Adam(
        extra.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.root+"/"+args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.root+"/"+args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")
    mean_latent = None
    train(args, loader, generator, discriminator, extra, g_optim, d_optim, e_optim, g_ema, device)
