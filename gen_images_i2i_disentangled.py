# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import math
import os
import click
from tqdm import tqdm
import dnnlib
import PIL.Image
import torch
import legacy
from visual_utils import image_grid


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dataroot', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--alpha', help='alpha base for latent manipulation', type=float, default=0.2, show_default=True)
@click.option('--num_images', help='number of generated images', type=click.IntRange(min=1, max=100), default=10, show_default=True)
@click.option('--fixed_style', help='fixed style for all images', is_flag=True)
@click.option('--full_quality', help='save image without loss data', is_flag=True)
def generate_images(
    network_pkl: str,
    dataroot: str,
    outdir: str,
    alpha: float,
    num_images: int,
    fixed_style: bool,
    full_quality: bool
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)
    eval_set_kwargs = dnnlib.EasyDict()
    eval_set_kwargs.class_name = 'training.unaligned_dataset.createDataset'
    eval_set_kwargs.dataroot = dataroot
    eval_set_kwargs.dataname = os.path.basename(dataroot)
    eval_set_kwargs.phase = 'test'
    eval_set_kwargs.preprocess = 'resize'
    eval_set_kwargs.load_size = 256
    eval_set_kwargs.crop_size = 256
    eval_set_kwargs.flip = False
    eval_set_kwargs.serial_batches = False
    eval_set_kwargs.max_dataset_size = 10000
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs)

    latent_changes = []
    for i in range(min(G.latent_dim, 8)):
        delta = torch.zeros([G.latent_dim], device=device, dtype=torch.float)
        delta[i] = 1
        l, h = -num_images//2, num_images//2
        latent_changes.append(torch.stack([delta * alpha * j for j in range(l, h)], dim=0))

    if fixed_style:
        the_style = torch.randn([1, G.latent_dim]).to(device)

    # Generate images.
    for i, imgs in tqdm(enumerate(eval_set), total=len(eval_set)):
        if i > len(eval_set):
            break

        img = imgs['A'].to(device).unsqueeze(0)
        img_path = imgs['A_paths']

        content, style = G.encode(img)
        out_images = []
        for deltas in latent_changes:
            style = style.expand(len(deltas), -1)
            if fixed_style:
                style = the_style
            if isinstance(content, list):
                content = list(map(lambda c: c.expand(len(deltas), -1, -1, -1), content))
            else:
                content = content.expand(len(deltas), -1, -1, -1)
            style = style + deltas
            out = G.decode(content, style + deltas)
            out_images.append(torch.cat([img, out], dim=0))

        out = image_grid(out_images, len(latent_changes[0]) + 1)
        out = (out.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        extra_args = {}
        if full_quality:
            extra_args['quality'] = 100
            extra_args['subsampling'] = 0
        PIL.Image.fromarray(out.cpu().numpy(), 'RGB').save(f'{outdir}/{os.path.basename(img_path)}', **extra_args)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
