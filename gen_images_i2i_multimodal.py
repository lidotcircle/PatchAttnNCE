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
from models.gaussian_vae import gaussian_reparameterization


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dataroot', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num_per_image', help='generator multiple image for one input', metavar='INT', type=click.IntRange(min=1), default=5)
@click.option('--same_style', help='using same style for all images', is_flag=True)
def generate_images(
    network_pkl: str,
    dataroot: str,
    outdir: str,
    num_per_image: int,
    same_style: bool,
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

    latent_dim: int = G.latent_dim
    latens = [ torch.randn([1, latent_dim]).to(device) for _ in range(num_per_image) ]
    # Generate images.
    for i, imgs in tqdm(enumerate(eval_set), total=len(eval_set)):
        if i > len(eval_set):
            break

        if not same_style:
            latens = [ torch.randn([1, latent_dim]).to(device) for _ in range(num_per_image) ]

        img = imgs['A'].to(device).unsqueeze(0)
        img_path = imgs['A_paths']

        content, style = G.encode(img)
        if hasattr(G, 'variational_style_encoder') and G.variational_style_encoder:
            style = gaussian_reparameterization(style[:,:style.size(1)//2], style[:,style.size(1)//2:])
        o1 = G.decode(content, style)
        out_images = [ img, o1 ]
        for lat in latens:
            out_images.append(G.decode(content, lat))

        out = image_grid([ torch.cat(out_images, dim=0) ], math.ceil(math.sqrt(len(out_images))))
        out = (out.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(out.cpu().numpy(), 'RGB').save(f'{outdir}/{os.path.basename(img_path)}', quality=100, subsampling=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
