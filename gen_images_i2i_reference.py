# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

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
@click.option('--serial', help='whether dataset use serial batches, fixed composition if set this flag', is_flag=True)
@click.option('--reverse_style_encoder', help='use B specific style encoder, this feature may not implement in all models', is_flag=True)
def generate_images(
    network_pkl: str,
    dataroot: str,
    outdir: str,
    serial: bool,
    reverse_style_encoder: bool
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
    eval_set_kwargs.serial_batches = serial
    eval_set_kwargs.max_dataset_size = 10000
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs)
    dataloader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=1)

    # Generate images.
    for i, imgs in tqdm(enumerate(dataloader)):
        if i > len(eval_set):
            break

        img_A = imgs['A'].to(device)
        img_B = imgs['B'].to(device)
        img_path = imgs['A_paths'][0]

        content, style_old = G.encode(img_A)
        _, style = G.reverse_se(img_B) if reverse_style_encoder else G.encode(img_B)

        if hasattr(G, 'variational_style_encoder') and G.variational_style_encoder:
            style_old_mu = style_old[:,:style_old.size(1)//2]
            style_old_logvar = style_old[:,style_old.size(1)//2:]
            style_old = gaussian_reparameterization(style_old_mu, style_old_logvar)
            style_mu = style[:,:style.size(1)//2]
            style_logvar = style[:,style.size(1)//2:]
            style = gaussian_reparameterization(style_mu, style_logvar)

        img = G.decode(content, style_old)
        out_img = G.decode(content, style)
        out = image_grid([img_A, img_B, out_img, img], 4)
        out = (out.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(out.cpu().numpy(), 'RGB').save(f'{outdir}/{os.path.basename(img_path)}', quality=100, subsampling=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
