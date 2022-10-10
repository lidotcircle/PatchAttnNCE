# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from typing import List, Optional, Tuple, Union
import click
from tqdm import tqdm
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from visual_utils import save_image, image_grid


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dataroot', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    dataroot: str,
    outdir: str,
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
    dataloader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=2)

    # Generate images.
    for i, imgs in tqdm(enumerate(dataloader)):
        if i > len(eval_set):
            break

        img = imgs['A'].to(device)
        img_path = imgs['A_paths'][0]

        content, style = G.encode(img)
        style_ex = torch.zeros_like(style)
        style_ex[0] = style[1]
        style_ex[1] = style[0]
        out_img = G.decode(content, style)
        out_img2 = G.decode(content, style_ex)
        out = image_grid([img, out_img, out_img2], 2)
        out = (out.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(out.cpu().numpy(), 'RGB').save(f'{outdir}/{os.path.basename(img_path)}', quality=100, subsampling=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
