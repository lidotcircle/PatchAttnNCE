# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import io
import os
import pickle
import click
from tqdm import tqdm
import dnnlib
import PIL.Image
import torch
import legacy


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
    if 'munit' in network_pkl:
        with io.open(network_pkl, 'rb') as pf:
            G = pickle.load(pf).requires_grad_(False).eval().to(device)
            def new_forward(img):
                out, _ = G.inference({'images_a': img, 'key': { 'images_a': { 'filename': '' }}}, random_style=True)
                return out
            G.forward = new_forward
    elif 'unit' in network_pkl:
        with io.open(network_pkl, 'rb') as pf:
            G = pickle.load(pf).requires_grad_(False).eval().to(device)
            def new_forward(img):
                out, _ = G.inference({'images_a': img, 'key': { 'images_a': { 'filename': [''], 'sequence_name': [''] }}})
                return out
            G.forward = new_forward
    elif 'GANsNRoses' in network_pkl or 'gansnroses' in network_pkl:
        with io.open(network_pkl, 'rb') as pf:
            G = pickle.load(pf, fix_imports=True).requires_grad_(False).eval().to(device)
            old_forward = G.forward
            def new_forward(img):
                out, _, _ = old_forward(img)
                return out
            G.forward = new_forward
    else:
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
    eval_set_kwargs.serial_batches = True
    eval_set_kwargs.max_dataset_size = 1000000
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs)

    # Generate images.
    for i, imgs in tqdm(enumerate(eval_set), total=len(eval_set)):
        if i > len(eval_set):
            break

        img = imgs['A'].to(device).unsqueeze(0)
        img_path = imgs['A_paths']

        out_img = G(img)
        out_img = (out_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(out_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{i:05}-{os.path.basename(img_path)}', quality=100, subsampling=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
