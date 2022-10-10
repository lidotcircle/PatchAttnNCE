# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import numpy as np
from typing import List, Union
import click
from tqdm import tqdm
import dnnlib
import PIL.Image
import torch
import legacy
from visual_utils import image_grid


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--alpha', help='alpha base for latent manipulation', type=float, default=0.2, show_default=True)
@click.option('--num_images', help='number of generated images', type=click.IntRange(min=1, max=100), default=10, show_default=True)
@click.option('--top_k', help='top k eigenvalues', type=click.IntRange(min=1, max=100), default=5, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    outdir: str,
    alpha: float,
    num_images: int,
    top_k: int
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    matrix_A: torch.Tensor = list(G.synthesis.init.parameters())[0]
    matrix_A = matrix_A.view(matrix_A.size(0), -1)
    matrix_ATA = torch.matmul(matrix_A, matrix_A.transpose(1, 0))
    eig_info = torch.eig(matrix_ATA, eigenvectors=True)
    eigenvectors = eig_info.eigenvectors
    eig_changes = []
    for i in range(eigenvectors.size(0)):
        if i >= top_k:
            break
        ev = eigenvectors[i]
        l, h = -num_images//2, num_images//2
        eig_changes.append(torch.stack([ev * alpha * j for j in range(l, h)], dim=0))

    # Generate images.
    for i, seed in tqdm(enumerate(seeds)):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).float()
        out_images = []
        for deltas in eig_changes:
            z = z.expand(len(deltas), -1)
            z = z + deltas
            out = G(z, c=None)
            out_images.append(out)

        out = image_grid(out_images, len(eig_changes[0]))
        out = (out.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(out.cpu().numpy(), 'RGB').save(f'{outdir}/{i}.png', quality=100, subsampling=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
