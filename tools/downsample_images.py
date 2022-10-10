import os
import glob
from PIL import Image
import argparse
from typing import List
from tqdm import tqdm


def downsample(dir: str, suffix: List[str]):
    target_dir = os.path.join(dir, f'downsampled')
    os.makedirs(target_dir, exist_ok=True)
    path = []
    for s in suffix:
        images = os.path.join(dir, f'*.{s}')
        path += glob.glob(images)

    for image in tqdm(path, desc=f'{dir}'):
        img = Image.open(image).convert('RGB')
        target_path = os.path.join(target_dir, os.path.basename(image))
        img.save(target_path)


parser = argparse.ArgumentParser(description='Scale Images')
parser.add_argument('-d', '--directories', type=str, help='directories separated by comma', required=True)
parser.add_argument('-s', '--image_suffix', type=str, help='image file suffix separated by comma', default='png,jpg,jpeg')

if __name__ == "__main__":
    args = parser.parse_args()
    str_dirs: str = args.directories
    dirs = str_dirs.split(',')

    str_suffix: str = args.image_suffix
    suffix = str_suffix.split(',')
    suffix = list(map(lambda s: s.strip(), suffix))
    for dir in dirs:
        downsample(dir, suffix)
