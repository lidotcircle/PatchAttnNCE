import os
import glob
from PIL import Image
import argparse
from typing import List, Tuple
from tqdm import tqdm


def resize_images(dir: str, target: Tuple[int,int], suffix: List[str]):
    width, height = target
    target_dir = os.path.join(dir, f'{width}x{height}')
    os.makedirs(target_dir, exist_ok=True)
    path = []
    for s in suffix:
        images = os.path.join(dir, f'*.{s}')
        path += glob.glob(images)

    for image in tqdm(path, desc=f'{dir}'):
        img = Image.open(image).convert('RGB')
        target_path = os.path.join(target_dir, os.path.basename(image))
        img = img.resize((width, height))
        img.save(target_path, quality=100, subsampling=0)


parser = argparse.ArgumentParser(description='Scale Images')
parser.add_argument('-t', '--target',      type=str, help='target image size [width x height]', default='256x256')
parser.add_argument('-d', '--directories', type=str, help='directories separated by comma', required=True)
parser.add_argument('-s', '--image_suffix', type=str, help='image file suffix separated by comma', default='png,jpg,jpeg')

if __name__ == "__main__":
    args = parser.parse_args()
    target: str = args.target
    size_pair = target.split('x')
    width = int(size_pair[0])
    height = int(size_pair[1])
    assert len(size_pair) == 2

    str_dirs: str = args.directories
    dirs = str_dirs.split(',')

    str_suffix: str = args.image_suffix
    suffix = str_suffix.split(',')
    suffix = list(map(lambda s: s.strip(), suffix))
    for dir in dirs:
        resize_images(dir, (width, height), suffix)
