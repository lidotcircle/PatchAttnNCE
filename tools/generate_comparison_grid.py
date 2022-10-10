from functools import reduce
import os
import glob
import numpy as np
from PIL import Image
import argparse
from typing import List
from tqdm import tqdm


def get_image_files(dir: str, suffix: List[str]):
    path = []
    for s in suffix:
        images = os.path.join(dir, f'*.{s}')
        path += glob.glob(images)
    return path

parser = argparse.ArgumentParser(description='Scale Images')
parser.add_argument('-d', '--directories', nargs='+', help='image directories', required=True)
parser.add_argument('-i', '--index', type=str, help='index', default=None)
parser.add_argument('-g', '--gap', type=int, help='gap between images in pixel', default=20)
parser.add_argument('-n', '--count', type=int, help='number of images', default=10)
parser.add_argument('-s', '--image_suffix', type=str, help='image file suffix separated by comma', default='png,jpg,jpeg')
parser.add_argument('--resize', action='store_true', help='resize to first image')
parser.add_argument('-o', '--output', type=str, help='output path', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    dirs: List[str] = args.directories
    str_suffix: str = args.image_suffix
    suffix = str_suffix.split(',')

    images_files = list(map(lambda d: get_image_files(d, suffix), dirs))
    for files in images_files:
        files.sort(key=lambda v: os.path.basename(v).split("-")[-1])
    min_len = reduce(lambda a, b: min(a, len(b)), images_files, 10000000000)
    max_len = reduce(lambda a, b: max(a, len(b)), images_files, 0)
    if min_len != max_len:
        print("not all directories contain the same number of files")
        exit(1)
    
    rows = []
    count = args.count
    gap = args.gap
    index = list(map(lambda x: int(x), args.index.split(","))) if args.index else np.random.permutation(min_len)[:count]
    count = len(index)
    for n in index:
        rows.append(list(map(lambda l: l[n], images_files)))

    w, h = Image.open(rows[0][0]).size
    result_image = Image.new("RGBA", (w * len(images_files) + gap * (len(images_files) - 1), h * count + gap * (count - 1)), (255,255,255,0))

    with tqdm(total=len(rows[0]) * count) as pbar:
        for i, l in enumerate(rows):
            for j, f in enumerate(l):
                img = Image.open(f)
                if img.size != (w, h):
                    if args.resize:
                        img = img.resize((w, h))
                    else:
                        print(f"size (width, height) of image {f} isn't ({w}, {h}), but get {img.size}")
                        exit(1)
                result_image.paste(img, (j*(w+gap), i*(h+gap)))
                pbar.update(1)
    
    result_image.save(args.output)
    print(index)
    print(f"write to {args.output}")