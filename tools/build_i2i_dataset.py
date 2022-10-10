import os
import glob
import click
from shutil import copy2
from random import shuffle
from tqdm import tqdm


@click.command()
@click.option('--sourceA', help='images directory of A domain', required=True)
@click.option('--sourceB', help='images directory of B domain', required=True)
@click.option('--outdir',  help='save dataset', type=str, required=True, metavar='DIR')
@click.option('--num_trainA', help='number of training images of A domain', default=3000, show_default=True)
@click.option('--num_trainB', help='number of training images of B domain', default=3000, show_default=True)
@click.option('--num_testA', help='number of test images of A domain', default=100, show_default=True)
@click.option('--num_testB', help='number of test images of B domain', default=100, show_default=True)
@click.option('--no_shuffle', help='don\'t shuffle dataset', is_flag=True)
@click.option('--image_suffix', type=str, help='image file suffix separated by comma', default='png,jpg,jpeg', show_default=True)
def build_dataset(
    sourcea: str,
    sourceb: str,
    num_traina: int,
    num_trainb: int,
    num_testa: int,
    num_testb: int,
    outdir: str,
    no_shuffle: bool,
    image_suffix: str,
):
    os.makedirs(outdir, exist_ok=True)
    suffix = image_suffix.split(',')
    sourceA_paths = []
    sourceB_paths = []
    for s in suffix:
        A_images = os.path.join(sourcea, f'*.{s}')
        sourceA_paths += glob.glob(A_images)
        B_images = os.path.join(sourceb, f'*.{s}')
        sourceB_paths += glob.glob(B_images)

    if len(sourceA_paths) < num_traina + num_testa:
        raise f"not enough images in {sourcea}"

    if len(sourceB_paths) < num_trainb + num_testb:
        raise f"not enough images in {sourceb}"

    if not no_shuffle:
        shuffle(sourceA_paths)
        shuffle(sourceB_paths)
    
    trainA = sourceA_paths[:num_traina]
    testA = sourceA_paths[num_traina:num_traina+num_testa]
    trainB = sourceB_paths[:num_trainb]
    testB = sourceB_paths[num_trainb:num_trainb+num_testb]
    for files, dir in zip([trainA, testA, trainB, testB], ['trainA', 'testA', 'trainB', 'testB']):
        thedir = os.path.join(outdir, dir)
        os.makedirs(thedir, exist_ok=True)
        for f in tqdm(files, desc=f'{dir}'):
            copy2(f, thedir)
    print("DONE")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    build_dataset() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------