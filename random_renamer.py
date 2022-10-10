import os
import random
import click


@click.command()
@click.option('--dir', help='directory to operate', metavar='DIR', required=True)
def main(dir: str):
    contents = os.listdir(dir)
    random.shuffle(contents)
    for i, file in enumerate(contents):
        fpath = os.path.join(dir, file)
        if not os.path.isfile(fpath):
            continue
        _, ext = os.path.splitext(file)
        fpath_to = os.path.join(dir, f'{i}{ext}')
        os.rename(fpath, fpath_to)
        print(f"{file} => {os.path.basename(fpath_to)}")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
