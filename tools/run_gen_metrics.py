import io
import os
import re
import sys
import click
from subprocess import Popen, PIPE
from tqdm import tqdm


def run_task(model_path: str, dataset_path: str, outdir: str) -> bool:
    metrics_file = os.path.join(outdir, "metrics.txt")
    if os.path.exists(metrics_file):
        return True

    old_pythonpath = ''
    if 'PYTHONPATH' in os.environ:
        old_pythonpath = ';' + os.environ['PYTHONPATH']

    env = {**os.environ}
    env["PYTHONPATH"] =  f"{os.path.join(os.path.curdir, 'thirdparty/GANsNRoses')};{os.path.join(os.path.curdir, 'thirdparty/imaginaire')}{old_pythonpath}"

    args = [
                sys.executable, 'gen_images_i2i.py', 
                f"--network={model_path}",
                f"--dataroot={dataset_path}",
                f"--outdir={outdir}",
            ]
    gen_image_proc = Popen(args, env=env, stdout=sys.stdout, stderr=sys.stderr, text=True)
    gen_image_proc.wait()
    if gen_image_proc.returncode != 0:
        os.makedirs(outdir, exist_ok=True)
        with io.open(metrics_file, 'w') as f:
            f.write('')
        return False

    metrics_args = [
                sys.executable, 'calc_metrics_path.py', 
                "--device", "cuda:0",
                "--fake", outdir,
                "--real", os.path.join(dataset_path, "testB"),
            ]
    lines  = []
    metrics_proc = Popen(metrics_args, stdout=PIPE, stderr=sys.stderr, text=True)
    while metrics_proc.poll() is None:
        try:
            l = metrics_proc.stdout.readline()
            if l is not None:
                lines.append(l)
        except:
            pass
    with io.open(metrics_file, "w") as f:
        f.writelines(filter(lambda line: 'FID' in line, lines))
    return metrics_proc.returncode == 0
 

@click.command()
@click.option('--dataset_list', help='dataset list seperated by \';\'', required=True)
@click.option('--outdir',  help='save dataset', type=str, required=True, metavar='DIR')
@click.option('--model_search_dirs', help='paths for recursively search models, seperated by \';\'',  required=True)
@click.option('--valid_model_regex', help='regex for matching valid models path', default='.*(pkl|pt)', show_default=True)
def run_gen_metrics(
    dataset_list: str,
    outdir: str,
    model_search_dirs: str,
    valid_model_regex: str,
):
    datasets_dict = {}
    for (name, dir) in map(lambda dir: (os.path.basename(dir), dir), dataset_list.split(';')):
        datasets_dict[name] = dir

    valid_model_matcher = re.compile(valid_model_regex)
    models = []
    for dir in model_search_dirs.split(';'):
        for cur_dir, _, files in os.walk(dir):
            for file in files:
                if valid_model_matcher.match(file):
                    models.append(os.path.join(cur_dir, file))
    dataset_list_str = ','.join(datasets_dict.keys())
    print(f'Total number of models: {len(models)}, datasets: [{dataset_list_str}]')
    tasks = []
    for model in models:
        for ds in datasets_dict.keys():
            if ds in model:
                d1 = os.path.basename(model)
                d1 = os.path.splitext(d1)[0]
                d2 = os.path.basename(os.path.dirname(model))
                tasks.append((model, ds, f'{d2}-{d1}'))
                break
    dnames = list(map(lambda v: os.path.basename(v), dataset_list.split(';')))
    tasks.sort(key = lambda a: dnames.index(a[1]))
    print(f'Total number of tasks: {len(tasks)}')
    print()
    failed_n = 0
    for task in tqdm(tasks, desc='Tasks'):
        print(f"run {task[0]}")
        try:
            result = run_task(task[0], datasets_dict[task[1]], os.path.join(outdir, task[1], task[2]))
        except:
            result = False
        if not result:
            print(f'Task for model {task[0]} failed')
            failed_n += 1
    print(f"Fail: {failed_n}")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_gen_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------