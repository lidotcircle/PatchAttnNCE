# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
from hashlib import md5
import os
import click
import re
import json
import tempfile
import torch
from dnnlib.util import EasyDict
import legacy

import dnnlib
from models.utils import get_current_git_hash
from tdlogger import TdLogger
from training.traning_loop_i2i import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    logger = c.logger
    if isinstance(logger, EasyDict):
        logger = TdLogger(**logger)
        c.logger = logger

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop(rank=rank, **c)


def launch_training(c, exconf, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)
    logger: TdLogger = exconf.logger

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    matching_dirs = [re.fullmatch(r'\d{5}' + f'-{desc}', x) for x in prev_run_dirs if re.fullmatch(r'\d{5}' + f'-{desc}', x) is not None]
    if c.restart_every > 0 and len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:                     # fallback to standard
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.dataroot}')
    print(f'Dataset transform:   {c.training_set_kwargs.preprocess}')
    print()

    c.extra_info = dnnlib.EasyDict()
    c.extra_info.githash = get_current_git_hash()
    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)
    logger.send(json.dumps(c, indent=2), desc + "/training options", direct=True)
    c.logger = logger if c.num_gpus == 1 else exconf.logger_args
    c.desc = desc
    c.loss_kwargs.run_dir = c.run_dir

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


@click.command()

# Required.
@click.option('--name',         help='model name', metavar='DIR', required=True)
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--netG',         help='generator', type=str, default='models.cyclegan_networks.ResnetGenerator', show_default=True)
@click.option('--netD',         help='discriminator', type=str, default='pg_modules.discriminator.ProjectedDiscriminator', show_default=True)
@click.option('--netF',         help='minimize mutual information', type=str, default='models.cut_networks.PatchSampleF', show_default=True)
@click.option('--train_loss',   help='train loss', type=str, default='training.ecut_loss.ECUTLoss', show_default=True)
@click.option('--resume_ema',   help='using ema models to generate images, should only use on debug', is_flag=True)
@click.option('--gpuinfo_interval',  help='interval of sampling GPU information (Load , Memory Usage)', type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--dont_sample_grid',  help='don\'t sample image grid for saving time', is_flag=True)

# Generator
@click.option('--ngf',  help='base channel', type=click.IntRange(min=0, max=512), default=0)
@click.option('--latent_dim',  help='style generator: style latent dimension', type=click.IntRange(min=0, max=4096), default=0)
@click.option('--unet_layers',   help='UNet Layers', type=str, default='16,32,64', show_default=True)

# Style
@click.option('--style_recon_nce',   help='using NCE loss instead of MSELoss for style reconstruction', is_flag=True)
@click.option('--style_recon_force_idt',   help='identity reconstruction error between input style and style of generated images', is_flag=True)
@click.option('--style_recon_nce_mlp_layers',  help='map style code into latent space for computing NCE loss', type=click.IntRange(min=0, max=8), default=0)
@click.option('--randn_style',   help='sample style from normal distribution', is_flag=True)
@click.option('--same_style_encoder',   help='single style encoder for images of both domains', is_flag=True)
@click.option('--reverse_style_is_norm',   help='style of B domain should be standard gaussian distribution', is_flag=True)
@click.option('--lambda_style_KLD', type=float, default=0.0, show_default=True)
@click.option('--lambda_style_GAN', type=float, default=2.0, show_default=True)
@click.option('--lambda_style_recon', type=float, default=8.0, show_default=True)
@click.option('--shuffle_style',   help='shuffle style codes of one batch', is_flag=True)
@click.option('--normalize_style',   help='normalize style code with p=2', is_flag=True)

# PatchNCE
@click.option('--nce_layers',       help='feature layers',          type=str,        default=None,                 show_default=True)
@click.option('--patch_max_shape',  help='patch grid max shape, two integer separated by comma', type=str, default='256,256', show_default=True)
@click.option('--feature_net',      help='nce feature extraction network',          type=click.Choice(['efficientnet_lite', 'vgg16', 'learned']), default='vgg16',                 show_default=True)
@click.option('--nce_idt',          help='identity', is_flag=True)
@click.option('--nce_adaptive',     help='patchnce adaptive', is_flag=True)
@click.option('--num_patches',      help='number of negative patches',           metavar='INT',   type=click.IntRange(min=1), default=256)
@click.option('--nce_mlp_layers',   help='NCE mlp layers',           metavar='INT',   type=click.IntRange(min=0, max=10), default=2)
@click.option('--nce_sim_pnorm',    help='if greater than 0, using pnorm to compute similarity instead of consine similarity', type=float,  default=0.0, show_default=True)
@click.option('--feature_attn_layers', help='if greater thatn 0 then apply ViT to feature before PatchNCE', metavar='INT', type=click.IntRange(min=0, max=6), default=0)
@click.option('--weight_strategy',     help='weight strategy',          type=click.Choice(['linear_decay', 'half_learned']), default='linear_decay', show_default=True)

# SMap
@click.option('--attn_real_fake', help='use both real and fake features to compute attention', is_flag=True)
@click.option('--attn_layers',   help='smap attn layers',           metavar='INT',   type=click.IntRange(min=1, max=5), default=2)
@click.option('--map_layers',    help='smap map layers',           metavar='INT',   type=click.IntRange(min=1, max=5), default=2)
@click.option('--attn_net',      help='smap attn net type',          type=click.Choice(['conv1x1', 'resnet', 'none']), default='conv1x1', show_default=True)
@click.option('--map_net',      help='smap map net type',          type=click.Choice(['conv1x1', 'resnet']), default='resnet', show_default=True)
@click.option('--attn_temperature', help='smap attn temperature',  metavar='INT',   type=click.FloatRange(min=0.0001, max=10), default=8)
@click.option('--sigmoid_attn',          help='using sigmoid to normalize weights into [0,1]', is_flag=True)
@click.option('--attn_detach',          help='detach attention map', is_flag=True)
@click.option('--cam_attn_weight',      help='single attention map by weighting multiple layers', is_flag=True)

# Spatial-Correlative
@click.option('--sc_layers',       help='feature layers',          type=str,        default=None,                 show_default=True)
@click.option('--patch_size',      help='patch size',           metavar='INT',   type=click.IntRange(min=16), default=32)
@click.option('--sc_idt',          help='identity', is_flag=True)

# Transtyle
@click.option('--style_extractor',      help='image global style extractor',          type=click.Choice(['vit', 'cvt', 'arc']), default='vit',                 show_default=True)
@click.option('--arc_path',             help='arcface pretrained model path',         type=str, default=None, show_default=True)

# loss weight
@click.option('--lambda_GAN',         type=float,                   default=1.0, show_default=True)
@click.option('--lambda_GAN_random',  type=float,                   default=0.0, show_default=True)
@click.option('--lambda_wvar',        type=float,                   default=1.0, show_default=True)
@click.option('--lambda_abdis',       type=float,                   default=1.0, show_default=True)
@click.option('--lambda_NCE',         type=float,                   default=1.0, show_default=True)
@click.option('--lambda_PMap',         type=float,                   default=1.0, show_default=True)
@click.option('--lambda_SC',          type=float,                   default=1.0, show_default=True)
@click.option('--lambda_identity',    type=float,                   default=0.0, show_default=True)
@click.option('--lambda_cosineSim',   type=float,                   default=1.0, show_default=True)
@click.option('--lambda_classify',    type=float,                   default=1.0, show_default=True)
@click.option('--lambda_r1',          type=float,                   default=0.0, show_default=True)

@click.option('--linear_lr_decay', help='learning rate linear decay', is_flag=True)
@click.option('--d_reg_every', help='reg frequency',                metavar='INT',   type=click.IntRange(min=1), default=16)

# dataset
@click.option('--dataroot',         help='Training data',             metavar='[DIR]', type=str,                   required=True)
@click.option('--batch',            help='Total batch size',          metavar='INT',   type=click.IntRange(min=1), required=True)
@click.option('--preprocess',       help='image preprocess',          type=str,        default='resize',           show_default=True)
@click.option('--load_size',        help='image load size',           metavar='INT',   type=click.IntRange(min=1), default=256)
@click.option('--crop_size',        help='image size',                metavar='INT',   type=click.IntRange(min=1), default=256)
@click.option('--flip',             help='random flip dataset image', metavar='INT',   type=bool,                  default=False, show_default=True)
@click.option('--max_dataset_size', help='maximum dataset size',      metavar='INT',   type=click.IntRange(min=1), default=300000)
@click.option('--serial_batches',   help='serial batch',              metavar='INT',   type=bool,                  default=False, show_default=True)

# logger
@click.option('--logger_endpoint',  help='logger endpoint',                                           type=str , default="http://192.168.44.43:5445")
@click.option('--logger_queuesize', help='logger message queue size, for reducing request frequency', type=click.IntRange(min=1) , default=10)
@click.option('--logger_prefix',    help='logger group prefix',                                       type=str,  default="", show_default=True)
@click.option('--logger_priority',  help='enable logger message priority',                            type=bool, default=False, show_default=True)
@click.option('--disable_logger',          help='disable logger', is_flag=True)

# Optional features.
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)

# Misc hyperparameters.
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0), default=0.0001)
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.0001, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--metrics_ticks',help='How often to eval metrics', metavar='TICKS',              type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)
@click.option('--restart_every',help='Time interval in seconds to restart code', metavar='INT', type=int, default=9999999, show_default=True)


def main(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=64, w_dim=128, mapping_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # logger set
    logger_args = dnnlib.EasyDict(endpoint=opts.logger_endpoint, default_group="i2i", bufferQueueSize=opts.logger_queuesize, group_prefix=opts.logger_prefix, credential=('admin', '123456'), disabled=opts.disable_logger)
    logger = TdLogger(**logger_args)
    exconf = dnnlib.EasyDict(logger = logger, logger_args = logger_args)

    # Training set.
    c.training_set_kwargs = dnnlib.EasyDict(
        class_name='training.unaligned_dataset.createDataset',
        phase='train',
        dataroot=opts.dataroot, preprocess=opts.preprocess, load_size=opts.load_size,
        crop_size=opts.crop_size, flip=opts.flip, max_dataset_size=opts.max_dataset_size,
        serial_batches=opts.serial_batches)
    c.training_set_kwargs.dataname = os.path.basename(opts.dataroot)

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = 2
    c.G_opt_kwargs.lr = opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.metrics_ticks = opts.metrics_ticks
    c.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.linear_lr_decay = opts.linear_lr_decay

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs = dnnlib.EasyDict(class_name=opts.netg)
    c.G_kwargs.style_extractor = opts.style_extractor
    c.G_kwargs.variational_style_encoder = opts.lambda_style_kld > 0
    if opts.netg == 'models.cyclegan_networks.ResnetGenerator':
        c.G_kwargs.input_nc = 3
        c.G_kwargs.output_nc = 3
    
    if opts.ngf > 0:
        c.G_kwargs.ngf = opts.ngf
    if opts.latent_dim > 0:
        c.G_kwargs.latent_dim = opts.latent_dim
    c.G_kwargs.unet_layers = list(map(lambda x: int(x), opts.unet_layers.split(','))) 
    c.G_kwargs.normalize_style = opts.normalize_style
    
    c.F_kwargs = dnnlib.EasyDict(class_name=opts.netf)
    c.F_kwargs.use_mlp = opts.nce_mlp_layers != 0
    c.F_kwargs.mlp_layers = opts.nce_mlp_layers
    p_mh, p_mw = opts.patch_max_shape.split(',')
    p_mh, p_mw = int(p_mh), int(p_mw)
    c.F_kwargs.max_shape = (p_mh, p_mw)
    c.F_kwargs.attn_net = opts.attn_net
    c.F_kwargs.map_net = opts.map_net
    c.F_kwargs.attn_layers = opts.attn_layers
    c.F_kwargs.map_layers = opts.map_layers
    c.F_kwargs.attn_temperature = opts.attn_temperature

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ema_rampup = None  # Disable EMA rampup.

    # Restart.
    c.restart_every = opts.restart_every

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    confjsonstr = json.dumps(opts, indent=2)
    conf_digest = md5(bytes(confjsonstr, 'utf8')).digest().hex()
    conf_digest = conf_digest[:6]
    desc = f'{opts.name}-{conf_digest}-{c.training_set_kwargs.dataname:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Projected and Multi-Scale Discriminators
    c.loss_kwargs = dnnlib.EasyDict(class_name=opts.train_loss)
    if opts.nce_layers is None:
        if opts.feature_net == 'efficientnet_lite':
            opts.nce_layers = '1,2,3,4,5,6' if opts.nce_adaptive else '2,4,6'
        elif opts.feature_net == 'vgg16':
            opts.nce_layers = '1,2,3,4,5,6,7,8,9,10' if opts.nce_adaptive else '4,7,9'
        elif opts.feature_net == 'learned':
            opts.nce_layers = '0,2,4,6,8,9,10,12,14,16' if opts.nce_adaptive else  '0,4,8,12,16'
    if opts.sc_layers is None:
        opts.sc_layers = opts.nce_layers
    c.loss_kwargs.nce_layers = [ int(val) for val in opts.nce_layers.split(',') ]
    c.loss_kwargs.sc_layers = [ int(val) for val in opts.sc_layers.split(',') ]
    c.loss_kwargs.feature_net = opts.feature_net
    c.loss_kwargs.nce_idt = opts.nce_idt
    c.loss_kwargs.feature_attn_layers = opts.feature_attn_layers
    c.loss_kwargs.normalize_transformer_out = opts.nce_mlp_layers == 0
    c.loss_kwargs.patch_max_shape = c.F_kwargs.max_shape
    c.loss_kwargs.sc_idt = opts.sc_idt
    c.loss_kwargs.adaptive_loss = opts.nce_adaptive
    c.loss_kwargs.num_patches = opts.num_patches
    c.loss_kwargs.patch_size = opts.patch_size
    c.loss_kwargs.sim_pnorm = opts.nce_sim_pnorm
    c.loss_kwargs.lambda_style_KLD = opts.lambda_style_kld
    c.loss_kwargs.lambda_style_GAN = opts.lambda_style_gan
    c.loss_kwargs.lambda_style_recon = opts.lambda_style_recon
    c.loss_kwargs.lambda_GAN = opts.lambda_gan
    c.loss_kwargs.lambda_GAN_random = opts.lambda_gan_random
    c.loss_kwargs.lambda_NCE = opts.lambda_nce
    c.loss_kwargs.lambda_wvar = opts.lambda_wvar
    c.loss_kwargs.lambda_abdis = opts.lambda_abdis
    c.loss_kwargs.lambda_PMap = opts.lambda_pmap
    c.loss_kwargs.lambda_SC = opts.lambda_sc
    c.loss_kwargs.lambda_identity = opts.lambda_identity
    c.loss_kwargs.lambda_cosineSim = opts.lambda_cosinesim
    c.loss_kwargs.lambda_classify = opts.lambda_classify
    c.loss_kwargs.lambda_r1 = opts.lambda_r1
    c.loss_kwargs.d_reg_every = opts.d_reg_every
    c.loss_kwargs.style_extractor = opts.style_extractor
    c.loss_kwargs.arc_path = opts.arc_path
    c.loss_kwargs.attn_net = opts.attn_net
    c.loss_kwargs.map_net = opts.map_net
    c.loss_kwargs.attn_real_fake = opts.attn_real_fake
    c.loss_kwargs.attn_layers = opts.attn_layers
    c.loss_kwargs.map_layers = opts.map_layers
    c.loss_kwargs.attn_temperature = opts.attn_temperature
    c.loss_kwargs.sigmoid_attn = opts.sigmoid_attn
    c.loss_kwargs.attn_detach = opts.attn_detach
    c.loss_kwargs.cam_attn_weight = opts.cam_attn_weight
    c.loss_kwargs.style_recon_nce = opts.style_recon_nce
    c.loss_kwargs.style_recon_force_idt = opts.style_recon_force_idt
    c.loss_kwargs.style_recon_nce_mlp_layers = opts.style_recon_nce_mlp_layers
    c.loss_kwargs.randn_style = opts.randn_style
    c.loss_kwargs.same_style_encoder = opts.same_style_encoder
    c.loss_kwargs.reverse_style_is_norm = opts.reverse_style_is_norm
    c.loss_kwargs.shuffle_style = opts.shuffle_style
    c.loss_kwargs.weight_strategy = opts.weight_strategy
    c.use_ema_model = opts.resume_ema
    c.gpuinfo_interval = opts.gpuinfo_interval
    c.sample_image_grid = not opts.dont_sample_grid

    c.D_kwargs = dnnlib.EasyDict(
        class_name=opts.netd,
        diffaug=True,
        interp224=False,
        backbone_kwargs=dnnlib.EasyDict(),
    )

    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.separable = False
    c.D_kwargs.backbone_kwargs.cond = False

    # Launch.
    launch_training(c=c, exconf=exconf, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

    # Check for restart
    last_snapshot = misc.get_ckpt_path(c.run_dir)
    if os.path.isfile(last_snapshot):
        # get current number of training images
        with dnnlib.util.open_url(last_snapshot) as f:
            cur_nimg = legacy.load_network_pkl(f)['progress']['cur_nimg'].item()
        if (cur_nimg//1000) < c.total_kimg:
            print('Restart: exit with code 3')
            exit(3)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
