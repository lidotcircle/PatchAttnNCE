import os
import click
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import dnnlib
import legacy
import timm
from time import perf_counter
from tqdm import tqdm
from torch import nn


class efficientnet_lite0_features(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
        pretrained = nn.Module()
        pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2])
        pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
        pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
        pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
        self.pretrained = pretrained
    
    def forward(self, x):
        out0 = self.pretrained.layer0(x)
        out1 = self.pretrained.layer1(out0)
        out2 = self.pretrained.layer2(out1)
        out3 = self.pretrained.layer3(out2)
        out0 = out0.view(out0.size(0), -1)
        out1 = out1.view(out0.size(0), -1)
        out2 = out2.view(out0.size(0), -1)
        out3 = out3.view(out0.size(0), -1)
        return torch.cat([out0, out1, out2, out3], dim=1)

def project(
    G,
    D,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    feature_extrator: nn.Module,
    *,
    num_steps                  = 1000,
    initial_learning_rate      = 1,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    verbose                    = False,
    device: torch.device
):
    def logprint(*args):
        if verbose:
            print(*args)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = feature_extrator(target_images)

    pimg = torch.tensor(target_images, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([pimg], betas=(0.9, 0.999), lr=initial_learning_rate)

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        synth_images = G(pimg)
        if synth_images.shape[2] != 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = feature_extrator(synth_images)
        gen_logits = D(synth_images)
        loss_adv = (-gen_logits).mean()
        loss_perc = (target_features - synth_features).square().mean()
        loss = loss_adv + loss_perc
        loss.backward()

        # Step
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f}, loss_adv {float(loss_adv):5.2f}, loss_perc {float(loss_perc):5.2f}')

    return pimg.detach()

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dataroot', help='Network pickle filename', required=True)
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=10000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    dataroot: str,
    outdir: str,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        pickle_net = legacy.load_network_pkl(fp)
        G = pickle_net['G_ema'].eval().requires_grad_(False).to(device) # type: ignore
        D = pickle_net['D'].eval().requires_grad_(False).to(device) # type: ignore

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
    eval_set_kwargs.max_dataset_size = 10000
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs)
    feature_extrator = efficientnet_lite0_features().to(device).eval().requires_grad_(False)

    for i, imgs in tqdm(enumerate(eval_set)):
        if i > len(eval_set):
            break

        img: torch.Tensor = imgs['B'].to(device)
        img_path = imgs['B_paths']

        # Optimize projection.
        start_time = perf_counter()
        out_img = project(
            G, D,
            target=img,
            feature_extrator=feature_extrator,
            num_steps=num_steps,
            device=device,
            verbose=True
        )
        out_img = (out_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(out_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{os.path.basename(img_path)}', quality=100, subsampling=0)
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
