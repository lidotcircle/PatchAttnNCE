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
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models.pmap_networks import PMapNet
from models import losses


class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class PMapLoss(Loss):
    def __init__(self, device, G, D, F, G_ema, resolution: int,
                 feature_net: str, nce_idt: bool,
                 lambda_GAN: float=1.0, lambda_PMap: float=1.0, lambda_identity: float = 0,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.F: PMapNet = F
        assert isinstance(self.F, PMapNet)
        self.resolution = resolution
        self.pmap_idt = nce_idt
        self.lambda_GAN = lambda_GAN
        self.lambda_PMap = lambda_PMap
        self.lambda_identity = lambda_identity
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.criterionIdt = torch.nn.MSELoss()

        if feature_net == 'efficientnet_lite':
            self.netPre = losses.EfficientNetLite().to(self.device)
            self.feat_layers = [ 1,3,4,6 ]
        elif feature_net == 'vgg16':
            self.netPre = losses.VGG16().to(self.device)
            self.feat_layers = [ 3,4,5,7,8,9 ]
        elif feature_net == 'learned':
            self.netPre = self.G
            self.feat_layers = [ 3,4,5,7,8,9 ]
        else:
            raise NotImplemented(feature_net)

        self.setup_F()
        self.F.train().requires_grad_(False).to(self.device)

    def setup_F(self):
        fimg = torch.empty([1, 3, self.resolution, self.resolution], device=self.device)
        feat = self.netPre(fimg, self.feat_layers, encode_only=True)
        if isinstance(feat, tuple):
            feat = feat[1]

        attn = feat[len(feat)//2:]
        feat = feat[:len(feat)//2]
        self.F.setup(feat, attn)

    def calculate_PMap_loss(self, feat_net: torch.nn.Module, real, fake):
        real_feats = feat_net(real, self.feat_layers, encode_only=True)
        fake_feats = feat_net(fake, self.feat_layers, encode_only=True)
        if isinstance(real_feats, tuple):
            real_feats = real_feats[1]
        if isinstance(fake_feats, tuple):
            fake_feats = fake_feats[1]

        attn_feats = real_feats[len(real_feats)//2:]
        real_feats = real_feats[:len(real_feats)//2]
        fake_feats = fake_feats[:len(fake_feats)//2]
        return self.F.forward(real_feats, attn_feats, fake_feats)

    def run_G(self, real, update_emas=False):
        fake = self.G(real)
        return fake

    def run_D(self, img, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img)
        return logits

    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                real = torch.cat([real_A, real_B], dim=0) if self.pmap_idt or self.lambda_identity > 0 else real_A
                fake = self.run_G(real)
                fake_B = fake[:real_A.size(0)]
                fake_idt_B = fake[real_A.size(0):]
                gen_logits = self.run_D(fake_B, blur_sigma=blur_sigma)
                loss_Gmain_GAN = (-gen_logits).mean()
                loss_Gmain = self.lambda_GAN * loss_Gmain_GAN
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/gan', loss_Gmain_GAN)

                if self.lambda_identity > 0:
                    loss_Gmain_idt = self.criterionIdt(fake_idt_B, real_B)
                    loss_Gmain = loss_Gmain + self.lambda_identity * loss_Gmain_idt
                    training_stats.report('Loss/G/identity', loss_Gmain_idt)

                if self.lambda_PMap > 0:
                    loss_Gmain_PMap = self.calculate_PMap_loss(self.netPre, real_A, fake_B)
                    training_stats.report('Loss/G/PMap', loss_Gmain_PMap)
                    if self.pmap_idt:
                        loss_Gmain_PMap_idt = self.calculate_PMap_loss(self.netPre, real_B, fake_idt_B)
                        training_stats.report('Loss/G/PMap_idt', loss_Gmain_PMap_idt)
                        loss_Gmain_PMap = (loss_Gmain_PMap + loss_Gmain_PMap_idt) * 0.5
                    loss_Gmain = loss_Gmain + loss_Gmain_PMap

                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(real_A, update_emas=True)
                gen_logits = self.run_D(gen_img, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_B.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()
