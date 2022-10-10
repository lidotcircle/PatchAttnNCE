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
import torch.nn.functional as FF
import dnnlib
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models import losses
from models.patchnce import PatchNCELoss
from torch.nn.parameter import Parameter
from models.transtyle import Transtyle


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps

    def forward(self, f1, f2):
        f1xf2 = (f1 * f2).sum(dim=1)
        f1_ = f1.pow(2).sum(dim=1).pow(0.5)
        f2_ = f2.pow(2).sum(dim=1).pow(0.5)
        cosine = f1xf2 / (f1_ * f2_ + self.eps)
        loss = 1 - cosine 
        return loss.mean()


class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ECUTPreStyleLoss(Loss):
    def __init__(self, device, G, D, F, G_ema, resolution: int,
                 nce_layers: list, feature_net: str, nce_idt: bool, num_patches: int,
                 adaptive_loss: bool, lambda_cosineSim: float = 1.0, arc_path: str = None,
                 style_extractor: str = None, lambda_GAN_random: float = 1,
                 lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_identity: float = 0,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        assert isinstance(G, Transtyle)
        assert isinstance(G_ema, Transtyle)
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.F = F
        self.resolution = resolution
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        self.lambda_GAN = lambda_GAN
        self.lambda_GAN_random = lambda_GAN_random
        self.lambda_NCE = lambda_NCE
        self.lambda_identity = lambda_identity
        self.lambda_cosineSim = lambda_cosineSim
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.adaptive_loss = adaptive_loss
        self.criterionIdt = torch.nn.MSELoss()
        self.cosineSim = CosineSimilarityLoss()
        self.criterionContrastiveNCE = losses.ContrastiveNCELoss()

        if style_extractor == 'arc':
            arc_checkpoint = torch.load(arc_path, map_location=torch.device("cpu"))
            self.netArc = arc_checkpoint['model'].module.to(self.device)
            self.netArc.eval()
            self.netArc.requires_grad_(False)
            self.F.se = torch.nn.Linear(512, G.num_style_outputs)

            def get_style(img):
                img_112 = FF.interpolate(img,size=(112,112), mode='bicubic')
                img_style = self.netArc(img_112)
                img_style = self.F.se(img_style)
                img_style = FF.normalize(img_style, p=2, dim=1)
                return img_style

            self.G_ema.styleformer = get_style
            self.G.styleformer = get_style
        else:
            self.netArc = None

        if feature_net == 'efficientnet_lite':
            self.netPre = losses.EfficientNetLite().to(self.device)
        elif feature_net == 'vgg16':
            self.netPre = losses.VGG16().to(self.device)
        elif feature_net == 'learned':
            self.netPre = self.G
        else:
            raise NotImplemented(feature_net)

        # define loss functions
        self.criterionNCE = []
        patchnce_opt = dnnlib.EasyDict(
            nce_includes_all_negatives_from_minibatch=False,
            batch_size=1,
            nce_T=0.07
        )

        self.nce_layers = nce_layers
        for _ in nce_layers:
            self.criterionNCE.append(PatchNCELoss(patchnce_opt).to(self.device))

        self.setup_F()
        self.F.train().requires_grad_(False).to(self.device)

    def setup_F(self):
        fimg = torch.empty([1, 3, self.resolution, self.resolution], device=self.device)
        feat = self.netPre(fimg, self.nce_layers, encode_only=True)
        if isinstance(feat, tuple):
            feat = feat[1]
        self.F.create_mlp(feat)
        if self.adaptive_loss:
            loss_weights = Parameter(torch.Tensor(len(feat)))
            loss_weights.data.fill_(1 / len(feat))
            self.F.loss_weights = loss_weights

    def calculate_NCE_loss(self, feat_net: torch.nn.Module, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = feat_net(tgt, self.nce_layers, encode_only=True)
        feat_k = feat_net(src, self.nce_layers, encode_only=True)
        if isinstance(feat_q, tuple):
            feat_q = feat_q[1]
        if isinstance(feat_k, tuple):
            feat_k = feat_k[1]

        feat_k_pool, sample_ids = self.F(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.F(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        if self.adaptive_loss:
            loss_weights = self.F.loss_weights
            posw = torch.abs(loss_weights)
            weights = posw / torch.sum(posw) + (1 / (5 * len(self.nce_layers)))
            weights = weights / torch.sum(weights)
        else:
            weights = [ 1 / n_layers for i in range(0, n_layers) ]

        for f_q, f_k, crit, weight, _ in zip(feat_q_pool, feat_k_pool, self.criterionNCE, weights, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean() * weight
        
        return total_nce_loss

    def run_G(self, real, gen_random_fake: bool=False, only_style: bool=False, update_emas=False):
        if self.netArc is not None:
            img_112 = F.interpolate(real,size=(112,112), mode='bicubic')
            img_latent = self.netArc(img_112)
            img_style = self.F.se(img_latent)
            img_style = F.normalize(img_style, p=2, dim=1)
        else:
            img_latent = None
            img_style = None

        if only_style:
            if img_style is None:
                img_style = self.G(real, only_style=True)

            return img_style, img_latent

        fake, img_style = self.G(real, in_styles=img_style, return_style=True)

        if gen_random_fake:
            if self.netArc is not None:
                random_img_style = torch.rand(img_style.shape).to(img_style.device)
                random_img_style = F.normalize(random_img_style, p=2, dim=1)
            else:
                random_img_style = self.G.styleformer.random_output(img_style.size(0), img_style.device)
            random_fake = self.G(real, in_styles=random_img_style)
        else:
            random_img_style = None
            random_fake = None

        return fake, img_style, random_fake, random_img_style , img_latent

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
                real = torch.cat([real_A, real_B], dim=0) if self.nce_idt or self.lambda_identity > 0 else real_A
                fake, real_style, random_fake, random_img_style, real_latent = self.run_G(real, gen_random_fake=self.lambda_GAN_random>0)
                # TODO
                fake_style, fake_latent = self.run_G(fake, only_style=True)
                fake_B = fake[:real_A.size(0)]
                fake_idt_B = fake[real_A.size(0):]
                gen_logits = self.run_D(fake_B, blur_sigma=blur_sigma)
                loss_Gmain_GAN = (-gen_logits).mean()
                loss_Gmain = self.lambda_GAN * loss_Gmain_GAN
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/gan', loss_Gmain_GAN)

                if self.lambda_GAN_random > 0:
                    random_gen_logits = self.run_D(random_fake, blur_sigma=blur_sigma)
                    loss_Gmain_GAN_random = (-random_gen_logits).mean()
                    loss_Gmain = self.lambda_GAN_random * loss_Gmain_GAN_random
                    training_stats.report('Loss/G/gan_random', loss_Gmain_GAN_random)
                    if real_latent is None:
                        random_fake_style_rec, _ = self.run_G(random_fake, only_style=True)
                        real_style = torch.cat([real_style, random_img_style], dim=0)
                        fake_style = torch.cat([fake_style, random_fake_style_rec], dim=0)

                if self.lambda_cosineSim > 0:
                    if real_latent is not None:
                        real_style = real_latent
                        fake_style = fake_latent
                        loss_Gmain_sim = self.cosineSim(fake_style, real_style)
                        loss_Gmain = loss_Gmain + self.lambda_cosineSim * loss_Gmain_sim
                        training_stats.report('Loss/G/cosineSimilarity', loss_Gmain_sim)
                    else:
                        loss_Gmain_sNCE = self.criterionContrastiveNCE(torch.cat([real_style, fake_style], dim=0))
                        loss_Gmain = loss_Gmain + self.lambda_cosineSim * loss_Gmain_sNCE
                        training_stats.report('Loss/G/StyleNCE', loss_Gmain_sNCE)

                if self.lambda_identity > 0:
                    loss_Gmain_idt = self.criterionIdt(fake_idt_B, real_B)
                    loss_Gmain = loss_Gmain + self.lambda_identity * loss_Gmain_idt
                    training_stats.report('Loss/G/identity', loss_Gmain_idt)

                if self.lambda_NCE > 0:
                    loss_Gmain_NCE = self.calculate_NCE_loss(self.netPre, real_A, fake_B)
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    if self.nce_idt:
                        loss_Gmain_NCE_idt = self.calculate_NCE_loss(self.netPre, real_B, fake_idt_B)
                        training_stats.report('Loss/G/NCE_idt', loss_Gmain_NCE_idt)
                        loss_Gmain_NCE = (loss_Gmain_NCE + loss_Gmain_NCE_idt) * 0.5
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE

                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _, _, _, _ = self.run_G(real_A, update_emas=True)
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
