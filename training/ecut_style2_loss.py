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
from functools import reduce
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from torch import nn, autograd
import kornia.augmentation as K
import kornia
from models.transformer import VisioniTransformer
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models import losses
from models.patchnce import PatchNCELoss
from models.fastae_networks import Encoder as Ev1, Generator as Gv1
from models.fastae_v2_networks import Encoder as Ev2, Generator as Gv2
from models.fastae_v3_networks import Encoder as Ev3, Generator as Gv3
from models.fastae_v5_networks import Encoder as Ev9, Generator as Gv9
from models.fastae_v6_networks import Encoder as Ev10, Generator as Gv10
from models.fastae_v7_networks import Encoder as Ev11, Generator as Gv11
from models.fastae_v8_networks import Encoder as Ev12, Generator as Gv12
from models.style_networks import Encoder as Ev4, Generator as Gv4
from models.style_v2_networks import Encoder as Ev5, Generator as Gv5
from models.style_v3_networks import Encoder as Ev7, Generator as Gv7
from models.style_v4_networks import Encoder as Ev8, Generator as Gv8
from thirdparty.GANsNRoses.model import Encoder as Ev_gnr, Generator as Gv_gnr, LatDiscriminator

valid_gen_encoder = [
    (Gv1, Ev1),
    (Gv2, Ev2),
    (Gv3, Ev3),
    (Gv4, Ev4),
    (Gv5, Ev5),
    (Gv7, Ev7),
    (Gv8, Ev8),
    (Gv9, Ev9),
    (Gv10, Ev10),
    (Gv11, Ev11),
    (Gv12, Ev12),
    (Gv_gnr, Ev_gnr)
]

class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.mean(), inputs=real_img, create_graph=True, only_inputs=True
    )
    grad_real = grad_real.pow(2)
    return grad_real.mean() * (grad_real.numel() / real_img.size(0))


class ECUTStyle2Loss(Loss):
    def __init__(self, device, G, D, F, resolution: int,
                 nce_layers: list, feature_net: str, nce_idt: bool, num_patches: int,
                 style_recon_nce: bool = False, style_recon_force_idt: bool = False, feature_attn_layers: int=0, patch_max_shape: Tuple[int,int]=(256,256),
                 same_style_encoder: bool = False, normalize_transformer_out: bool = True, sim_pnorm: float = 0,
                 lambda_style_GAN: float=2.0, lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_identity: float = 0,
                 lambda_r1: float = 0, d_reg_every: int = 16, reverse_style_is_norm: bool=False,
                 lambda_style_consis: float=50.0, lambda_style_recon: float = 5,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        assert reduce(lambda a, b: a or b, map(lambda u: isinstance(G, u[0]), valid_gen_encoder))
        self.G: Gv1 = G
        self.D = D
        self.F = F
        self.resolution = resolution
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        self.feature_attn_layers = feature_attn_layers
        self.patch_max_shape = patch_max_shape
        self.normalize_transformer_out = normalize_transformer_out
        self.lambda_GAN = lambda_GAN
        self.lambda_style_GAN = lambda_style_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_identity = lambda_identity
        self.lambda_style_consis = lambda_style_consis
        self.lambda_style_recon = lambda_style_recon
        self.lambda_r1 = lambda_r1
        self.d_reg_every = d_reg_every
        self.reverse_style_is_norm = reverse_style_is_norm
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.criterionIdt = torch.nn.MSELoss()
        self.criterionStyleRecon = losses.ContrastiveNCELoss() if style_recon_nce else torch.nn.MSELoss()
        self.same_style_encoder = same_style_encoder
        self.style_recon_force_idt = style_recon_force_idt
        self.style_recon_nce = style_recon_nce
        self.latent_dim = self.G.latent_dim
        self.aug = nn.Sequential(
            K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
            kornia.geometry.transform.Resize(256+30),
            K.RandomCrop((256,256)),
            K.RandomHorizontalFlip(),
        )

        if feature_net == 'efficientnet_lite':
            self.netPre = losses.EfficientNetLite().to(self.device)
        elif feature_net == 'vgg16':
            self.netPre = losses.VGG16().to(self.device)
        elif feature_net == 'learned':
            self.netPre = self.G
            if isinstance(self.G, Gv3):
                nce_layers = [2,4,6,8]
            elif isinstance(self.G, Gv4) or isinstance(self.G, Gv7) or isinstance(self.G, Gv8):
                nce_layers = [2,6,9,12,14,18]
            elif isinstance(self.G, Gv5):
                nce_layers = [2,6,9,12,15,18]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(feature_net)

        # define loss functions
        self.criterionNCE = []
        patchnce_opt = dnnlib.EasyDict(
            nce_includes_all_negatives_from_minibatch=False,
            batch_size=1,
            nce_T=0.07,
        )

        self.nce_layers = nce_layers
        for _ in nce_layers:
            self.criterionNCE.append(PatchNCELoss(patchnce_opt, pnormSim=sim_pnorm).to(self.device))

        self.setup_F()
        self.F.train().requires_grad_(False).to(self.device)

    def setup_nce_features_attn(self, img):
        if self.feature_attn_layers == 0:
            return

        feat = self.netPre(img, self.nce_layers, encode_only=True)
        if isinstance(feat, tuple):
            feat = feat[1]
        
        max_h, max_w = self.patch_max_shape
        vit_modules = nn.ModuleList()
        for ft in feat:
            _, c, h, w = ft.shape
            out_h, out_w = min(max_h, h), min(max_w, w)
            assert h % out_h == 0 and w % out_w == 0
            patch_size = (h // out_h, w // out_w)
            vit = VisioniTransformer(c, (h, w), patch_size, min(512, c * patch_size[0] * patch_size[1]), self.feature_attn_layers, 4, self.normalize_transformer_out)
            vit_modules.append(vit)
        self.F.vit_modules = vit_modules.to(self.device)

    def get_nce_features(self, img):
        feats = self.netPre(img, self.nce_layers, encode_only=True)
        if isinstance(feats, tuple):
            feats = feats[1]
        
        if self.feature_attn_layers > 0:
            feats = list(map(lambda a: a[0](a[1]), zip(self.F.vit_modules, feats)))

        return feats

    def setup_F(self):
        fimg = torch.empty([1, 3, self.resolution, self.resolution], device=self.device)
        self.setup_nce_features_attn(fimg)
        feat = self.get_nce_features(fimg)

        self.F.create_mlp(feat)
        self.D.latent_dis = LatDiscriminator(self.latent_dim).to(self.device).requires_grad_(False)
        encoder = reduce(lambda a, b: a or b, map(lambda u: isinstance(self.G, u[0]) and u[1], valid_gen_encoder))
        if not self.same_style_encoder:
            if encoder == Ev_gnr:
                self.G.reverse_se = encoder(self.G.size, 8, 3, 1).to(self.device).requires_grad_(False)
            else:
                additional = {}
                if hasattr(self.G, "variational_style_encoder"):
                    additional['variational_style_encoder'] = self.G.variational_style_encoder
                self.G.reverse_se = encoder(
                    latent_dim=self.G.latent_dim, ngf=self.G.ngf, nc=self.G.nc,
                    img_resolution=self.G.img_resolution, lite=self.G.lite, **additional).to(self.device).requires_grad_(False)
        if self.reverse_style_is_norm:
            self.D.reverse_latent_dis = LatDiscriminator(self.latent_dim).to(self.device).requires_grad_(False)

    def calculate_NCE_loss(self, feat_k, feat_q):
        n_layers = len(self.nce_layers)

        feat_k_pool, sample_ids = self.F(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.F(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        weights = [ 1 / n_layers for i in range(0, n_layers) ]

        for f_q, f_k, crit, weight, _ in zip(feat_q_pool, feat_k_pool, self.criterionNCE, weights, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean() * weight
        
        return total_nce_loss

    def run_D(self, img, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img)
        if isinstance(logits, list):
            l = []
            for lg in logits:
                l.append(lg.mean())
            logits = torch.stack(l, dim=0)
        return logits

    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG
        batch_size = real_A.size(0)
        device = real_A.device

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        n_iter = cur_nimg // real_A.size(0)

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                aug_A = self.aug(real_A)
                aug_B = self.aug(real_B)
                A = self.aug(real_A[[np.random.randint(batch_size)]].expand_as(real_A))
                B = self.aug(real_B[[np.random.randint(batch_size)]].expand_as(real_B))
                reverse_se = self.G.encoder if self.same_style_encoder else self.G.reverse_se

                A_content, A_style = self.G.encode(A)
                B_style = reverse_se.style_encode(B)
                aug_A_style = self.G.style_encode(aug_A)
                aug_B_style = reverse_se.style_encode(aug_B)
                rand_A_style = torch.randn([batch_size, self.latent_dim]).to(device)

                idx = torch.randperm(3 * batch_size)
                input_A_style = torch.cat([aug_A_style, rand_A_style, aug_B_style], 0)[idx][:batch_size]
                fake_B = self.G.decode(A_content, input_A_style)
                
                # Adversarial loss
                gen_logits = self.run_D(fake_B, blur_sigma=blur_sigma)
                loss_Gmain_GAN = (-gen_logits).mean()
                loss_Gmain = self.lambda_GAN * loss_Gmain_GAN
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/gan', loss_Gmain_GAN)
                gen_style_logits = torch.cat(self.D.latent_dis(aug_A_style), dim=1)
                loss_Gmain_GAN_style = (-gen_style_logits).mean()
                loss_Gmain = loss_Gmain + loss_Gmain_GAN_style * self.lambda_style_GAN
                training_stats.report('Loss/G/gan_style', loss_Gmain_GAN_style)
                if self.reverse_style_is_norm:
                    aug_B_style = reverse_se.style_encode(aug_B)
                    gen_style_logits_B = torch.cat(self.D.reverse_latent_dis(aug_B_style), dim=1)
                    loss_Gmain_GAN_style_B = (-gen_style_logits_B).mean()
                    loss_Gmain = loss_Gmain + loss_Gmain_GAN_style_B * self.lambda_style_GAN
                    training_stats.report('Loss/G/gan_style_B', loss_Gmain_GAN_style_B)
                    self.aug_B_style = aug_B_style.detach()

                if self.lambda_identity > 0:
                    pass
                    # loss_Gmain_idt = self.criterionIdt(fake_idt_B, real_B)
                    # loss_Gmain = loss_Gmain + self.lambda_identity * loss_Gmain_idt
                    # training_stats.report('Loss/G/identity', loss_Gmain_idt)

                if self.lambda_NCE > 0:
                    loss_Gmain_NCE = self.calculate_NCE_loss(self.get_nce_features(A), self.get_nce_features(fake_B))
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    # if self.nce_idt:
                    #     loss_Gmain_NCE_idt = self.calculate_NCE_loss(self.netPre, real_B, fake_idt_B)
                    #     training_stats.report('Loss/G/NCE_idt', loss_Gmain_NCE_idt)
                    #     loss_Gmain_NCE = (loss_Gmain_NCE + loss_Gmain_NCE_idt) * 0.5
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE * self.lambda_NCE
                
                if self.lambda_style_consis > 0:
                    loss_Gmain_consis_A = A_style.var(0, unbiased=False).sum()
                    training_stats.report('Loss/G/StyleConsistency_A', loss_Gmain_consis_A)

                    loss_Gmain_consis_B = B_style.var(0, unbiased=False).sum()
                    training_stats.report('Loss/G/StyleConsistency_B', loss_Gmain_consis_B)

                    loss_Gmain = loss_Gmain + (loss_Gmain_consis_A + loss_Gmain_consis_B) * self.lambda_style_consis

                if self.lambda_style_recon > 0:
                    recon_style = reverse_se.style_encode(fake_B)
                    loss_Gmain_style_recon = self.criterionStyleRecon(input_A_style, recon_style)
                    if self.style_recon_nce and self.style_recon_force_idt:
                        loss_Gmain_style_recon += self.criterionIdt(input_A_style, recon_style)
                    training_stats.report('Loss/G/StyleReconstruction', loss_Gmain_style_recon)
                    loss_Gmain = loss_Gmain + loss_Gmain_style_recon * self.lambda_style_recon

                self.aug_B = aug_B.detach()
                self.fake_B = fake_B.detach()
                self.rand_A_style = rand_A_style
                self.aug_A_style =aug_A_style.detach()
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_logits = self.run_D(self.fake_B, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                style_gen_logits = torch.cat(self.D.latent_dis(self.aug_A_style), dim=1)
                loss_Dgen_style = (F.relu(torch.ones_like(style_gen_logits) + style_gen_logits)).mean()
                if self.reverse_style_is_norm:
                    style_gen_logits_B = torch.cat(self.D.reverse_latent_dis(self.aug_B_style), dim=1)
                    loss_Dgen_style_B = (F.relu(torch.ones_like(style_gen_logits_B) + style_gen_logits_B)).mean()
                    loss_Dgen_style += loss_Dgen_style_B

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()
                loss_Dgen_style.backward()

            if n_iter % self.d_reg_every == 0 and self.lambda_r1 > 0:
                self.aug_B.requires_grad = True
                self.rand_A_style.requires_grad = True

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_logits = self.run_D(self.aug_B, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                style_real_logits = torch.cat(self.D.latent_dis(self.rand_A_style), dim=1)
                loss_Dreal_style = (F.relu(torch.ones_like(style_real_logits) - style_real_logits)).mean()
                if self.reverse_style_is_norm:
                    rand_B_style = torch.randn([batch_size, self.latent_dim]).to(device).requires_grad_()
                    style_real_logits_B = torch.cat(self.D.reverse_latent_dis(rand_B_style), dim=1)
                    loss_Dreal_style_B = (F.relu(torch.ones_like(style_real_logits_B) - style_real_logits_B)).mean()
                    loss_Dreal_style += loss_Dreal_style_B

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                training_stats.report('Loss/DStyle/loss', loss_Dgen_style + loss_Dreal_style)
            
            if n_iter % self.d_reg_every == 0 and self.lambda_r1 > 0:
                r1_A_loss = d_r1_loss(real_logits, self.aug_B)
                r1_A_s_loss = d_r1_loss(style_real_logits, self.rand_A_style)
                if self.reverse_style_is_norm:
                    r1_B_s_loss = d_r1_loss(style_real_logits_B, rand_B_style)
                    r1_A_s_loss += r1_B_s_loss
                training_stats.report('Loss/D/r1_Aimg', r1_A_loss)
                training_stats.report('Loss/D/r1_Astyle', r1_A_s_loss)
                loss_Dreal += r1_A_loss * self.lambda_r1 * self.d_reg_every
                loss_Dreal_style += r1_A_s_loss * self.lambda_r1 * self.d_reg_every

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()
                loss_Dreal_style.backward()
