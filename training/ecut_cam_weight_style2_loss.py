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
import os
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
import kornia.augmentation as K
import kornia
from torch import nn
from models.cam_weight_networks import CamWeightNet
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models import losses
from models.patchnce import PatchNCELoss
from torch.nn.parameter import Parameter
from visual_utils import image_blend_normal, image_grid, visualize_feature, save_image
from models.gnr_networks import LatDiscriminator
from models.fastae_networks import Encoder as Ev1, Generator as Gv1
from models.fastae_v2_networks import Encoder as Ev2, Generator as Gv2
from models.fastae_v3_networks import Encoder as Ev3, Generator as Gv3
from models.style_networks import Encoder as Ev4, Generator as Gv4
from models.style_v2_networks import Encoder as Ev5, Generator as Gv5
from models.style_v3_networks import Encoder as Ev6, Generator as Gv6
from models.fastae_v4_networks import Encoder as Ev7, Generator as Gv7

valid_gen_encoder = [
    (Gv1, Ev1),
    (Gv2, Ev2),
    (Gv3, Ev3),
    (Gv4, Ev4),
    (Gv5, Ev5),
    (Gv6, Ev6),
    (Gv7, Ev7),
]


class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ECUTCAMWeightStyle2Loss(Loss):
    def __init__(self, device, G, D, F, G_ema, resolution: int,
                 feature_net: str, nce_idt: bool, num_patches: int,
                 adaptive_loss: bool, lambda_abdis: float=1.0, sigmoid_attn: bool = False,
                 run_dir: str='.', attn_detach: bool = True, cam_attn_weight: bool=False,
                 style_recon_nce: bool = False,
                 lambda_style_consis: float=50.0, lambda_style_recon: float = 5,
                 lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_identity: float = 0,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.F = F
        self.run_dir = run_dir
        self.resolution = resolution
        self.sigmoid_attn = sigmoid_attn
        self.attn_detach = attn_detach
        self.cam_attn_weight = cam_attn_weight
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_abdis = lambda_abdis
        self.lambda_identity = lambda_identity
        self.lambda_style_consis = lambda_style_consis
        self.lambda_style_recon = lambda_style_recon
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.adaptive_loss = adaptive_loss
        self.criterionIdt = torch.nn.MSELoss()
        self.criterionStyleReCon = losses.ContrastiveNCELoss() if style_recon_nce else torch.nn.MSELoss()
        self.latent_dim = self.G.latent_dim
        self.aug = nn.Sequential(
            K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
            kornia.geometry.transform.Resize(256+30),
            K.RandomCrop((256,256)),
            K.RandomHorizontalFlip(),
        )

        if feature_net == 'efficientnet_lite':
            self.netPre = losses.EfficientNetLite().to(self.device)
            self.nce_layers = [ 2,4,6 ]
            self.attn_layers = [ 4,6,7 ]
        elif feature_net == 'vgg16':
            self.netPre = losses.VGG16().to(self.device)
            self.nce_layers = [ 4,7,9 ]
            self.attn_layers = [ 7,9,10 ]
        elif feature_net == 'learned':
            self.netPre = self.G
            if isinstance(self.G, Gv3):
                self.nce_layers = [2,4,6,8]
            elif isinstance(self.G, Gv4) or isinstance(self.G, Gv6):
                self.nce_layers = [2,6,9,12,14,18]
            elif isinstance(self.G, Gv5):
                self.nce_layers = [2,6,9,12,15,18]
            elif isinstance(self.G, Gv7):
                self.nce_layers = [0,2,4,6]
            self.attn_layers = [ 3,5,7,9 ]
            self.netPre_attn = losses.VGG16().to(self.device)
        else:
            raise NotImplemented(feature_net)

        self.feat_layers = list(set(self.nce_layers + self.attn_layers)) if feature_net != 'learned' else self.nce_layers
        self.feat_layers.sort()
        self.nce_layers_index = []
        self.attn_layers_index = []
        for val in self.nce_layers:
            self.nce_layers_index.append(self.feat_layers.index(val))
        if feature_net != 'learned':
            for val in self.attn_layers:
                self.attn_layers_index.append(self.feat_layers.index(val))

        # define loss functions
        self.BCELogit_loss = torch.nn.BCEWithLogitsLoss()
        self.criterionNCE = []
        patchnce_opt = dnnlib.EasyDict(
            nce_includes_all_negatives_from_minibatch=False,
            batch_size=1,
            nce_T=0.07
        )

        for _ in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(patchnce_opt).to(self.device))

        self.setup_F()
        self.F.train().requires_grad_(False).to(self.device)

    def get_features(self, img):
        feats = self.netPre(img, self.feat_layers, encode_only=True)
        if isinstance(feats, tuple):
            feats = feats[1]

        if hasattr(self, 'netPre_attn'):
            nce = feats
            attn = self.netPre_attn(img, self.attn_layers, encode_only=True)
            if len(attn) < len(nce):
                attn += [ attn[-1] ] * (len(nce) - len(attn))
        else:
            nce = list(map(lambda v: feats[v], self.nce_layers_index))
            attn = list(map(lambda v: feats[v], self.attn_layers_index))
        return attn, nce

    def setup_F(self):
        fimg = torch.empty([1, 3, self.resolution, self.resolution], device=self.device)
        attn_feat, nce_feat = self.get_features(fimg)
        self.F.create_mlp(nce_feat)
        self.F.attn_net = CamWeightNet(ap_weight=0.5, weight_multi=self.cam_attn_weight, sigmoid=self.sigmoid_attn, detach=self.attn_detach)
        self.F.attn_net.setup(attn_feat, nce_feat)
        if self.adaptive_loss:
            loss_weights = Parameter(torch.Tensor(len(nce_feat)))
            loss_weights.data.fill_(1 / len(nce_feat))
            self.F.loss_weights = loss_weights
        self.D.latent_dis = LatDiscriminator(self.latent_dim).to(self.device).requires_grad_(False)
        encoder = reduce(lambda a, b: a or b, map(lambda u: isinstance(self.G, u[0]) and u[1], valid_gen_encoder))
        self.G.reverse_se = encoder(
            latent_dim=self.G.latent_dim, ngf=self.G.ngf, nc=self.G.nc,
            img_resolution=self.G.img_resolution, lite=self.G.lite).to(self.device).requires_grad_(False)

    def calculate_NCE_loss(self, feat_real, feat_fake, attn_weight_hw):
        n_layers = len(feat_real)
        feat_real_pool, sample_ids = self.F(feat_real, self.num_patches, None)
        feat_fake_pool, _ = self.F(feat_fake, self.num_patches, sample_ids)
        attn_weight = []
        if attn_weight_hw is not None:
            for aw, ids in zip(attn_weight_hw, sample_ids):
                aw = 1 - aw
                baw = aw.view(aw.size(0), -1)[:, ids]
                attn_weight.append(baw.view(-1))
        else:
            attn_weight = [ None ] * len(sample_ids)

        total_nce_loss = 0.0
        if self.adaptive_loss:
            loss_weights = self.F.loss_weights
            posw = torch.abs(loss_weights)
            weights = posw / torch.sum(posw) + (1 / (5 * len(self.nce_layers)))
            weights = weights / torch.sum(weights)
        else:
            weights = [ 1 / n_layers for i in range(0, n_layers) ]

        for f_real, f_fake, crit, aw, weight, _ in zip(feat_real_pool, feat_fake_pool, self.criterionNCE, attn_weight, weights, self.nce_layers):
            loss = crit(f_fake, f_real, aw)
            total_nce_loss += loss.mean() * weight
        
        return total_nce_loss

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
        batch_size = real_A.size(0)
        device = real_A.device

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                aug_A = self.aug(real_A)
                aug_B = self.aug(real_B)
                A = self.aug(real_A[[np.random.randint(batch_size)]].expand_as(real_A))

                A_content, A_style = self.G.encode(A)
                aug_A_style = self.G.style_encode(aug_A)
                rand_A_style = torch.randn([batch_size, self.latent_dim]).to(device).requires_grad_()

                idx = torch.randperm(2 * batch_size)
                input_A_style = torch.cat([aug_A_style, rand_A_style], 0)[idx][:batch_size]
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
                loss_Gmain = loss_Gmain + loss_Gmain_GAN_style * self.lambda_GAN * 2
                training_stats.report('Loss/G/gan_style', loss_Gmain_GAN_style)

                if self.lambda_identity > 0:
                    # TODO
                    pass

                attn_real_A, _ = self.get_features(real_A)
                attn_real_B, _ = self.get_features(real_B)
                attn_maps_real_A, logits_real_A = self.F.attn_net(attn_real_A)
                _, logits_real_B = self.F.attn_net(attn_real_B)
                ab_dis_loss = 0
                for logit_real_A, logit_real_B in zip(logits_real_A, logits_real_B):
                    ab_dis_loss += (-logit_real_A).mean()
                    ab_dis_loss += logit_real_B.mean()
                training_stats.report('Loss/G/abdis', ab_dis_loss)

                if self.lambda_NCE > 0:
                    attn_A, feats_A = self.get_features(A)
                    attn_maps_A, _ = self.F.attn_net(attn_A)
                    feats_fake_B = self.netPre(fake_B, self.nce_layers, encode_only=True)
                    if isinstance(feats_fake_B, tuple):
                        feats_fake_B = feats_fake_B[1]
                    loss_Gmain_NCE = self.calculate_NCE_loss(feats_A, feats_fake_B, attn_maps_A)
                    if False:
                        fake_B_ = self.G(real_A)
                        out = image_grid([
                                real_A, 
                                fake_B_,
                                visualize_feature(attn_maps_real_A[0]),
                                visualize_feature(attn_maps_real_A[1]),
                                visualize_feature(attn_maps_real_A[2]),
                                image_blend_normal(visualize_feature(attn_maps_real_A[0]), A),
                                image_blend_normal(visualize_feature(attn_maps_real_A[1]), A),
                                image_blend_normal(visualize_feature(attn_maps_real_A[2]), A),
                            ], 10)
                        save_image(out, os.path.join(self.run_dir, "debug_output.png"))
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    if self.nce_idt:
                        # TODO
                        pass
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE * self.lambda_NCE + ab_dis_loss * self.lambda_abdis

                if self.lambda_style_consis > 0:
                    loss_Gmain_consis = A_style.var(0, unbiased=False).sum()
                    training_stats.report('Loss/G/StyleConsistency', loss_Gmain_consis)
                    loss_Gmain = loss_Gmain + loss_Gmain_consis * self.lambda_style_consis

                if self.lambda_style_recon > 0:
                    recon_style = self.G.reverse_se.style_encode(fake_B)
                    loss_Gmain_style_recon = self.criterionStyleReCon(input_A_style, recon_style)
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

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()
                loss_Dgen_style.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_logits = self.run_D(self.aug_B, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                style_real_logits = torch.cat(self.D.latent_dis(self.rand_A_style), dim=1)
                loss_Dreal_style = (F.relu(torch.ones_like(style_real_logits) - style_real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                training_stats.report('Loss/DStyle/loss', loss_Dgen_style + loss_Dreal_style)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()
                loss_Dreal_style.backward()
