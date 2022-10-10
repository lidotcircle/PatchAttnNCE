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
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from models.cut_networks import AttnPatchSampleF
from models.transformer import VisioniTransformer
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models import losses
from models.patchnce import PatchNCELoss
from torch.nn.parameter import Parameter


class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ECUTAttnLoss(Loss):
    def __init__(self, device, G, D, F, G_ema, resolution: int,
                 nce_layers: list, feature_net: str, nce_idt: bool, num_patches: int,
                 total_kimg: int, sim_pnorm: float = 0, weights_strategy: str = 'linear_decay',
                 feature_attn_layers: int=0, patch_max_shape: Tuple[int,int]=(256,256),
                 lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_identity: float = 0,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.F = F
        assert feature_attn_layers > 0
        assert isinstance(F, AttnPatchSampleF)
        self.resolution = resolution
        self.total_kimg = total_kimg
        self.weights_strategy = weights_strategy
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        self.feature_attn_layers = feature_attn_layers
        self.patch_max_shape = patch_max_shape
        self.normalize_transformer_out = True
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_identity = lambda_identity
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.criterionIdt = torch.nn.MSELoss()

        if feature_net == 'efficientnet_lite':
            self.netPre = losses.EfficientNetLite().to(self.device)
        elif feature_net == 'vgg16':
            self.netPre = losses.VGG16().to(self.device)
        elif feature_net == 'learned':
            self.netPre = self.G
        else:
            raise NotImplemented(feature_net)

        # define loss functions
        self.nce_layers = nce_layers
        patchnce_opt = dnnlib.EasyDict(
            nce_includes_all_negatives_from_minibatch=False,
            batch_size=1,
            nce_T=0.07,
        )
        self.criterionNCE = PatchNCELoss(patchnce_opt, pnormSim=sim_pnorm).to(self.device)

        self.setup_F()
        self.F.train().requires_grad_(False).to(self.device)

    def setup_nce_features_attn(self, img):
        feat = self.netPre(img, self.nce_layers, encode_only=True)
        if isinstance(feat, tuple):
            feat = feat[1]
        
        max_h, max_w =  16, 16
        vit_modules = torch.nn.ModuleList()
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
        
        attn_feats = list(map(lambda a: a[0](a[1]), zip(self.F.vit_modules, feats)))
        return feats, attn_feats

    def setup_F(self):
        fimg = torch.empty([1, 3, self.resolution, self.resolution], device=self.device)
        self.setup_nce_features_attn(fimg)
        feat, attn_feat = self.get_nce_features(fimg)
        self.F.create_mlp((feat, attn_feat))

    def calculate_NCE_loss(self, feat_k, feat_q, cur_nimg):
        feat_k_pool, patches_weights, sample_ids = self.F(feat_k, self.num_patches, None, weights_strategy=self.weights_strategy)
        feat_q_pool, _, _ = self.F(feat_q, self.num_patches, sample_ids)

        pwm, pwv = 0, 0
        if self.weights_strategy == 'linear_decay':
            attn_weight = cur_nimg / (1000 * self.total_kimg)
            raw_weight = 1 - attn_weight
            k1 = [ torch.zeros([x.size(0)], device=x.device, dtype=x.dtype).fill_(raw_weight) for x in feat_k_pool[0] ]
            k2 = [ torch.zeros([x.size(0)], device=x.device, dtype=x.dtype).fill_(raw_weight) for x in feat_k_pool[1] ]
            patches_weights = k1 + k2
            pass
        else:
            patches_weights: List[torch.Tensor] = patches_weights[0] + patches_weights[1]
            pwm, pwv = reduce(lambda b, pw: pw.mean() + b, patches_weights), reduce(lambda b, pw: pw.var() + b, patches_weights)
            pwm, pwv = pwm / len(patches_weights), pwv / len(patches_weights)

        feat_k_pool = feat_k_pool[0] + feat_k_pool[1]
        feat_q_pool = feat_q_pool[0] + feat_q_pool[1]

        total_nce_loss = 0.0
        n_layers = len(feat_k_pool)
        weights = [ 2 / n_layers for i in range(0, n_layers) ]

        for f_q, f_k, patches_weight, weight in zip(feat_q_pool, feat_k_pool, patches_weights, weights):
            loss = self.criterionNCE(f_q, f_k, patches_weight)
            total_nce_loss += loss.mean() * weight
        
        return total_nce_loss, (pwm, pwv)

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
                real = torch.cat([real_A, real_B], dim=0) if self.nce_idt or self.lambda_identity > 0 else real_A
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

                if self.lambda_NCE > 0:
                    loss_Gmain_NCE, weight_stats = self.calculate_NCE_loss(self.get_nce_features(real_A), self.get_nce_features(fake_B), cur_nimg)
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    training_stats.report('Loss/F/weight_mean', weight_stats[0])
                    training_stats.report('Loss/F/weight_var', weight_stats[1])
                    if self.nce_idt:
                        loss_Gmain_NCE_idt, idt_weight_stats = self.calculate_NCE_loss(self.get_nce_features(real_B), self.get_nce_features(fake_idt_B), cur_nimg)
                        training_stats.report('Loss/G/NCE_idt', loss_Gmain_NCE_idt)
                        training_stats.report('Loss/F/idt_weight_mean', idt_weight_stats[0])
                        training_stats.report('Loss/F/idt_weight_var', idt_weight_stats[1])
                        loss_Gmain_NCE = (loss_Gmain_NCE + loss_Gmain_NCE_idt) * 0.5
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE

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
