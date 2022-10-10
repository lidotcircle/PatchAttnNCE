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
import os
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from models.cam_weight_networks import CamWeightNet
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models import losses
from models.patchnce import PatchNCELoss
from torch.nn.parameter import Parameter
from visual_utils import image_blend_normal, image_grid, visualize_feature, save_image


class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ECUTCAMWeightLoss(Loss):
    def __init__(self, device, G, D, F, G_ema, resolution: int,
                 feature_net: str, nce_idt: bool, num_patches: int,
                 adaptive_loss: bool, lambda_abdis: float=1.0, sigmoid_attn: bool = False,
                 run_dir: str='.', attn_detach: bool = True, cam_attn_weight: bool=False,
                 lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_classify: float=1, lambda_identity: float = 0,
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
        self.lambda_classify = lambda_classify
        self.lambda_abdis = lambda_abdis
        self.lambda_identity = lambda_identity
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.adaptive_loss = adaptive_loss
        self.criterionIdt = torch.nn.MSELoss()

        if feature_net == 'efficientnet_lite':
            self.netPre = losses.EfficientNetLite().to(self.device)
            self.nce_layers = [ 2,4,6 ]
            self.attn_layers = [ 2,4,6, ]
        elif feature_net == 'vgg16':
            self.netPre = losses.VGG16().to(self.device)
            self.nce_layers = [ 4, 7, 9 ]
            self.attn_layers = [ 7, 9, 10 ]
        elif feature_net == 'learned':
            self.netPre = self.G
            self.nce_layers = [ 4,8,12,16 ]
            self.attn_layers = [ 4,8,12,16 ]
        else:
            raise NotImplemented(feature_net)

        self.feat_layers = list(set(self.nce_layers + self.attn_layers))
        self.feat_layers.sort()
        self.nce_layers_index = []
        self.attn_layers_index = []
        for val in self.nce_layers:
            self.nce_layers_index.append(self.feat_layers.index(val))
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

    def get_nce_attn(self, feats):
        attn = list(map(lambda v: feats[v], self.attn_layers_index))
        nce = list(map(lambda v: feats[v], self.nce_layers_index))
        return attn, nce

    def setup_F(self):
        fimg = torch.empty([1, 3, self.resolution, self.resolution], device=self.device)
        feat = self.netPre(fimg, self.feat_layers, encode_only=True)
        if isinstance(feat, tuple):
            feat = feat[1]
        
        attn_feat, nce_feat = self.get_nce_attn(feat)
        self.F.create_mlp(nce_feat)
        self.F.attn_net = CamWeightNet(ap_weight=0.5, weight_multi=self.cam_attn_weight, sigmoid=self.sigmoid_attn, detach=self.attn_detach)
        self.F.attn_net.setup(attn_feat, nce_feat)
        if self.adaptive_loss:
            loss_weights = Parameter(torch.Tensor(len(nce_feat)))
            loss_weights.data.fill_(1 / len(nce_feat))
            self.F.loss_weights = loss_weights

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

                feats_real_A = self.netPre(real_A, self.feat_layers, encode_only=True)
                feats_real_B = self.netPre(real_B, self.feat_layers, encode_only=True)
                feats_fake_B = self.netPre(real_B, self.feat_layers, encode_only=True)
                attn_real_A, feats_real_A = self.get_nce_attn(feats_real_A)
                attn_real_B, feats_real_B = self.get_nce_attn(feats_real_B)
                attn_fake_B, feats_fake_B = self.get_nce_attn(feats_fake_B)
                attn_A, logits_real_A = self.F.attn_net(attn_real_A)
                attn_B, logits_real_B = self.F.attn_net(attn_real_B, logit_only=True)
                ab_dis_loss = 0
                for logit_A, logit_B in zip(logits_real_A, logits_real_B):
                    ab_dis_loss += (-logit_A).mean()
                    ab_dis_loss += logit_B.mean()
                training_stats.report('Loss/G/abdis', ab_dis_loss)

                if self.lambda_classify > 0:
                    _, logits_fake_B = self.F.attn_net(attn_fake_B, logit_only=True)
                    loss_Gmain_classify = 0
                    for logit in logits_fake_B:
                        loss_Gmain_classify += logit.mean()
                    loss_Gmain = loss_Gmain + F.sigmoid(2 * loss_Gmain_classify / torch.abs(ab_dis_loss.detach())) * self.lambda_classify
                    training_stats.report('Loss/G/classify', loss_Gmain_classify)

                if self.lambda_NCE > 0:
                    loss_Gmain_NCE = self.calculate_NCE_loss(feats_real_A, feats_fake_B, attn_A)
                    if False:
                        out = image_grid([
                                real_A, 
                                fake_B,
                                image_blend_normal(visualize_feature(attn_A[0]), real_A),
                                image_blend_normal(visualize_feature(attn_A[1]), real_A),
                                image_blend_normal(visualize_feature(attn_A[2]), real_A),
                            ], 10)
                        save_image(out, os.path.join(self.run_dir, "debug_output.png"))
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    if self.nce_idt:
                        feats_fake_idt_B = self.netPre(fake_idt_B, self.nce_layers, encode_only=True)
                        loss_Gmain_NCE_idt = self.calculate_NCE_loss(feats_real_B, feats_fake_idt_B, None)
                        training_stats.report('Loss/G/NCE_idt', loss_Gmain_NCE_idt)
                        loss_Gmain_NCE = (loss_Gmain_NCE + loss_Gmain_NCE_idt) * 0.5
                        if False:
                            out = image_grid([
                                    real_B, 
                                    fake_idt_B,
                                    image_blend_normal(visualize_feature(attn_B[0]), real_B),
                                    image_blend_normal(visualize_feature(attn_B[1]), real_B),
                                    image_blend_normal(visualize_feature(attn_B[2]), real_B),
                                ], 10)
                            save_image(out, os.path.join(self.run_dir, "debug_output_idt.png"))
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE * self.lambda_NCE + ab_dis_loss * self.lambda_abdis

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
