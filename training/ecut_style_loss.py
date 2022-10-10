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
from torch import nn
import kornia.augmentation as K
import kornia
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models import losses
from models.patchnce import PatchNCELoss
from models.gnr_networks import Encoder, Generator, LatDiscriminator


class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ECUTStyleLoss(Loss):
    def __init__(self, device, G, D, F, resolution: int,
                 nce_layers: list, feature_net: str, nce_idt: bool, num_patches: int,
                 lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_identity: float = 0,
                 lambda_style_consis: float=50.0, lambda_style_recon: float = 5,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        assert isinstance(G, Generator)
        self.G: Generator = G
        self.D = D
        self.F = F
        self.resolution = resolution
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_identity = lambda_identity
        self.lambda_style_consis = lambda_style_consis
        self.lambda_style_recon = lambda_style_recon
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.criterionIdt = torch.nn.MSELoss()
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
        else:
            raise NotImplemented(feature_net)

        # define loss functions
        self.criterionNCE = []
        patchnce_opt = dnnlib.EasyDict(
            nce_includes_all_negatives_from_minibatch=False,
            batch_size=1,
            nce_T=0.07,
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
        self.D.latent_dis = LatDiscriminator(self.latent_dim).to(self.device).requires_grad_(False)
        self.G.reverse_se = Encoder(self.G.size, self.G.latent_dim, self.G.encoder.num_down, self.G.n_res).to(self.device).requires_grad_(False)

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
                    pass
                    # loss_Gmain_idt = self.criterionIdt(fake_idt_B, real_B)
                    # loss_Gmain = loss_Gmain + self.lambda_identity * loss_Gmain_idt
                    # training_stats.report('Loss/G/identity', loss_Gmain_idt)

                if self.lambda_NCE > 0:
                    loss_Gmain_NCE = self.calculate_NCE_loss(self.netPre, A, fake_B)
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    # if self.nce_idt:
                    #     loss_Gmain_NCE_idt = self.calculate_NCE_loss(self.netPre, real_B, fake_idt_B)
                    #     training_stats.report('Loss/G/NCE_idt', loss_Gmain_NCE_idt)
                    #     loss_Gmain_NCE = (loss_Gmain_NCE + loss_Gmain_NCE_idt) * 0.5
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE * self.lambda_NCE
                
                if self.lambda_style_consis > 0:
                    loss_Gmain_consis = A_style.var(0, unbiased=False).sum()
                    training_stats.report('Loss/G/StyleConsistency', loss_Gmain_consis)
                    loss_Gmain = loss_Gmain + loss_Gmain_consis * self.lambda_style_consis

                if self.lambda_style_recon > 0:
                    recon_style = self.G.reverse_se.style_encode(fake_B)
                    loss_Gmain_style_recon = self.criterionIdt(input_A_style, recon_style)
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
