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
import kornia.augmentation as K
import kornia
from torch import nn, autograd
from torch.nn.utils import spectral_norm
from models.gaussian_vae import gaussian_reparameterization, univariate_gaussian_KLD
from models.gnr_networks import LatDiscriminator
from models.transformer import VisioniTransformer
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from models import losses
from models.patchnce import PatchNCELoss
from models.gnr_networks import Encoder as Es, Generator as Gs
from models.fastae_v3_networks import Encoder as Ev3, Generator as Gv3
from models.fastae_v4_networks import Encoder as Ev9, Generator as Gv9
from models.fastae_v8_networks import Encoder as Ev10, Generator as Gv10
from models.style_networks import Encoder as Ev4, Generator as Gv4
from models.style_v2_networks import Encoder as Ev5, Generator as Gv5
from models.style_v3_networks import Encoder as Ev7, Generator as Gv7
from models.style_v4_networks import Encoder as Ev8, Generator as Gv8

valid_gen_encoder = [
    (Gs,  Es),
    (Gv3, Ev3),
    (Gv4, Ev4),
    (Gv5, Ev5),
    (Gv7, Ev7),
    (Gv8, Ev8),
    (Gv9, Ev9),
    (Gv10, Ev10),
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


class ECUTPreStyle2Loss(Loss):
    def __init__(self, device, G, D, F, resolution: int,
                 nce_layers: list, feature_net: str, nce_idt: bool, num_patches: int,
                 style_recon_nce: bool = True, style_recon_nce_mlp_layers: int=0, randn_style: bool=False,
                 lambda_style_KLD: float=0, shuffle_style: bool=False,
                 feature_attn_layers: int=0, patch_max_shape: Tuple[int,int]=(256,256),
                 normalize_transformer_out: bool = True, same_style_encoder: bool = True,
                 lambda_r1: float = 0, d_reg_every: int = 16,
                 lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_identity: float = 0,
                 lambda_style_consis: float=50.0, lambda_style_recon: float = 5,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        assert reduce(lambda a, b: a or b, map(lambda u: isinstance(G, u[0]), valid_gen_encoder))
        assert lambda_style_KLD == 0 or not randn_style
        self.G: Gv3 = G
        self.D = D
        self.F = F
        self.resolution = resolution
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        self.feature_attn_layers = feature_attn_layers
        self.patch_max_shape = patch_max_shape
        self.normalize_transformer_out = normalize_transformer_out
        self.same_style_encoder = same_style_encoder
        self.style_recon_nce = style_recon_nce
        self.style_recon_nce_mlp_layers = style_recon_nce_mlp_layers
        self.randn_style = randn_style
        self.shuffle_style = shuffle_style
        self.lambda_r1 = lambda_r1
        self.d_reg_every = d_reg_every
        self.lambda_style_KLD = lambda_style_KLD
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_identity = lambda_identity
        self.lambda_style_consis = lambda_style_consis
        self.lambda_style_recon = lambda_style_recon
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.criterionIdt = torch.nn.MSELoss()
        self.criterionStyleRecon = losses.ContrastiveNCELoss() if style_recon_nce else torch.nn.MSELoss()
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
            elif isinstance(self.G, Gs):
                raise NotImplementedError("styleGAN generator is not implemented feature extraction")
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
            self.criterionNCE.append(PatchNCELoss(patchnce_opt).to(self.device))

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

        if self.randn_style:
            self.D.latent_dis = LatDiscriminator(self.latent_dim).to(self.device).requires_grad_(False)

        if self.style_recon_nce and self.style_recon_nce_mlp_layers > 0:
            expand_dim = 512
            mlp = [ spectral_norm(nn.Linear(self.latent_dim, expand_dim)) ]
            mlp += [ nn.Sequential(nn.ReLU(), spectral_norm(nn.Linear(expand_dim, expand_dim))) for _ in range(1,self.style_recon_nce_mlp_layers) ]
            self.F.style_nce_mlp = nn.Sequential(*mlp)

        self.F.create_mlp(feat)
        if not self.same_style_encoder:
            encoder = reduce(lambda a, b: a or b, map(lambda u: isinstance(self.G, u[0]) and u[1], valid_gen_encoder))
            if isinstance(self.G, Gs):
                self.G.reverse_se = encoder(self.G.size, self.G.latent_dim, n_res=self.G.n_res, variational_style_encoder=self.lambda_style_KLD>0).to(self.device).requires_grad_(False)
            else:
                self.G.reverse_se = encoder(
                    latent_dim=self.G.latent_dim, ngf=self.G.ngf, nc=self.G.nc,
                    img_resolution=self.G.img_resolution, lite=self.G.lite, variational_style_encoder=self.lambda_style_KLD>0).to(self.device).requires_grad_(False)

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
            reverse_se = self.G.encoder if self.same_style_encoder else self.G.reverse_se

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                A = self.aug(real_A[[np.random.randint(batch_size)]].expand_as(real_A))
                B = self.aug(real_B[[np.random.randint(batch_size)]].expand_as(real_B))
                real_A = self.aug(real_A)
                real_B = self.aug(real_B)

                real_A_content, real_A_style = self.G.encode(real_A)
                real_B_style = reverse_se.style_encode(real_B)
                A_style = self.G.style_encode(A)
                B_style = reverse_se.style_encode(B)
                loss_Gmain = 0

                if self.shuffle_style:
                    idx = torch.randperm(batch_size * 2)
                    real_A_style = torch.cat([real_A_style, real_B_style], dim=0)[idx][:batch_size]

                if self.randn_style:
                    rand_A_style = torch.randn([batch_size, self.latent_dim]).to(device).requires_grad_()
                    idx = torch.randperm(2 * batch_size)
                    input_A_style = torch.cat([real_A_style, rand_A_style], 0)[idx][:batch_size]
                elif self.lambda_style_KLD > 0:
                    real_A_style_mu = real_A_style[:,:real_A_style.size(1)//2]
                    real_A_style_logvar = real_A_style[:,real_A_style.size(1)//2:]
                    input_A_style = gaussian_reparameterization(real_A_style_mu, real_A_style_logvar)
                else:
                    input_A_style = real_A_style

                fake_B = self.G.decode(real_A_content, input_A_style)
                
                # Adversarial loss
                gen_logits = self.run_D(fake_B, blur_sigma=blur_sigma)
                loss_Gmain_GAN = (-gen_logits).mean()
                loss_Gmain = loss_Gmain + self.lambda_GAN * loss_Gmain_GAN
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/gan', loss_Gmain_GAN)

                if self.lambda_style_consis > 0:
                    loss_Gmain_consis_A = A_style.var(0, unbiased=False).sum()
                    training_stats.report('Loss/G/StyleConsistency_A', loss_Gmain_consis_A)

                    loss_Gmain_consis_B = B_style.var(0, unbiased=False).sum()
                    training_stats.report('Loss/G/StyleConsistency_B', loss_Gmain_consis_B)

                    loss_Gmain = loss_Gmain + (loss_Gmain_consis_A + loss_Gmain_consis_B) * self.lambda_style_consis

                if self.lambda_style_KLD > 0:
                    loss_Gmain_style_KLD = univariate_gaussian_KLD(real_A_style_mu, real_A_style_logvar)
                    loss_Gmain = loss_Gmain + loss_Gmain_style_KLD * self.lambda_style_KLD
                    training_stats.report('Loss/G/styleKLD', loss_Gmain_style_KLD)

                # TODO is lambda_GAN good for this ???
                if self.randn_style:
                    gen_style_logits = torch.cat(self.D.latent_dis(real_A_style), dim=1)
                    loss_Gmain_GAN_style = (-gen_style_logits).mean()
                    loss_Gmain = loss_Gmain + loss_Gmain_GAN_style * self.lambda_GAN * 2
                    training_stats.report('Loss/G/gan_style', loss_Gmain_GAN_style)

                if self.lambda_identity > 0:
                    pass
                    # loss_Gmain_idt = self.criterionIdt(fake_idt_B, real_B)
                    # loss_Gmain = loss_Gmain + self.lambda_identity * loss_Gmain_idt
                    # training_stats.report('Loss/G/identity', loss_Gmain_idt)

                if self.lambda_NCE > 0:
                    loss_Gmain_NCE = self.calculate_NCE_loss(self.get_nce_features(real_A), self.get_nce_features(fake_B))
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    # if self.nce_idt:
                    #     loss_Gmain_NCE_idt = self.calculate_NCE_loss(self.netPre, real_B, fake_idt_B)
                    #     training_stats.report('Loss/G/NCE_idt', loss_Gmain_NCE_idt)
                    #     loss_Gmain_NCE = (loss_Gmain_NCE + loss_Gmain_NCE_idt) * 0.5
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE * self.lambda_NCE
                
                if self.lambda_style_recon > 0:
                    recon_style = reverse_se.style_encode(fake_B)
                    if self.lambda_style_KLD > 0:
                        recon_style_mu = recon_style[:,:recon_style.size(1)//2]
                        recon_style_logvar = recon_style[:,recon_style.size(1)//2:]
                        recon_style = gaussian_reparameterization(recon_style_mu, recon_style_logvar)
                        loss_Gmain_recon_style_KLD = univariate_gaussian_KLD(recon_style_mu, recon_style_logvar)
                        loss_Gmain = loss_Gmain + loss_Gmain_recon_style_KLD * self.lambda_style_KLD
                        training_stats.report('Loss/G/reconStyleKLD', loss_Gmain_recon_style_KLD)

                    if self.style_recon_nce and self.style_recon_nce_mlp_layers > 0:
                        mlp = self.F.style_nce_mlp
                        input_style_ = mlp(input_A_style)
                        recon_style_ = mlp(recon_style)
                    else:
                        input_style_ = input_A_style
                        recon_style_ = recon_style

                    loss_Gmain_style_recon = self.criterionStyleRecon(input_style_, recon_style_)
                    training_stats.report('Loss/G/StyleReconstruction', loss_Gmain_style_recon)
                    loss_Gmain = loss_Gmain + loss_Gmain_style_recon * self.lambda_style_recon

                self.real_B = real_B
                self.fake_B = fake_B.detach()
                self.real_A_style = real_A_style.detach()
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_logits = self.run_D(self.fake_B, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                if self.randn_style:
                    style_gen_logits = torch.cat(self.D.latent_dis(self.real_A_style), dim=1)
                    loss_Dgen_style = (F.relu(torch.ones_like(style_gen_logits) + style_gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()
                if self.randn_style:
                    loss_Dgen_style.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_B = self.real_B.detach()
                if n_iter % self.d_reg_every == 0 and self.lambda_r1 > 0:
                    real_B.requires_grad = True
                real_logits = self.run_D(real_B, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                if n_iter % self.d_reg_every == 0 and self.lambda_r1 > 0:
                    r1_A_loss = d_r1_loss(real_logits, real_B)
                    training_stats.report('Loss/D/r1_Aimg', r1_A_loss)
                    loss_Dreal += r1_A_loss * self.lambda_r1 * self.d_reg_every

                if self.randn_style:
                    rand_style = torch.randn([batch_size, self.latent_dim]).to(device).requires_grad_()
                    style_real_logits = torch.cat(self.D.latent_dis(rand_style), dim=1)
                    loss_Dreal_style = (F.relu(torch.ones_like(style_real_logits) - style_real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                if self.randn_style:
                    training_stats.report('Loss/DStyle/loss', loss_Dgen_style + loss_Dreal_style)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()
                if self.randn_style:
                    loss_Dreal_style.backward()