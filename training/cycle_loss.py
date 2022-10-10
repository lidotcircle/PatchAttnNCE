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
import copy
import torch
import torch.nn.functional as F
from torch_utils import training_stats


class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class CycleLoss(Loss):
    def __init__(self, device, G, D, F, G_ema, resolution: int,
                 lambda_GAN: float=1.0, lambda_cycle: float=1.0, lambda_identity: float = 0,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.F = F
        self.resolution = resolution
        self.lambda_GAN = lambda_GAN
        self.lambda_identity = lambda_identity
        self.lambda_cycle = lambda_cycle
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.criterionIdt = torch.nn.MSELoss()
        self.criterionCycle = torch.nn.MSELoss()

        self.setup_F()
        self.F.train().requires_grad_(False).to(self.device)

    def setup_F(self):
        self.G.b2a = copy.deepcopy(self.G).to(self.device).train().requires_grad_(False)
        self.D.a = copy.deepcopy(self.D).to(self.device).train().requires_grad_(False)

    def run_G(self, real_A, real_B):
        fake_B = self.G(real_A)
        fake_A = self.G.b2a(real_B)
        return fake_B, fake_A

    def run_D(self, A, B):
        A_logits = self.D.a(A)
        B_logits = self.D(B)
        return A_logits, B_logits

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
                fake_A2B, fake_B2A = self.run_G(real_A, real_B)
                fake_A_logits, fake_B_logits = self.run_D(fake_B2A, fake_A2B)
                loss_Gmain_GAN = (-(fake_A_logits + fake_B_logits)).mean()
                loss_Gmain = self.lambda_GAN * loss_Gmain_GAN
                training_stats.report('Loss/scores/fake', fake_A_logits)
                training_stats.report('Loss/scores/fake', fake_B_logits)
                training_stats.report('Loss/signs/fake', fake_A_logits.sign())
                training_stats.report('Loss/signs/fake', fake_B_logits.sign())
                training_stats.report('Loss/G/gan', loss_Gmain_GAN)

                if self.lambda_identity > 0:
                    fake_B2B, fake_A2A = self.run_G(real_B, real_A)
                    loss_Gmain_idt = self.criterionIdt(fake_A2A, real_A) + self.criterionIdt(fake_B2B, real_B)
                    loss_Gmain = loss_Gmain + self.lambda_identity * loss_Gmain_idt
                    training_stats.report('Loss/G/identity', loss_Gmain_idt)

                if self.lambda_cycle > 0:
                    fake_B2A2B, fake_A2B2A = self.run_G(fake_B2A, fake_A2B)
                    loss_Gmain_cycle = self.criterionCycle(fake_B2A2B, real_B) + self.criterionCycle(fake_A2B2A, real_A)
                    training_stats.report('Loss/G/cycle', loss_Gmain_cycle)
                    loss_Gmain = loss_Gmain + loss_Gmain_cycle

                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                fake_A2B, fake_B2A = self.run_G(real_A, real_B)
                fake_A_logits, fake_B_logits = self.run_D(fake_B2A, fake_A2B)
                loss_Dgen = (F.relu(torch.ones_like(fake_A_logits) + fake_A_logits)).mean() + \
                            (F.relu(torch.ones_like(fake_B_logits) + fake_B_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', fake_A_logits)
                training_stats.report('Loss/scores/fake', fake_B_logits)
                training_stats.report('Loss/signs/fake', fake_A_logits.sign())
                training_stats.report('Loss/signs/fake', fake_B_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_A_logits, real_B_logits = self.run_D(real_A, real_B)
                loss_Dreal = (F.relu(torch.ones_like(real_A_logits) - real_A_logits)).mean() + \
                             (F.relu(torch.ones_like(real_B_logits) - real_B_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_A_logits)
                training_stats.report('Loss/scores/real', real_B_logits)
                training_stats.report('Loss/signs/real', real_A_logits.sign())
                training_stats.report('Loss/signs/real', real_B_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()