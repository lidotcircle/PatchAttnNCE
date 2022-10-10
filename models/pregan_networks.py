import itertools
import re
from torch.nn.parameter import Parameter
from typing import List
import torch
import torch.nn as nn
from .basic_networks import ILN


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        ActMap = []
        ActMap += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, ngf * mult, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * mult),
            nn.ReLU(True),
            ResnetBlock(ngf * mult, use_bias=False),
            nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, stride=1, bias=True),
            nn.ReLU(True),
            FMNorm(num_channels=ngf * mult),
        ]

        # Up-Sampling
        UpBlockResnet = []
        for i in range(n_blocks):
            UpBlockResnet += [ResnetBlock(ngf * mult, use_bias=False)]

        UpBlock = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * mult),
            nn.ReLU(True)
        ]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.ActMap = nn.Sequential(*ActMap)
        self.UpBlockResnet = nn.Sequential(*UpBlockResnet)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input: torch.Tensor, feature: torch.Tensor, features:List=None, max_layer: int=-1): 
        x: torch.Tensor = input
        def forward_x(x, layers):
            if features is None:
                x = layers(x)
                return x
            else:
                for _, layer in enumerate(layers):
                    x = layer(x)
                    features.append(x)
                    if max_layer >= 0 and len(features) > max_layer:
                        return None
                return x
        
        x = forward_x(x, self.DownBlock)
        if x is None:
            return
        pre_x = x

        actMap = self.ActMap(feature)
        x = x * actMap
        pre_x = pre_x * (1 - actMap)
        heatmap = torch.mean(actMap, dim = 1)
        if features is not None:
            features.append(x)
            if max_layer >= 0 and len(features) > max_layer:
                return

        x = forward_x(x, self.UpBlockResnet)
        if x is None:
            return

        x = torch.cat((pre_x, x), dim=1)
        x = forward_x(x, self.UpBlock)
        if x is None:
            return

        return x, heatmap


class ResnetGeneratorV2(nn.Module):
    """
    @param merge_mode [ 'middle', 'middle_add', 'last', 'last_add' ]
    """
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, merge_mode: str='middle', attn_mode:str='upsample', interp_mode:str='nearest', only_focus: bool=False, decoder_dropout: float=0.0, vae_mode: bool=False):
        assert(n_blocks >= 0)
        super(ResnetGeneratorV2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.merge_mode = merge_mode
        self.only_focus = only_focus
        self.vae_mode = vae_mode

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        ActMap = []
        if attn_mode == 'upsample':
            ActMap += [
                nn.Upsample(scale_factor=2, mode=interp_mode),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
                ILN(128),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode=interp_mode),
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, ngf * mult, kernel_size=3, stride=1, padding=0, bias=False),
                ILN(ngf * mult),
                nn.ReLU(True),
                ResnetBlock(ngf * mult, use_bias=False),
                nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, stride=1, bias=True),
                nn.ReLU(True),
                FMNorm(num_channels=ngf * mult),
            ]
        elif attn_mode =='interp1':
            ActMap += [
                nn.Upsample(scale_factor=4, mode=interp_mode),
                FMNorm(num_channels=ngf * mult),
            ]
        elif attn_mode =='interp2':
            ActMap += [
                nn.Conv2d(256, ngf * mult, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, mode=interp_mode),
                FMNorm(num_channels=ngf * mult),
            ]
        elif attn_mode == 'interp3':
            ActMap += [
                nn.Conv2d(512, ngf * mult, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=8, mode=interp_mode),
                FMNorm(num_channels=ngf * mult),
            ]
        else:
            assert False

        # Transmodule
        Transmodule1 = []
        for i in range(n_blocks // 2 if self.merge_mode in ['middle', 'middle_add'] else n_blocks):
            Transmodule1 += [ResnetBlock(ngf * mult, use_bias=False)]
        if self.merge_mode not in [ 'last_add' ]:
            Transmodule2= []
            if self.merge_mode in [ 'middle', 'last' ]:
                Transmodule2.append(nn.Conv2d(ngf * mult * 2, ngf * mult, 1, stride=1))
                Transmodule2.append(nn.ReLU())
            if self.merge_mode in [ 'middle', 'middle_add' ]:
                for i in range(n_blocks // 2):
                    Transmodule2 += [ResnetBlock(ngf * mult, use_bias=False)]
            self.Transmodule2 = nn.Sequential(*Transmodule2)

        if self.vae_mode:
            self.mu_logvar_net = nn.Conv2d(ngf * mult, 2 * ngf * mult, kernel_size=1, stride=1, bias=True)

        # Up-Sampling
        UpBlock = []

        if decoder_dropout > 0:
            UpBlock.append(nn.Dropout(p=decoder_dropout))

        for i in range(n_blocks):
            UpBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.ActMap = nn.Sequential(*ActMap)
        self.Transmodule1 = nn.Sequential(*Transmodule1)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input: torch.Tensor, feature: torch.Tensor, features:List=None, max_layer: int=-1, mu_logvar_out: list=None):
        x: torch.Tensor = input
        def forward_x(x, layers):
            if features is None:
                x = layers(x)
                return x
            else:
                for _, layer in enumerate(layers):
                    x = layer(x)
                    features.append(x)
                    if max_layer >= 0 and len(features) > max_layer:
                        return None
                return x
        
        x = forward_x(x, self.DownBlock)
        if x is None:
            return
        latent = x

        actMap = self.ActMap(feature)
        in_x = latent * actMap
        out_x: torch.Tensor = latent * (1 - actMap)
        if self.only_focus:
            out_x.fill_(0)
        heatmap = torch.mean(actMap, dim = 1)

        in_x = forward_x(in_x, self.Transmodule1)
        if in_x is None:
            return
        x = torch.cat([in_x, out_x], dim=1) if self.merge_mode in [ 'middle', 'last' ] else in_x + out_x
        if self.merge_mode not in [ 'last_add' ]:
            x = forward_x(x, self.Transmodule2)
            if x is None:
                return

        if self.vae_mode:
            x = self.vae_reparam(x, mu_logvar_out)
            latent = self.vae_reparam(latent, mu_logvar_out)

        x = forward_x(x, self.UpBlock)
        if x is None:
            return

        recon = self.UpBlock(latent)
        return x, recon, heatmap

    def vae_reparam(self, x, mu_logvar_out: list=None):
        if not self.vae_mode:
            return x
        x = self.mu_logvar_net(x)
        mu = x[:, :x.size(1) // 2]
        logvar = x[:, x.size(1) // 2:]
        if mu_logvar_out is not None:
            mu_logvar_out += [mu, logvar]
        x = self.gaussian_reparameterization(mu, logvar)
        return x

    def gaussian_reparameterization(self, mu, logvar):
        mu_shape = mu.shape
        mu = mu.view(mu.size(0), -1)
        logvar = logvar.view(logvar.size(0), -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        ans = mu + eps * std
        return ans.view(mu_shape)

 
class ResnetGeneratorV3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256):
        assert(n_blocks >= 0)
        super(ResnetGeneratorV3, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        ActMap = []
        for i in range(n_blocks):
            ActMap += [ResnetBlock(ngf * mult, use_bias=False)]
        ActMap += [ FMNorm(num_channels=ngf * mult) ]

        # Transmodule
        Transmodule1 = []
        for i in range(n_blocks // 2):
            Transmodule1 += [ResnetBlock(ngf * mult, use_bias=False)]
        Transmodule2= [
            nn.Conv2d(ngf * mult * 2, ngf * mult, 1, stride=1),
            nn.ReLU()
            ]
        for i in range(n_blocks // 2):
            Transmodule2 += [ResnetBlock(ngf * mult, use_bias=False)]

        # Up-Sampling
        UpBlock = []
        for i in range(n_blocks):
            UpBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.ActMap = nn.Sequential(*ActMap)
        self.Transmodule1 = nn.Sequential(*Transmodule1)
        self.Transmodule2 = nn.Sequential(*Transmodule2)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input: torch.Tensor, features:List=None, max_layer: int=-1): 
        x: torch.Tensor = input
        def forward_x(x, layers):
            if features is None:
                x = layers(x)
                return x
            else:
                for _, layer in enumerate(layers):
                    x = layer(x)
                    features.append(x)
                    if max_layer >= 0 and len(features) > max_layer:
                        return None
                return x
        
        x = forward_x(x, self.DownBlock)
        if x is None:
            return
        latent = x

        actMap = self.ActMap(latent)
        in_x = latent * actMap
        out_x = latent * (1 - actMap)
        heatmap = torch.mean(actMap, dim = 1)

        in_x = forward_x(in_x, self.Transmodule1)
        if in_x is None:
            return
        x = torch.cat([in_x, out_x], dim=1)
        x = forward_x(x, self.Transmodule2)
        if x is None:
            return

        x = forward_x(x, self.UpBlock)
        if x is None:
            return

        recon = self.UpBlock(latent)
        return x, recon, heatmap


class ResnetSepGenerator(nn.Module):
    """
    @param merge_mode [ 'middle', 'middle_add', 'last', 'last_add' ]
    """
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, merge_mode: str='middle', attn_mode:str='upsample', interp_mode:str='nearest', only_focus: bool=False, decoder_dropout: float=0.0):
        assert(n_blocks >= 0)
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.merge_mode = merge_mode
        self.only_focus = only_focus

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        ActMap = []
        if attn_mode == 'upsample':
            ActMap += [
                nn.Upsample(scale_factor=2, mode=interp_mode),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
                ILN(128),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode=interp_mode),
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, ngf * mult, kernel_size=3, stride=1, padding=0, bias=False),
                ILN(ngf * mult),
                nn.ReLU(True),
                ResnetBlock(ngf * mult, use_bias=False),
                nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, stride=1, bias=True),
                nn.ReLU(True),
                FMNorm(num_channels=ngf * mult),
            ]
        elif attn_mode =='interp1':
            ActMap += [
                nn.Upsample(scale_factor=4, mode=interp_mode),
                FMNorm(num_channels=ngf * mult),
            ]
        elif attn_mode =='interp2':
            ActMap += [
                nn.Conv2d(256, ngf * mult, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, mode=interp_mode),
                FMNorm(num_channels=ngf * mult),
            ]
        elif attn_mode == 'interp3':
            ActMap += [
                nn.Conv2d(512, ngf * mult, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=8, mode=interp_mode),
                FMNorm(num_channels=ngf * mult),
            ]
        else:
            assert False

        # Transmodule
        Transmodule1 = []
        for i in range(n_blocks // 2 if self.merge_mode in ['middle', 'middle_add'] else n_blocks):
            Transmodule1 += [ResnetBlock(ngf * mult, use_bias=False)]
        if self.merge_mode not in [ 'last_add' ]:
            Transmodule2= []
            if self.merge_mode in [ 'middle', 'last' ]:
                Transmodule2.append(nn.Conv2d(ngf * mult * 2, ngf * mult, 1, stride=1))
                Transmodule2.append(nn.ReLU())
            if self.merge_mode in [ 'middle', 'middle_add' ]:
                for i in range(n_blocks // 2):
                    Transmodule2 += [ResnetBlock(ngf * mult, use_bias=False)]
            self.Transmodule2 = nn.Sequential(*Transmodule2)

        # Up-Sampling
        UpBlock = []

        if decoder_dropout > 0:
            UpBlock.append(nn.Dropout(p=decoder_dropout))

        for i in range(n_blocks):
            UpBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.ActMap = nn.Sequential(*ActMap)
        self.Transmodule1 = nn.Sequential(*Transmodule1)
        self.UpBlock = nn.Sequential(*UpBlock)

    def AEParameters(self):
        return itertools.chain(self.DownBlock.parameters(), self.UpBlock.parameters())
    
    def TransParameters(self):
        params = [self.ActMap.parameters(), self.Transmodule1.parameters()]
        if hasattr(self, 'Transmodule2'):
            params.append(self.Transmodule2.parameters())
        return itertools.chain(*params)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def setAEMode(self):
        self.set_requires_grad([self.DownBlock, self.UpBlock], True)
        params = [self.ActMap, self.Transmodule1]
        if hasattr(self, 'Transmodule2'):
            params.append(self.Transmodule2)
        self.set_requires_grad(params, False)
    
    def setTransMode(self):
        self.set_requires_grad([self.DownBlock, self.UpBlock], False)
        params = [self.ActMap, self.Transmodule1]
        if hasattr(self, 'Transmodule2'):
            params.append(self.Transmodule2)
        self.set_requires_grad(params, True)

    def forward(self, input: torch.Tensor, feature: torch.Tensor, features:List=None, max_layer: int=-1):
        x: torch.Tensor = input
        def forward_x(x, layers):
            if features is None:
                x = layers(x)
                return x
            else:
                for _, layer in enumerate(layers):
                    x = layer(x)
                    features.append(x)
                    if max_layer >= 0 and len(features) > max_layer:
                        return None
                return x
        
        x = forward_x(x, self.DownBlock)
        if x is None:
            return
        latent = x

        if feature is not None:
            actMap = self.ActMap(feature)
            in_x = latent * actMap
            out_x: torch.Tensor = latent * (1 - actMap)
            if self.only_focus:
                out_x.fill_(0)
            heatmap = torch.mean(actMap, dim = 1)

            in_x = forward_x(in_x, self.Transmodule1)
            if in_x is None:
                return
            x = torch.cat([in_x, out_x], dim=1) if self.merge_mode in [ 'middle', 'last' ] else in_x + out_x
            if self.merge_mode not in [ 'last_add' ]:
                x = forward_x(x, self.Transmodule2)
                if x is None:
                    return

            x = forward_x(x, self.UpBlock)
            if x is None:
                return

            return x, heatmap
        else:
            x = forward_x(x, self.UpBlock)
            if x is None:
                return

            return x


class FMNorm(nn.Module):
    def __init__(self, eps=1e-6, history_stat=True, num_channels=256):
        super(FMNorm, self).__init__()
        self.eps = eps
        self.history_stat = history_stat
        if self.history_stat:
            self.his_min = Parameter(torch.Tensor(1, num_channels, 1, 1), requires_grad=False)
            self.his_max = Parameter(torch.Tensor(1, num_channels, 1, 1), requires_grad=False)
            self.hisn = Parameter(torch.Tensor(1), requires_grad=False)
            self.his_min.data.fill_(0.0)
            self.his_max.data.fill_(1.0)
            self.hisn.data.fill_(0.0)

    
    def forward(self, feat: torch.Tensor):
        min_val = torch.min(feat.view(feat.size(0), feat.size(1), -1), dim=2).values
        max_val = torch.max(feat.view(feat.size(0), feat.size(1), -1), dim=2).values
        min_val = min_val.unsqueeze(2).unsqueeze(3)
        max_val = max_val.unsqueeze(2).unsqueeze(3)
        if self.history_stat:
            batch_size = feat.size(0)
            m1 = min_val.mean(dim=0).unsqueeze(0)
            m2 = max_val.mean(dim=0).unsqueeze(0)
            new_min = self.hisn * self.his_min + m1 * batch_size
            new_max = self.hisn * self.his_max + m2 * batch_size
            self.hisn = Parameter(self.hisn.data + batch_size, requires_grad=False)
            min_val = new_min / self.hisn
            max_val = new_max / self.hisn
            self.his_min = Parameter(min_val, requires_grad=False)
            self.his_max = Parameter(max_val, requires_grad=False)
        feat = (feat - min_val) / (max_val + self.eps)
        return feat


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class YAPatch(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(YAPatch, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        model += [nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class MultiYAPatch(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers: List[int]=[5]):
        super(MultiYAPatch, self).__init__()
        self.n_nets = len(n_layers)
        for i in range(self.n_nets):
            net = YAPatch(input_nc, ndf, n_layers[i])
            setattr(self, f"net_{i}", net)

    def forward(self, input):
        outputs = []

        for i in range(self.n_nets):
            net = getattr(self, f"net_{i}")
            out = net(input)
            out = out.view(out.size(0), -1)
            outputs.append(out)

        return torch.cat(outputs, dim=1)