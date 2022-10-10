import math
from cv2 import normalize
import torch.nn.functional as F 
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils import spectral_norm
from models.gaussian_vae import gaussian_reparameterization
from op import fused_leaky_relu


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def embedding(*args, **kwargs):
    return spectral_norm(nn.Embedding(*args, **kwargs))

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        bias = self.bias*self.lr_mul if self.bias is not None else None
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, input):
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        return (input - ln_mean) / torch.sqrt(ln_var + self.eps)

def NormLayer(c, mode='instance'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)
    elif mode == 'instance':
        return nn.InstanceNorm2d(c)
    elif mode == 'layer':
        return LayerNorm(c)
    elif mode == 'iln':
        return ILN(c)
    else:
        raise NotImplementedError()

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class EEBlock(nn.Module):
    def __init__(self, style_dim, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            EqualLinear(style_dim, style_dim, bias=True, activation='fused_lrelu'),
            EqualLinear(style_dim, ch_out, bias=True),
        )
    
    def forward(self, style, feat):
        return feat * self.main(style).unsqueeze(2).unsqueeze(3)

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='instance'):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1),
            NormLayer(out_planes, mode=norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, feat):
        return self.main(feat)

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

def UpBlockBig(in_planes, out_planes, norm: str='instance'):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2, norm), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2, norm), GLU()
        )
    return block

def UpBlockSmall(in_planes, out_planes, norm: str='instance'):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NormLayer(out_planes*2, norm), GLU())
    return block

class UpBlockStyleBig(nn.Module):
    def __init__(self, in_planes, out_planes, style_dim):
        super().__init__()
        self.excitation = EEBlock(style_dim, out_planes)
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
            NoiseInjection(),
            NormLayer(out_planes*2, 'instance'), GLU(),
            conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
            NoiseInjection(),
            NormLayer(out_planes*2, 'instance'), GLU()
            )
    
    def forward(self, input, style):
       return self.excitation(style, self.block(input))

class UpBlockStyleSmall(nn.Module):
    def __init__(self, in_planes, out_planes, style_dim):
        super().__init__()
        self.excitation = EEBlock(style_dim, out_planes)
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
            NormLayer(out_planes*2, 'instance'), GLU())
      
    def forward(self, input, style):
       return self.excitation(style, self.block(input))

class ForwardWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, input):
        x = input if not isinstance(input, list) else input[-1]
        input = [] if not isinstance(input, list) else input
        input.append(self.module(x))
        return input

class ContentEncoder(nn.Module):
    def __init__(self, ngf=128, img_resolution=256, nc=3):
        super().__init__()
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        assert img_resolution in nfc
        act_layers = []
        act_layers += [ nn.Conv2d(nc, nfc[img_resolution], kernel_size=7, stride=1, padding=3) ]
        if img_resolution > 512:
            act_layers += [ DownBlock(nfc[1024], nfc[512], 'layer') ]
        elif img_resolution == 512:
            act_layers += [ DownBlock(nfc[512], nfc[256], 'layer') ]

        act_layers += [ DownBlock(nfc[256], nfc[128], 'layer') ]
        self.act_layers = nn.Sequential(*act_layers)

        layers = []
        layers += [ ForwardWrapper(DownBlock(nfc[128], nfc[64], 'instance')) ]
        layers += [ ForwardWrapper(DownBlock(nfc[64], nfc[32], 'instance')) ]
        layers += [ ForwardWrapper(DownBlock(nfc[32], nfc[16], 'instance')) ]
        layers += [ ForwardWrapper(DownBlock(nfc[16], nfc[8], 'instance')) ]
        self.layers = nn.Sequential(*layers)

    def forward(self, img, **kwargs):
        return self.layers(self.act_layers(img))

class ExcitationFeature(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, feat):
        return self.main(feat).flatten(1)

class Normalize(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, input):
        return F.normalize(input, *self.args, **self.kwargs)

class StyleEncoder(nn.Module):
    def __init__(self, latent_dim, ngf=128, img_resolution=256, nc=3, normalize_style: bool=False, variational_style_encoder: bool=False):
        super().__init__()
        latent_multiplier = 2 if variational_style_encoder else 1
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        assert img_resolution in nfc
        act_layers = []
        act_layers += [ nn.Conv2d(nc, nfc[img_resolution], kernel_size=7, stride=1, padding=3) ]
        if img_resolution > 512:
            act_layers += [ DownBlock(nfc[1024], nfc[512], 'layer') ]
        elif img_resolution == 512:
            act_layers += [ DownBlock(nfc[512], nfc[256], 'layer') ]

        act_layers += [ DownBlock(nfc[256], nfc[128], 'layer') ]
        self.act_layers = nn.Sequential(*act_layers)

        for key in nfc.keys():
            nfc[key] = nfc[key] // 4
        excitation_dim = nfc[64]  + nfc[32] + nfc[16] + nfc[8] + nfc[4]

        self.down_blocks = nn.ModuleList()
        self.excitation_blocks = nn.ModuleList()
        self.down_blocks.append(DownBlock(nfc[128] * 4, nfc[64], 'layer'))
        self.down_blocks.append(DownBlock(nfc[64],  nfc[32], 'layer'))
        self.down_blocks.append(DownBlock(nfc[32],  nfc[16], 'layer'))
        self.down_blocks.append(DownBlock(nfc[16],  nfc[8],  'layer'))
        self.down_blocks.append(DownBlock(nfc[8],   nfc[4],  'layer'))
        self.excitation_blocks.append(ExcitationFeature(nfc[64], nfc[64]))
        self.excitation_blocks.append(ExcitationFeature(nfc[32], nfc[32]))
        self.excitation_blocks.append(ExcitationFeature(nfc[16], nfc[16]))
        self.excitation_blocks.append(ExcitationFeature(nfc[8],  nfc[8]))
        self.excitation_blocks.append(ExcitationFeature(nfc[4],  nfc[4]))

        out = [
            EqualLinear(excitation_dim, excitation_dim, activation='fused_lrelu'), 
            EqualLinear(excitation_dim, excitation_dim, activation='fused_lrelu'), 
            EqualLinear(excitation_dim, latent_dim * latent_multiplier) 
        ]
        if normalize_style:
            out.append(Normalize(p=2, dim=1))
        self.out = nn.Sequential(*out)

    def forward(self, img, **kwargs):
        act = self.act_layers(img)
        excitations = []
        for exci, down in zip(self.excitation_blocks, self.down_blocks):
            act = down(act)
            excitations.append(exci(act))
        return self.out(torch.cat(excitations, dim=1))

class Encoder(nn.Module):
    def __init__(self, latent_dim: int, ngf=128, nc: int=3, img_resolution: int=256, normalize_style: bool=False, variational_style_encoder: bool=False, **kwargs):
        super().__init__()
        self.style_encoder = StyleEncoder(latent_dim, ngf=ngf, img_resolution=img_resolution, nc=nc, normalize_style=normalize_style, variational_style_encoder=variational_style_encoder)
        content_encoder = ContentEncoder(ngf=ngf, img_resolution=img_resolution, nc=nc)
        self.content_layers = content_encoder.layers
    
    def style_encode(self, input):
        return self.style_encoder(input)
    
    def content_encode(self, input):
        return self.content_layers(self.style_encoder.act_layers(input))
       
    def forward(self, input):
        act = self.style_encoder.act_layers(input)
        content = self.content_layers(act)
        excitations = []
        for exci, down in zip(self.style_encoder.excitation_blocks, self.style_encoder.down_blocks):
            act = down(act)
            excitations.append(exci(act))
        style = self.style_encoder.out(torch.cat(excitations, dim=1))
        return content, style

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, style_dim: int=512, ngf: int=128, nc=3, img_resolution=256, lite: bool=False):
        super().__init__()
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)
        
        mapping = [EqualLinear(latent_dim, style_dim, lr_mul=0.01, activation='fused_lrelu')]
        for _ in range(4):
            mapping.append(EqualLinear(style_dim, style_dim, lr_mul=0.01, activation='fused_lrelu'))
        self.mapping = nn.Sequential(*mapping)

        UpBlock = UpBlockSmall if lite else UpBlockBig
        UpBlockStyle = UpBlockStyleSmall if lite else UpBlockStyleBig
        self.sfeat_16  = UpBlockStyle(nfc[8],  nfc[16], style_dim)
        self.sfeat_32  = UpBlockStyle(nfc[16]*2, nfc[32], style_dim)
        self.sfeat_64  = UpBlockStyle(nfc[32]*2, nfc[64], style_dim)
        self.sfeat_128 = UpBlockStyle(nfc[64]*2, nfc[128], style_dim)
        self.sfeat_256 = UpBlockStyle(nfc[128], nfc[256], style_dim)

        if img_resolution > 256:
            self.sfeat_512 = UpBlockStyle(nfc[256], nfc[512], style_dim)
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

    def forward(self, content, z):
        o64, o32, o16, o8 = content
        style = self.mapping(z)
        feat_16 = self.sfeat_16(o8, style)
        feat_32 = self.sfeat_32(torch.cat([feat_16, o16], dim=1), style)
        feat_64 = self.sfeat_64(torch.cat([feat_32, o32], dim=1), style)
        feat_128 = self.sfeat_128(torch.cat([feat_64, o64], dim=1), style)
        feat_256 = self.sfeat_256(feat_128, style)
        feat_last = feat_256

        if self.img_resolution > 256:
            feat_512 = self.sfeat_512(feat_256, style)
            feat_last = feat_512

        if self.img_resolution > 512:
            feat_last = self.feat_1024(feat_512)

        return self.to_big(feat_last)


class Generator(nn.Module):
    def __init__(self, latent_dim: int=8, ngf: int=128, nc=3, img_resolution=256, lite: bool=False, normalize_style: bool=False, variational_style_encoder: bool=False, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.nc = nc
        self.img_resolution = img_resolution
        self.lite = lite
        self.variational_style_encoder = variational_style_encoder
        self.encoder = Encoder(latent_dim=latent_dim, ngf=ngf, nc=nc, img_resolution=img_resolution, lite=lite, normalize_style=normalize_style, variational_style_encoder=variational_style_encoder)
        self.decoder = Decoder(latent_dim=latent_dim, ngf=ngf, nc=nc, img_resolution=img_resolution, lite=lite)
    
    def style_encode(self, img):
        return self.encoder.style_encode(img)
    
    def content_encode(self, img):
        return self.encoder.content_encode(img)

    def encode(self, img):
        return self.encoder(img)

    def decode(self, content, style):
        return self.decoder(content, style)
    
    def forward(self, img, z=None):
        if z is not None:
            content = self.content_encode(img)
        else:
            content, z = self.encode(img)

        if self.variational_style_encoder:
            z_mu = z[:,:z.size(1)//2]
            z_logvar = z[:,z.size(1)//2:]
            z = gaussian_reparameterization(z_mu, z_logvar)

        return self.decode(content, z)