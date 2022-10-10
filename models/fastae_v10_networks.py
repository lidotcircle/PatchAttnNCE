import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils import spectral_norm


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def embedding(*args, **kwargs):
    return spectral_norm(nn.Embedding(*args, **kwargs))

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

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

class ForwardWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, input):
        x = input if not isinstance(input, list) else input[-1]
        input = [] if not isinstance(input, list) else input
        input.append(self.module(x))
        return input

class Encoder(nn.Module):
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
        layers += [ ForwardWrapper(DownBlock(nfc[128], nfc[64], 'layer')) ]
        layers += [ ForwardWrapper(DownBlock(nfc[64], nfc[32], 'layer')) ]
        layers += [ ForwardWrapper(DownBlock(nfc[32], nfc[16], 'layer')) ]
        layers += [ ForwardWrapper(DownBlock(nfc[16], nfc[8], 'layer')) ]
        layers += [ ForwardWrapper(DownBlock(nfc[8],  nfc[4], 'layer')) ]
        self.layers = nn.Sequential(*layers)

    def forward(self, img, **kwargs):
        return self.layers(self.act_layers(img))


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)

class Decoder(nn.Module):
    def __init__(self, ngf: int=128, nc=3, img_resolution=256, lite: bool=False, unet_layers: list = None):
        super().__init__()
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)
        
        self.unet_layers = unet_layers

        UpBlock = UpBlockSmall if lite else UpBlockBig
        self.feat_8   = UpBlock(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8] * (2 if 8 in unet_layers else 1), nfc[16])
        self.feat_32  = UpBlock(nfc[16]* (2 if 16 in unet_layers else 1), nfc[32])
        self.feat_64  = UpBlock(nfc[32]* (2 if 32 in unet_layers else 1), nfc[64])
        self.feat_128 = UpBlock(nfc[64]* (2 if 64 in unet_layers else 1), nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

    def forward(self, content):
        o64, o32, o16, o8, o4 = content
        feat_8 = self.feat_8(o4)
        feat_16 = self.feat_16(torch.cat([feat_8, o8], dim=1) if 8 in self.unet_layers else feat_8 )
        feat_32 = self.feat_32(torch.cat([feat_16, o16], dim=1) if 16 in self.unet_layers else feat_16)
        feat_64 = self.se_64(o4, self.feat_64(torch.cat([feat_32, o32], dim=1) if 32 in self.unet_layers else feat_32))
        feat_128 = self.se_128(feat_8, self.feat_128(torch.cat([feat_64, o64], dim=1) if 64 in self.unet_layers else feat_64))
        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))
        feat_last = feat_256

        if self.img_resolution > 256:
            feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
            feat_last = feat_512

        if self.img_resolution > 512:
            feat_last = self.feat_1024(feat_512)

        return self.to_big(feat_last)


class Generator(nn.Module):
    def __init__(self, ngf: int=128, nc=3, img_resolution=256, lite: bool=False, unet_layers: list=None, **kwargs):
        super().__init__()
        self.ngf = ngf
        self.nc = nc
        self.img_resolution = img_resolution
        self.lite = lite
        self.encoder = Encoder(ngf=ngf, nc=nc, img_resolution=img_resolution)
        self.decoder = Decoder(ngf=ngf, nc=nc, img_resolution=img_resolution, lite=lite, unet_layers=unet_layers)
    
    def encode(self, img):
        return self.encoder(img)

    def decode(self, content):
        return self.decoder(content)
    
    def forward(self, img, z=None):
        content = self.encode(img)
        return self.decode(content)