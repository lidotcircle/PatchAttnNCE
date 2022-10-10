import torch.nn.functional as F 
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
        super(ILN, self).__init__()
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
        return nn.LayerNorm(c)
    elif mode == 'iln':
        return ILN(c)

class AdaIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps
        self.num_features = num_features

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out = (input - in_mean) / torch.sqrt(in_var + self.eps)
        gamma = gamma.expand(1, 1, -1, -1).permute(2, 3, 0, 1)
        beta = beta.expand(1, 1, -1, -1).permute(2, 3, 0, 1)
        out = out * gamma.expand(input.shape[0], -1, -1, -1) + beta.expand(input.shape[0], -1, -1, -1)

        return out

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

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
        self.sty2params = nn.Sequential(
            linear(style_dim, out_planes * 2),
            nn.LeakyReLU(0.3),
            linear(out_planes * 2, out_planes * 2 * 4)
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.adaIN1 = AdaIN(out_planes*2)
        self.glu1 = GLU()
        self.conv2 = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)
        self.adaIN2 = AdaIN(out_planes*2)
        self.glu2 = GLU()
        
    def forward(self, input, style):
        params = self.sty2params(style)
        params = params.view(params.size(0), 4, -1)
        gamma1, beta1, gamma2, beta2 = params[:,0], params[:,1], params[:,2], params[:,3]
        x = self.up(input)
        x = self.conv1(x)
        x = self.adaIN1(x, gamma1, beta1)
        x = self.glu1(x)
        x = self.conv2(x)
        x = self.adaIN2(x, gamma2, beta2)
        return self.glu2(x)

class UpBlockStyleSmall(nn.Module):
    def __init__(self, in_planes, out_planes, style_dim):
        super().__init__()
        self.sty2params = nn.Sequential(
            linear(style_dim, out_planes * 2),
            nn.LeakyReLU(0.3),
            linear(out_planes * 2, out_planes * 2 * 2)
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.adaIN = AdaIN(out_planes*2)
        self.glu = GLU()
       
    def forward(self, input, style):
        params = self.sty2params(style)
        params = params.view(params.size(0), 2, -1)
        gamma, beta = params[:,0], params[:,1]
        x = self.up(input)
        x = self.conv(x)
        x = self.adaIN(x, gamma, beta)
        return self.glu(x)


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
        layers = []
        layers += [ nn.Conv2d(nc, nfc[img_resolution], kernel_size=7, stride=1, padding=3) ]
        if img_resolution > 512:
            layers += [ DownBlock(nfc[1024], nfc[512], 'instance') ]
        elif img_resolution == 512:
            layers += [ DownBlock(nfc[512], nfc[256], 'instance') ]

        layers += [ DownBlock(nfc[256], nfc[128], 'instance') ]
        layers += [ DownBlock(nfc[128], nfc[64], 'instance') ]
        layers += [ DownBlock(nfc[64], nfc[32], 'instance') ]
        layers += [ DownBlock(nfc[32], nfc[16], 'instance') ]

        self.layers = nn.Sequential(*layers)

    def forward(self, img, **kwargs):
        return self.layers(img)


class StyleEncoder(nn.Module):
    def __init__(self, style_dim, lite: bool=False, ngf=64, img_resolution=256, nc=3):
        super().__init__()
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        assert img_resolution in nfc
        layers = []
        layers += [ nn.Conv2d(nc, nfc[img_resolution], kernel_size=7, stride=1, padding=3) ]
        if img_resolution > 512:
            layers += [ DownBlock(nfc[1024], nfc[512], 'iln') ]
        elif img_resolution == 512:
            layers += [ DownBlock(nfc[512], nfc[256], 'iln') ]

        layers += [ DownBlock(nfc[256], nfc[128], 'iln') ]
        layers += [ DownBlock(nfc[128], nfc[64], 'iln') ]
        layers += [ DownBlock(nfc[64], nfc[32], 'iln') ]
        layers += [ DownBlock(nfc[32], nfc[16], 'iln') ]
        layers += [ DownBlock(nfc[16], nfc[8], 'iln') ]
        layers += [ DownBlock(nfc[8], nfc[4], 'iln') ]
        if not lite:
            layers += [
                nn.Flatten(),
                linear(4 * 4 * nfc[4], nfc[4]),
                nn.LeakyReLU(0.1),
                linear(nfc[4], style_dim)
            ]
        else:
            layers += [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                linear(nfc[4], nfc[4]),
                nn.LeakyReLU(0.1),
                linear(nfc[4], style_dim)
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, img, **kwargs):
        return self.layers(img)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, ngf=128, nc: int=3, img_resolution: int=256, lite: bool=False):
        super().__init__()
        self.style_encoder = StyleEncoder(latent_dim, lite=lite, ngf=ngf//2, img_resolution=img_resolution, nc=nc)
        self.content_encoder = ContentEncoder(ngf=ngf, img_resolution=img_resolution, nc=nc)
    
    def style_encode(self, input):
        return self.style_encoder(input)
    
    def content_encode(self, input):
        return self.content_encoder(input)
       
    def forward(self, input):
        return self.content_encode(input), self.style_encode(input)


class Decoder(nn.Module):
    def __init__(self, style_dim: int, ngf: int=128, nc=3, img_resolution=256, lite: bool=False):
        super().__init__()
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        UpBlock = UpBlockSmall if lite else UpBlockBig
        UpBlockStyle = UpBlockStyleSmall if lite else UpBlockStyleBig
        self.sfeat_32  = UpBlockStyle(nfc[16], nfc[32], style_dim)
        self.sfeat_64  = UpBlockStyle(nfc[32], nfc[64], style_dim)
        self.sfeat_128 = UpBlockStyle(nfc[64], nfc[128], style_dim)
        self.sfeat_256 = UpBlockStyle(nfc[128], nfc[256], style_dim)

        self.se_64  = SEBlock(nfc[16], nfc[64])
        self.se_128 = SEBlock(nfc[32], nfc[128])
        self.se_256 = SEBlock(nfc[64], nfc[256])

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[64], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

    def forward(self, content, style):
        feat_32 = self.sfeat_32(content, style)
        feat_64 = self.se_64(content, self.sfeat_64(feat_32, style))
        feat_128 = self.se_128(feat_32, self.sfeat_128(feat_64, style))
        feat_256 = self.se_256(feat_64, self.sfeat_256(feat_128, style))
        feat_last = feat_256

        if self.img_resolution > 256:
            feat_512 = self.se_512(feat_64, self.feat_512(feat_256))
            feat_last = feat_512

        if self.img_resolution > 512:
            feat_last = self.feat_1024(feat_512)

        return self.to_big(feat_last)


class Generator(nn.Module):
    def __init__(self, style_dim: int=512, ngf: int=128, nc=3, img_resolution=256, lite: bool=False, **kwargs):
        super().__init__()
        self.latent_dim = style_dim
        self.ngf = ngf
        self.nc = nc
        self.img_resolution = img_resolution
        self.lite = lite
        self.encoder = Encoder(latent_dim=style_dim, ngf=ngf, nc=nc, img_resolution=img_resolution, lite=lite)
        self.decoder = Decoder(style_dim=style_dim, ngf=ngf, nc=nc, img_resolution=img_resolution, lite=lite)
    
    def style_encode(self, img):
        return self.encoder.style_encode(img)
    
    def content_encode(self, img):
        return self.encoder.content_encode(img)

    def encode(self, img):
        return self.content_encode(img), self.style_encode(img)

    def decode(self, content, style):
        return self.decoder(content, style)
    
    def forward(self, img, z=None):
        content = self.content_encode(img)
        z = z if z is not None else self.style_encode(img)
        return self.decode(content, z)