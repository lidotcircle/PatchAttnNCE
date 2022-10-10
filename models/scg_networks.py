import torch.nn.functional as F 
import torch
from torch import nn
from pg_modules import blocks
from pg_modules.blocks import UpBlockBig, UpBlockSmall, DownBlock, SEBlock, conv2d
from pg_modules.networks_fastgan import FastganSynthesis


class Encoder(nn.Module):
    def __init__(self, ngf=128, img_resolution: int=256, num_outputs:int=256):
        super().__init__()
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        assert img_resolution in nfc
        self.init_layer = nn.Conv2d(3, nfc[img_resolution], kernel_size=7, stride=1, padding=3)
        DownBlock = blocks.DownBlock
        self.down_layers = nn.ModuleList()

        while img_resolution > 2:
            down = DownBlock(nfc[img_resolution], nfc[img_resolution//2])
            self.down_layers.append(down)
            img_resolution = img_resolution // 2
        
        self.out_layer = nn.Linear(nfc[img_resolution], num_outputs)
        
    def forward(self, input):
        x = self.init_layer(input)
        for _, module in enumerate(self.down_layers):
            x = module(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.out_layer(x)


class UncondGenerator(nn.Module):
    def __init__(self, ngf=128, img_resolution=256, z_dim=256, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.gen = FastganSynthesis(ngf=ngf, z_dim=z_dim, img_resolution=img_resolution,lite=False)

    def forward(self, img: torch.Tensor, **kwargs):
        z = torch.randn((img.size(0), 1, self.z_dim)).to(img.device)
        return self.gen(z, c=None)


class Generator(nn.Module):
    def __init__(self, ngf=128, img_resolution=256, z_dim=512):
        super().__init__()
        self.encoder = Encoder(ngf=ngf, img_resolution=img_resolution, num_outputs=z_dim)
        self.decoder = FastganSynthesis(ngf=ngf, z_dim=z_dim, img_resolution=img_resolution,lite=False)

    def forward(self, img, latent_out: list=None):
        latent = self.encoder(img)

        if latent_out is not None:
            latent_out.append(latent)

        return self.decoder(latent.unsqueeze(1), c=None)


class FGGenerator(nn.Module):
    def __init__(self, ngf=128, img_resolution=256, nc=3, lite=False, **kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        assert img_resolution in nfc
        self.init_layer = nn.Conv2d(3, nfc[img_resolution], kernel_size=7, stride=1, padding=3)
        if img_resolution > 512:
            self.dfeat_1024 = self.init_layer
            self.dfeat_512 = DownBlock(nfc[1024], nfc[512])
        elif img_resolution == 512:
            self.dfeat_512 = self.init_layer
        self.dfeat_256 = self.init_layer if img_resolution == 256 else DownBlock(nfc[512], nfc[256])
        self.dfeat_128 = DownBlock(nfc[256], nfc[128])
        self.dfeat_64 = DownBlock(nfc[128], nfc[64])
        self.dfeat_32 = DownBlock(nfc[64], nfc[32])
        self.dfeat_16 = DownBlock(nfc[32], nfc[16])
        self.dfeat_8 = DownBlock(nfc[16], nfc[8])

        UpBlock = UpBlockSmall if lite else UpBlockBig
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64  = SEBlock(nfc[8], nfc[64])
        self.se_128 = SEBlock(nfc[16], nfc[128])
        self.se_256 = SEBlock(nfc[32], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[64], nfc[512])
            self.se_512 = SEBlock(nfc[16], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, img, **kwargs):
        feat = img
        if self.img_resolution >= 1024:
            feat = self.dfeat_1024(feat)
        if self.img_resolution >= 512:
            feat = self.dfeat_512(feat)
        dfeat_256 = self.dfeat_256(feat)
        dfeat_128 = self.dfeat_128(dfeat_256)
        dfeat_64 = self.dfeat_64(dfeat_128)
        dfeat_32 = self.dfeat_32(dfeat_64)
        dfeat_16 = self.dfeat_16(dfeat_32)
        dfeat_8 = self.dfeat_8(dfeat_16)

        feat_16 = self.feat_16(dfeat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.se_64(dfeat_8, self.feat_64(feat_32))
        feat_128 = self.se_128(feat_16,  self.feat_128(feat_64))

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_32, self.feat_256(feat_last))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_64, self.feat_512(feat_last))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)

        return self.to_big(feat_last)

class FGGenerator2(nn.Module):
    def __init__(self, ngf=128, img_resolution=256, nc=3, lite=False, **kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        assert img_resolution in nfc
        self.init_layer = nn.Conv2d(3, nfc[img_resolution], kernel_size=7, stride=1, padding=3)
        if img_resolution > 512:
            self.dfeat_1024 = self.init_layer
            self.dfeat_512 = DownBlock(nfc[1024], nfc[512])
        elif img_resolution == 512:
            self.dfeat_512 = self.init_layer
        self.dfeat_256 = self.init_layer if img_resolution == 256 else DownBlock(nfc[512], nfc[256])
        self.dfeat_128 = DownBlock(nfc[256], nfc[128])
        self.dfeat_64 = DownBlock(nfc[128], nfc[64])
        self.dfeat_32 = DownBlock(nfc[64], nfc[32])
        self.dfeat_16 = DownBlock(nfc[32], nfc[16])

        UpBlock = UpBlockSmall if lite else UpBlockBig
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64  = SEBlock(nfc[16], nfc[64])
        self.se_128 = SEBlock(nfc[32], nfc[128])
        self.se_256 = SEBlock(nfc[64], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[64], nfc[512])
            self.se_512 = SEBlock(nfc[16], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, img, **kwargs):
        feat = img
        if self.img_resolution >= 1024:
            feat = self.dfeat_1024(feat)
        if self.img_resolution >= 512:
            feat = self.dfeat_512(feat)
        dfeat_256 = self.dfeat_256(feat)
        dfeat_128 = self.dfeat_128(dfeat_256)
        dfeat_64 = self.dfeat_64(dfeat_128)
        dfeat_32 = self.dfeat_32(dfeat_64)
        dfeat_16 = self.dfeat_16(dfeat_32)

        feat_32 = self.feat_32(dfeat_16)
        feat_64 = self.se_64(dfeat_16, self.feat_64(feat_32))
        feat_128 = self.se_128(dfeat_32,  self.feat_128(feat_64))

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(dfeat_64, self.feat_256(feat_last))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_64, self.feat_512(feat_last))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)

        return self.to_big(feat_last)