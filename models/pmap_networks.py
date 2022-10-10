from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, norm_layer=nn.InstanceNorm2d):
        super().__init__()

        planes = in_planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RecoverResNet(nn.Module):
    def __init__(self, nc: int, num_layers: int=3):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(BasicBlock(nc))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, fake):
        return self.layers(fake)


class RecoverConv1x1(nn.Module):
    def __init__(self, nc: int, num_layers: int=2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(nc, nc, 1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(nc, nc, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, fake):
        return self.layers(fake)

    
class AttnResnet(nn.Module):
    def __init__(self, nc: int, num_layers: int, output_shape: Tuple[int, int], temperature: float=8):
        super().__init__()
        self.output_shape = output_shape
        self.temperature = temperature
        layers = []
        for _ in range(num_layers):
            layers.append(BasicBlock(nc))
        layers.append(nn.Conv2d(nc, 1, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, feat: torch.Tensor):
        nh, nw = self.output_shape
        feat = F.interpolate(feat, self.output_shape)
        feat = self.layers(feat)
        feat = feat / self.temperature
        feat = feat.exp()
        sum = feat.sum(dim=[2,3]).unsqueeze(2).unsqueeze(3).expand(-1, -1, feat.size(2), feat.size(2))
        feat = (feat / sum) * (nh * nw)
        feat = feat.clamp(min = 0, max = 5)
        return feat


class AttnConv1x1(nn.Module):
    def __init__(self, nc: int, num_layers: int, output_shape: Tuple[int, int], temperature: float=8):
        super().__init__()
        self.output_shape = output_shape
        self.temperature = temperature
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(nc, nc, 1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(nc, 1, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, feat: torch.Tensor):
        nh, nw = self.output_shape
        feat = F.interpolate(feat, self.output_shape)
        feat = self.layers(feat)
        feat = feat / self.temperature
        feat = torch.log(F.relu(feat) + 1.01)
        sum = feat.sum(dim=[2,3]).unsqueeze(2).unsqueeze(3).expand(-1, -1, feat.size(2), feat.size(3))
        feat = (feat / sum) * (nh * nw)
        feat = feat.clamp(min = 0.01, max = 100)
        return feat


class AttnNone(nn.Module):
    def __init__(self,  output_shape: Tuple[int, int], **kwargs):
        super().__init__()
        self.output_shape = output_shape
   
    def forward(self, feat: torch.Tensor):
        nh, nw = self.output_shape
        return torch.ones([feat.size(0), 1, nh, nw], dtype=feat.dtype).to(feat.device)


class PMapNet(nn.Module):
    def __init__(self, attn_net='conv1x1', map_net='resnet', attn_layers: int=2, map_layers: int=2, attn_temperature: float=8, **kwargs):
        super().__init__()
        self.attn_net_type = attn_net
        self.map_net_type = map_net
        self.attn_layers = attn_layers
        self.map_layers = map_layers
        self.attn_temperature = attn_temperature

    def setup(self, feats: List[torch.Tensor], attn_feats: List[torch.Tensor]):
        assert len(feats) == len(attn_feats)
        for i, feat, attn in zip(range(len(feats)), feats, attn_feats):
            _, _, h, w = feat.shape
            if self.attn_net_type == 'conv1x1':
                attn_net = AttnConv1x1(attn.size(1), self.attn_layers, (h, w), self.attn_temperature)
            elif self.attn_net_type == 'resnet':
                attn_net = AttnResnet(attn.size(1), self.attn_layers, (h, w), self.attn_temperature)
            elif self.attn_net_type == 'none':
                attn_net = AttnNone((h, w))
            else:
                raise NotImplemented(self.attn_net_type)
            setattr(self, f'attn_layer_{i}', attn_net)

            if self.map_net_type == 'conv1x1':
                map_net = RecoverConv1x1(feat.size(1), self.map_layers)
            elif self.map_net_type == 'resnet':
                map_net = RecoverResNet(feat.size(1), self.map_layers)
            setattr(self, f'recover_net_{i}', map_net)

    def forward(
        self,
        real_feats: List[torch.Tensor],
        real_attn_feats: List[torch.Tensor],
        fake_feats: List[torch.Tensor]
        ):
        loss = 0
        for i, real_feat, attn_feat, fake_feat in zip(range(len(real_feats)), real_feats, real_attn_feats, fake_feats):
            attn_net = getattr(self, f'attn_layer_{i}')
            rec_net = getattr(self, f'recover_net_{i}')
            attn_map: torch.Tensor = attn_net(attn_feat)
            rec_feat = rec_net(fake_feat)
            diff: torch.Tensor = (rec_feat - real_feat) * attn_map.expand(-1, rec_feat.size(1), -1, -1)
            loss += diff.abs().mean()
        return loss / len(real_feats)


class PMapAttention(nn.Module):
    def __init__(self, attn_net='conv1x1', attn_layers: int=2, attn_temperature: float=8, **kwargs):
        super().__init__()
        self.attn_net_type = attn_net
        self.attn_layers = attn_layers
        self.attn_temperature = attn_temperature

    def setup(self, feats: List[torch.Tensor], attn_feats: List[torch.Tensor]):
        assert len(feats) == len(attn_feats)
        for i, feat, attn in zip(range(len(feats)), feats, attn_feats):
            _, _, h, w = feat.shape
            if self.attn_net_type == 'conv1x1':
                attn_net = AttnConv1x1(attn.size(1), self.attn_layers, (h, w), self.attn_temperature)
            elif self.attn_net_type == 'resnet':
                attn_net = AttnResnet(attn.size(1), self.attn_layers, (h, w), self.attn_temperature)
            elif self.attn_net_type == 'none':
                attn_net = AttnNone((h, w))
            else:
                raise NotImplemented(self.attn_net_type)
            setattr(self, f'attn_layer_{i}', attn_net)

    def forward(self, attn_feats: List[torch.Tensor]):
        attn_values = []
        for i, attn_feat in zip(range(len(attn_feats)), attn_feats):
            attn_net = getattr(self, f'attn_layer_{i}')
            attn_map: torch.Tensor = attn_net(attn_feat)
            attn_values.append(attn_map)
        return attn_values