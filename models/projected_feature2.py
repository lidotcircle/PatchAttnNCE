import math
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import timm


class CrossChannelMixing(nn.Module):
    def __init__(self, in_channels, cout, expand=False):
        super().__init__()
        out_channels = [ cout*(2**i) for i in range(len(in_channels)) ] if expand else [cout]*len(in_channels)
        networks = []
        for inc, onc in zip(in_channels, out_channels):
            ccm = nn.Conv2d(inc, onc, kernel_size=1, stride=1, padding=0, bias=True)
            networks.append(ccm)
        self.channels = out_channels
        self.networks = nn.ModuleList(networks)
    
    def forward(self, features: List[torch.Tensor]):
        assert len(features) == len(self.channels)
        result = []
        for feat, net in zip(features, self.networks):
            result.append(net(feat))
        return result

class ScaleMerger(nn.Module):
    def __init__(self, shapes: List[Tuple[int,int,int]], cout):
        super().__init__()
        min_h, min_w = 1024, 1024
        for c, h, w in shapes:
            min_h = min(min_h, h)
            min_w = min(min_w, w)

        total_channels = 0
        self.prenets = nn.ModuleList()
        for shape in shapes:
            if shape[1]!=min_h or shape[2]!=min_w:
                out_nc = shape[0]*int(math.sqrt(shape[2]*shape[1]/(min_h*min_w)))
                self.prenets.append(nn.Conv2d(shape[0], out_nc, (shape[1]//min_h,shape[1]//min_w), stride=(shape[1]//min_h,shape[2]//min_w)))
                total_channels += out_nc
            else:
                self.prenets.append(nn.Sequential())
                total_channels += shape[0]
        self.out_shape = (min_h, min_w)
        self.outlayer =  nn.Conv2d(total_channels, cout, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, tensors: List[torch.Tensor]):
        aligned = []
        for ten, net in zip(tensors, self.prenets):
            aligned.append(net(ten))
        return self.outlayer(torch.cat(aligned, dim=1))

class FeatureMapper(nn.Module):
    def __init__(self, shapes: List[Tuple[int,int,int]], ccm_expand_scale: int, cout: int, expand: bool):
        super().__init__()
        ccm_cout = max(list(map(lambda x: x[0], shapes))) * ccm_expand_scale
        ccm = CrossChannelMixing(list(map(lambda x: x[0], shapes)), ccm_cout, expand=expand)
        merger = ScaleMerger(list(map(lambda x: (x[1],x[0][1],x[0][2]), zip(shapes, ccm.channels))), cout)
        self.out_shape = merger.out_shape
        self.layers = nn.Sequential(ccm, merger)
    
    def forward(self, features: List[torch.Tensor]):
        return self.layers(features)
    

class FeatureProjector(nn.Module):
    def __init__(self, model: Union[nn.Module,str], image_shape: Tuple[int,int,int], out_dim: int, expand: bool):
        super().__init__()
        self.model = get_model(model)
        input = torch.zeros(image_shape, dtype=torch.float).unsqueeze(0)
        features = self.model(input)
        in_shape = list(map(lambda feat: (feat.shape[1],feat.shape[2],feat.shape[3]), features))
        self.mapper = FeatureMapper(in_shape, 1, out_dim, expand=expand)
        self.out_shape = self.mapper.out_shape
    
    def forward(self, input):
        features = self.model(input)
        return self.mapper(features)

class ModuleWrapper(nn.Module):
    def __init__(self, moduleList, layers: List[int]):
        super().__init__()
        self.moduleList = moduleList
        self.layers = [ *layers ]
    
    def forward(self, x):
        ans = []
        for layer_id, layer in enumerate(self.moduleList):
            x = layer(x)
            if layer_id in self.layers:
                ans.append(x)
        return ans

def get_model(model: Union[nn.Module,str]) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    
    if model == 'efficientnet':
        model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
        return ModuleWrapper(_make_efficientnet(model), [0,1,2,3])
    else:
        raise NotImplementedError(model)

def _make_efficientnet(model):
    pretrained = nn.ModuleList()
    pretrained.append(nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2]))
    pretrained.append(nn.Sequential(*model.blocks[2:3]))
    pretrained.append(nn.Sequential(*model.blocks[3:5]))
    pretrained.append(nn.Sequential(*model.blocks[5:9]))
    return pretrained