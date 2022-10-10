import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import spectral_norm


class CamWeightNet(nn.Module):
    def __init__(self, ap_weight: float=0.5, weight_multi: bool=False, sigmoid: bool=False, detach: bool=False):
        super().__init__()
        assert ap_weight >= 0 and ap_weight <= 1
        self.ap_weight = ap_weight
        self.weight_multi = weight_multi
        self.sigmoid_act = sigmoid
        self.detach = detach
    
    def setup(self, attn_feats, feats):
        assert len(attn_feats) == len(feats)
        max_h, max_w = 0, 0
        for afeat in attn_feats:
            _, _, fh, fw = afeat.shape
            max_h, max_w = max(max_h, fh), max(max_w, fw)

        for i, attn_feat, feat in zip(range(len(attn_feats)), attn_feats, feats):
            _, _, ah, aw = attn_feat.shape
            _, _, h, w = feat.shape
            if self.weight_multi:
                setattr(self, f'f_shape_{i}', (h, w))
                h, w = max_h, max_w
            gap_ln = spectral_norm(nn.Linear(attn_feat.size(1), 1))
            gmp_ln = spectral_norm(nn.Linear(attn_feat.size(1), 1))
            setattr(self, f'gap_ln_{i}', gap_ln)
            setattr(self, f'gmp_ln_{i}', gmp_ln)
            setattr(self, f'interp_{i}', ah != h or aw != w)
            setattr(self, f'shape_{i}', (h, w))
        if self.weight_multi:
            self.rho = Parameter(torch.Tensor(1, len(attn_feats), 1, 1))
            self.rho.data.fill_(1)
    
    def forward(self, attn_feats, logit_only: bool=False):
        logits_list = []
        attn_map_list = []

        for i, attn_feat in enumerate(attn_feats):
            gap_ln: nn.Linear = getattr(self, f'gap_ln_{i}')
            gmp_ln: nn.Linear = getattr(self, f'gmp_ln_{i}')
            should_interp = getattr(self, f'interp_{i}')

            ap_val = F.adaptive_avg_pool2d(attn_feat, output_size=1)
            mp_val = F.adaptive_max_pool2d(attn_feat, output_size=1)
            ap_val = ap_val.view(ap_val.size(0), -1)
            mp_val = mp_val.view(mp_val.size(0), -1)
            gap_logits = gap_ln(ap_val)
            gmp_logits = gmp_ln(ap_val)
            logits = torch.cat([gap_logits, gmp_logits], dim=1)
            logits_list.append(logits)
            if logit_only:
                continue

            gap_ln_param = list(gap_ln.parameters())[1].unsqueeze(2).unsqueeze(3)
            gmp_ln_param = list(gmp_ln.parameters())[1].unsqueeze(2).unsqueeze(3)
            gap_map = (attn_feat * gap_ln_param).sum(dim=1).unsqueeze(1)
            gmp_map = (attn_feat * gmp_ln_param).sum(dim=1).unsqueeze(1)

            if should_interp:
                h, w = getattr(self, f'shape_{i}')
                gap_map = F.interpolate(gap_map, (h, w))
                gmp_map = F.interpolate(gmp_map, (h, w))

            gap_map = gap_map.view(gap_map.size(0), gap_map.size(2), gap_map.size(3))
            gmp_map = gmp_map.view(gmp_map.size(0), gmp_map.size(2), gmp_map.size(3))

            attnmap: torch.Tensor = gap_map * self.ap_weight + gmp_map * (1 - self.ap_weight)
            if self.sigmoid_act:
                attnmap = F.sigmoid(attnmap)
            else:
                amin = attnmap.min(dim=2).values.min(dim=1).values
                attnmap = attnmap - amin.unsqueeze(1).unsqueeze(2)
                amax = attnmap.max(dim=2).values.max(dim=1).values
                attnmap = attnmap / amax.unsqueeze(1).unsqueeze(2)
            
            if self.detach:
                attnmap = attnmap.detach()

            oldshape= attnmap.shape
            attnmap = attnmap.view(attnmap.size(0), -1)
            attnmap = F.normalize(attnmap, p=2, dim=1)
            attnmap = attnmap.view(oldshape)
            if self.weight_multi:
                attnmap = attnmap.unsqueeze(1)
            attn_map_list.append(attnmap)

        if self.weight_multi:
            maps = torch.cat(attn_map_list, dim=1)
            weights = F.normalize(torch.log(self.rho.abs() + 1), p=1, dim=1)
            maps = weights.expand(maps.shape) * maps
            attn_map = maps.sum(dim=1).unsqueeze(1)
            new_attn_map_list = []
            for i, _ in enumerate(attn_map_list):
                true_shape = getattr(self, f'f_shape_{i}')
                new_attn_map_list.append(F.interpolate(attn_map, true_shape).flatten(0,1))
            attn_map_list = new_attn_map_list
        
        return attn_map_list, logits_list