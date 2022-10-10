from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cyclegan_networks import init_net


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)


class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)

class TensorView(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, tensor: torch.Tensor):
        return tensor.view(*self.args, **self.kwargs)

class TensorPermute(nn.Module):
    def __init__(self, *args, contiguous: bool=False, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.contiguous = contiguous
    
    def forward(self, tensor: torch.Tensor):
        tensor = tensor.permute(*self.args, **self.kwargs)
        return tensor.contiguous() if self.contiguous else tensor

class PatchSampleF(nn.Module):
    def __init__(self, mlp_layers: int=2, max_shape: Tuple[int,int]=(256,256), use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], **kwargs):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.max_shape = max_shape
        self.mlp_layers = mlp_layers
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        mh, mw = self.max_shape

        for mlp_id, feat in enumerate(feats):
            assert len(feat.shape) == 4
            _, input_nc, h, w = feat.shape

            pre = []
            if h > mh or w > mw:
                assert h % mh == 0 and w % mw == 0
                pre += [
                    TensorView(-1, input_nc, h // mh, mh, w // mw, mw),
                    TensorPermute(0, 1, 2, 4, 3, 5, contiguous=True),
                    TensorView(-1, input_nc*(h//mh)*(w//mw), mh, mw),
                ]
                input_nc = input_nc*(h//mh)*(w//mw)
            setattr(self, f'pre_{mlp_id}', nn.Sequential(*pre))

            mlp_list = [nn.Linear(input_nc, self.nc)]
            for i in range(1, self.mlp_layers):
                mlp_list += [  nn.ReLU(), nn.Linear(self.nc, self.nc)]
            mlp = nn.Sequential(*mlp_list)
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            pre = getattr(self, f'pre_{feat_id}')
            feat = pre(feat)

            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class AttnPatchSampleF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], **kwargs):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        raw_feats, attn_feats = feats

        for mlp_id, feat, attn_feat in zip(range(len(raw_feats)), raw_feats, attn_feats):
            assert len(feat.shape) == 4
            _, input_nc, _, _ = feat.shape
            mlp_list = [ nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc) ]
            mlp = nn.Sequential(*mlp_list)
            setattr(self, 'mlp_%d' % mlp_id, mlp)

            attn_dim = attn_feat.shape[1]
            LNClassifier = [ nn.Linear(attn_dim, 1), nn.Sigmoid() ]
            lnc = nn.Sequential(*LNClassifier)
            setattr(self, 'lnc_%d' % mlp_id, lnc)
 
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, weights_strategy: str=None):
        assert num_patches > 0
        if patch_ids is not None:
            patch_ids_raw, patch_ids_attn = patch_ids
        else:
            patch_ids_raw, patch_ids_attn = None, None
        return_ids_raw, return_ids_attn = [], []
        return_ids = (return_ids_raw, return_ids_attn)
        return_feats_raw, return_feats_attn = [], []
        return_feats = (return_feats_raw, return_feats_attn)
        return_weights_raw, return_weights_attn = [], []
        return_weights = (return_weights_raw, return_weights_attn)
        if not self.mlp_init:
            self.create_mlp(feats)

        raw_feats, attn_feats = feats
        for feat_id, feat, attn_feat in zip(range(len(raw_feats)), raw_feats, attn_feats):
            device = feat.device
            _, _, fh, fw = feat.shape
            feat = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if patch_ids_raw is not None:
                patch_id = patch_ids_raw[feat_id]
            else:
                patch_id = np.random.permutation(feat.shape[1])
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
            patch_id = torch.tensor(patch_id, dtype=torch.long, device=device)
            return_ids_raw.append(patch_id)
            x_sample = feat[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            mlp = getattr(self, 'mlp_%d' % feat_id)
            x_sample = mlp(x_sample)
            x_sample = self.l2norm(x_sample)
            return_feats_raw.append(x_sample)

            attn_feat_old = attn_feat.permute(0, 2, 3, 1)
            attn_feat = attn_feat_old.flatten(1, 2)
            if patch_ids_attn is not None:
                patch_id_attn = patch_ids_attn[feat_id]
            else:
                patch_id_attn = np.random.permutation(attn_feat.shape[1])
                patch_id_attn = patch_id_attn[:int(min(64, patch_id_attn.shape[0]))]  # .to(patch_ids.device)
            patch_id_attn = torch.tensor(patch_id_attn, dtype=torch.long, device=device)
            return_ids_attn.append(patch_id_attn)
            attn_sample = attn_feat[:, patch_id_attn, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            attn_sample = self.l2norm(attn_sample)
            return_feats_attn.append(attn_sample)

            if weights_strategy == 'half_learned':
                lnc = getattr(self, 'lnc_%d' % feat_id)
                weights = lnc(attn_feat_old).flatten(2,3)
                v1 = weights.flatten(1,2)
                w = weights.mean(dim=1).mean(dim=1)
                w = w.view(w.size(0), 1, 1)
                weights = 0.5 * weights / w.expand(weights.shape)
                return_weights_attn.append(v1[:, patch_id_attn].flatten(0, 1))

                _, wh, ww = weights.shape
                if wh != fh:
                    assert fh % wh == 0
                    weights = torch.repeat_interleave(weights, fh // wh, 1)
                if ww != fw:
                    assert fw % ww == 0
                    weights = torch.repeat_interleave(weights, fw // ww, 2)
                v2 = 1 - weights.flatten(1,2)
                return_weights_raw.append(v2[:, patch_id].flatten(0,1))
        return return_feats, return_weights, return_ids