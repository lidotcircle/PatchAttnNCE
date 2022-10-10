import torch.nn as nn
import torch.nn.functional as F
from .styleformer import StyleFormer
from .projected_feature import FeatureProjector
from dnnlib import EasyDict


class StyleEncoder(nn.Module):
    def __init__(self, latent_dim: int, model: str='vit', backbone_kwargs={}):
        super().__init__()
        self.projector = FeatureProjector('efficientnet', (3,256,256), out_dim=1024, expand=True)
        if model == 'vit':
            kwargs = EasyDict(num_outputs=latent_dim, input_channels=1024, num_layers=4, image_size=self.projector.out_shape)
            kwargs = { **kwargs, **backbone_kwargs }
            self.model = StyleFormer(**kwargs)
        else:
            raise NotImplementedError(model)
    
    def forward(self, img):
        _, _, h, w = img.shape
        if h != 256 or w != 256:
            img = F.interpolate(img, (256,256), mode='bilinear')

        projected = self.projector(img)
        return self.model(projected)