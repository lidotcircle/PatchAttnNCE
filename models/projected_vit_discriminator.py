import torch.nn as nn
import torch.nn.functional as F
from .transformer_sn import VisionTransformer
from .projected_feature2 import FeatureProjector


class ProjectedViTDiscriminator(nn.Module):
    def __init__(self, vit_dim: int=512, **kwargs):
        super().__init__()
        self.projector = FeatureProjector('efficientnet', (3,256,256), out_dim=1024, expand=True)
        h, w = self.projector.out_shape
        patch_size=(max(1,h//16), max(1,w//16))
        self.vit = VisionTransformer(channels=1024, shape=self.projector.out_shape, patch_size=patch_size, dim=vit_dim, depth=6, heads=4, out_normalize=False, linear_classifier=True)
    
    def forward(self, img):
        _, _, h, w = img.shape
        if h != 256 or w != 256:
            img = F.interpolate(img, (256,256), mode='bilinear')

        projected = self.projector(img)
        out = self.vit(projected)
        return out.view(out.size(0), -1)