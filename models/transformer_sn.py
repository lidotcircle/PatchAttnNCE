from typing import Tuple
from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = spectral_norm(nn.Linear(dim, dim * 3, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = spectral_norm(nn.Linear(dim, dim))
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """
    Implementation of MLP for transformer
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0, revised=False):
        super(FeedForward, self).__init__()
        if not revised:
            """
            Original: https://arxiv.org/pdf/2010.11929.pdf
            """
            self.net = nn.Sequential(
                spectral_norm(nn.Linear(dim, hidden_dim)),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                spectral_norm(nn.Linear(hidden_dim, dim)),
            )
        else:
            """
            Scaled ReLU: https://arxiv.org/pdf/2109.03810.pdf
            """
            self.net = nn.Sequential(
                spectral_norm(nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1)),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                spectral_norm(nn.Conv1d(hidden_dim, dim, kernel_size=1, stride=1)),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            )

        self.revised = revised
        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x):
        if self.revised:
            x = x.permute(0, 2, 1)
            x = self.net(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.net(x)

        return x


class OutputLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=1000,
        representation_size=None,
        cls_head=False,
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes
        modules = []
        if representation_size:
            modules.append(spectral_norm(nn.Linear(embedding_dim, representation_size)))
            modules.append(nn.Tanh())
            modules.append(spectral_norm(nn.Linear(representation_size, num_classes)))
        else:
            modules.append(spectral_norm(nn.Linear(embedding_dim, num_classes)))

        self.net = nn.Sequential(*modules)

        if cls_head:
            self.to_cls_token = nn.Identity()

        self.cls_head = cls_head
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                if module.weight.shape[0] == self.num_classes:
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.cls_head:
            x = self.to_cls_token(x[:, 0])
        else:
            """
            Scaling Vision Transformer: https://arxiv.org/abs/2106.04560
            """
            x = torch.mean(x, dim=1)

        return self.net(x)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        qkv_bias=True,
        revised=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x, features = []):
        features = features or []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            features.append(x)
        
        return x


from .patch_embed_sn import EmbeddingStem

class VisionTransformer(nn.Module):
    def __init__(self, channels: int, shape: Tuple[int,int], patch_size: Tuple[int,int], dim: int, depth: int, heads: int, out_normalize: bool=True, linear_classifier: bool=False):
        super().__init__()
        self.out_resize = (shape[0]//patch_size[0], shape[1]//patch_size[1])

        # embedding layer
        self.embedding_layer = EmbeddingStem(
            image_size=shape,
            patch_size=patch_size,
            channels=channels,
            embedding_dim=dim,
            hidden_dims=None,
            conv_patch=True,
            linear_patch=False,
            conv_stem=False,
            conv_stem_original=True,
            conv_stem_scaled_relu=False,
            position_embedding_dropout=0,
            cls_head=False,
        )

        # transformer
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=4.0,
            attn_dropout=0,
            dropout=0,
            qkv_bias=True,
            revised=False,
        )

        # transformer normalization
        self.out_normalize = out_normalize
        if out_normalize:
            self.post_transformer = torch.nn.LayerNorm(dim)

        self.linear_classifier = linear_classifier
        if linear_classifier:
            self.linear_classifier_layer = spectral_norm(nn.Linear(dim, 1, False))
    
    def forward(self, input):
        embeded_input = self.embedding_layer(input)
        x = self.transformer(embeded_input)
        if self.out_normalize:
            x = self.post_transformer(x)
        if self.linear_classifier:
            x = self.linear_classifier_layer(x)
        return x.view(x.size(0), *self.out_resize, -1).permute(0, 3, 1, 2)