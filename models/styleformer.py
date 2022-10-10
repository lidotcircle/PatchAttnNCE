import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .patch_embed import EmbeddingStem
from .transformer import Transformer


class StyleFormer(torch.nn.Module):
    def __init__(
        self, 
        num_outputs: int,
        image_size = (256,256),
        input_channels: int=3,
        num_layers: int = 6, 
        num_heads: int = 4, 
        mlp_ratio: float = 4.0,
        attn_dropout_rate: float = 0.0,
        dropout_rate: float =0.0,
        qkv_bias: bool = True,
        use_revised_ffn: bool = False,
        ):
        super(StyleFormer, self).__init__()
        self.embedding_dim = 384
        self.num_outputs = num_outputs
        h, w = image_size
        assert h == w
        self.patch_len = 16 * 16
        self.patch_size = h // 16

        # embedding layer
        self.embedding_layer = EmbeddingStem(
            image_size=h,
            patch_size=self.patch_size,
            channels=input_channels,
            embedding_dim=self.embedding_dim,
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
            dim=self.embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )

        # transformer normalization
        self.post_transformer = torch.nn.LayerNorm(self.embedding_dim)

        # TODO num_outputs instead of 1
        self.token_merge = torch.nn.Linear(self.patch_len, 1)
        self.output_layer = torch.nn.Linear(self.embedding_dim, num_outputs)
    
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.post_transformer(x)
        x = x.permute(0, 2, 1)
        x = self.token_merge(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        x = self.output_layer(x)
        return x
    
    def random_output(self, batch_size, device):
        x = torch.rand([batch_size, self.embedding_dim]).to(device)
        x = F.normalize(x)
        return self.output_layer(x)