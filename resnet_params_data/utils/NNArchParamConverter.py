# handle conversion between NN Arch & Params and input/output of Diffusion model
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPosEmbed(nn.Module):
    def __init__(self, embed_dim, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.embed_dim = embed_dim // 2
        self.normalize = normalize
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.embed_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding= 1, groups = embed_dim)
        
    def forward(self, x):
        b, n, c = x.shape
        patch_n = int((n-1) ** 0.5)
        not_mask = torch.ones((b, patch_n, patch_n), device = x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.embed_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embed_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # B, C, H, W
        pos = self.embed_layer(pos).reshape(b, c, -1).transpose(1, 2)
        pos_cls = torch.zeros((b, 1, c), device = x.device)
        pos =  torch.cat((pos_cls, pos),dim=1)
        return pos + x


def ResNetToFullFlatten(model):
    pass


def FullFlattenToResNet(params_seq, ref_model):
    pass
