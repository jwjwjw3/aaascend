import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, DropPath, _cfg
from timm.models.layers import lecun_normal_, trunc_normal_, to_2tuple
from timm.models.helpers import named_apply
from functools import partial
from collections import OrderedDict
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.get_v = nn.Conv2d(head_dim, head_dim, kernel_size=3, stride=1, padding=1,groups=head_dim)
        nn.init.zeros_(self.get_v.weight)
        nn.init.zeros_(self.get_v.bias)
        
    # def get_local_pos_embed(self, x):
    #     B, _, N, C = x.shape
    #     H = W = int(np.sqrt(N-1))
    #     x = x[:, :, 1:].transpose(-2, -1).contiguous().reshape(B * self.num_heads, -1, H, W)
    #     local_pe = self.get_v(x).reshape(B, -1, C, N-1).transpose(-2, -1).contiguous() # B, H, N-1, C
    #     local_pe = torch.cat((torch.zeros((B, self.num_heads, 1, C), device=x.device), local_pe), dim=2) # B, H, N, C
    #     return local_pe
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # local_pe = self.get_local_pos_embed(v) 
        # x = ((attn @ v + local_pe)).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class BNResFormer(nn.Module):
    def __init__(self, input_dim,  output_dim, input_mlp_dims=[4, 8], embed_dim=12,
                 depth=8, num_heads=4, add_pos_emb=True, mlp_ratio=4., qkv_bias=True, in_out_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='', flat_output=True):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.flat_output = flat_output
        # init input encoding MLP layers
        self.add_pos_emb = add_pos_emb
        if self.add_pos_emb:
            input_dim = input_dim + 1
        self.input_mlp = self.init_mlp([input_dim, *input_mlp_dims, embed_dim], bias=in_out_bias, last_activation=True)
        # init transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[ 
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # init norm layers
        self.norm = norm_layer(embed_dim)
        # init output MLP layers
        self.output_mlp = self.init_mlp([embed_dim, *input_mlp_dims[::-1], output_dim], bias=in_out_bias, last_activation=False)
        # init params
        self.init_weights(weight_init)
        
    def init_mlp(self, mlp_dims, bias=True, last_activation=False):
        mlp_layers = []
        for i in range(len(mlp_dims)-1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1], bias=bias))
            if i < len(mlp_dims)-2 or last_activation == True:
                mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)

    def forward(self, x):
        if self.add_pos_emb:
            _pos_emb = torch.linspace(-1, 1, 4).repeat_interleave(int(x.shape[1]/4))
            _pos_emb = torch.tile(_pos_emb, (x.shape[0], 1)).to(x.device)
            assert len(x.shape) == 2
            x = torch.stack([x, _pos_emb], dim=2)
        elif len(x.shape) < 3:
            x = torch.unsqueeze(x, -1)
        x = self.input_mlp(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.output_mlp(x)
        if self.flat_output:
            x = torch.squeeze(x, dim=2)
        return x         
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay_params = set()
        # keywords = {'cls_token'}
        keywords = {'pos_embed', 'get_v', 'cls_token'}
        for name, param in self.named_parameters():
            if param.requires_grad:
                for key in keywords:
                    if key in name:
                        no_weight_decay_params.add(name)
                        break
        return no_weight_decay_params


def _init_vit_weights(module: nn.Module, name: str='', head_bias: float=0., jax_impl: bool=False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# def detach_module(module):
#     for param in module.parameters():
#         param.requires_grad = False
        