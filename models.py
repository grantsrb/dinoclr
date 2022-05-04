# Copyright (c)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_

def update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or
    deconvolution. Does not operate in place on shape.

    shape: list-like (chan, height, width)
    kernel: int or list-like
        size of the kernel
    padding: list-like or int
    stride: list-like or int
    op: str
        'conv' or 'deconv'
    """
    heightwidth = np.asarray([*shape[-2:]])
    if type(kernel) == type(int()):
        kernel = np.asarray([kernel, kernel])
    else:
        kernel = np.asarray(kernel)
    if type(padding) == type(int()):
        padding = np.asarray([padding,padding])
    else:
        padding = np.asarray(padding)
    if type(stride) == type(int()):
        stride = np.asarray([stride,stride])
    else:
        stride = np.asarray(stride)

    if op == "conv":
        heightwidth = (heightwidth - kernel + 2*padding)/stride + 1
    elif op == "deconv" or op == "conv_transpose":
        heightwidth = (heightwidth - 1)*stride + kernel - 2*padding
    return tuple(heightwidth)


class Flatten(nn.Module):
    """
    Reshapes the activations to be of shape (B,-1) where B
    is the batch size
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def conv_block(chan_in: int,
               chan_out: int,
               ksize: int=2,
               stride: int=1,
               padding: int=0,
               lnorm: bool=True,
               shape: tuple=None,
               actv_layer=nn.GELU,
               drop: float=0):
    """
    Args:
        chan_in: int
        chan_out: int
        ksize: int
        stride: int
        padding: int
        lnorm: bool
        actv_layer: torch module
        drop: float
            dropout probability
    """
    modules = []
    if lnorm:
        modules.append( nn.LayerNorm(shape) )
    modules.append( nn.Conv2d(
        chan_in,
        chan_out,
        ksize,
        stride=stride,
        padding=padding
    ))
    torch.nn.init.kaiming_normal_(modules[-1].weight,nonlinearity='gelu')
    if drop > 0:
        modules.append( nn.Dropout(drop) )
    modules.append( actv_layer )
    return nn.Sequential( *modules )


class CNN(nn.Module):
    def __init__(self, inpt_shape=(3,32,32),
                       chans=[12,18,24],
                       ksizes=2,
                       strides=1,
                       paddings=0,
                       lnorm=True,
                       outp_dim=64,
                       actv_layer=nn.GELU,
                       drop=0.,
                       n_outlayers=1,
                       *args, **kwargs):
        """
        Simple CNN architecture

        Args:
            inpt_shape: tuple of ints (C,H,W)
            chans: list of ints
            ksizes: int or list of ints
                if single int, will use as kernel size for all layers.
            strides: int or list of ints
                if single int, will use as stride for all layers.
            paddings: int or list of ints
                if single int, will use as padding for all layers.
            lnorm: bool
            outp_dim: int
            actv_layer: torch module
            drop: float
                the probability to drop a node in the network
            n_outlayers: int
                the number of dense layers following the convolutional
                features
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.outp_dim = outp_dim
        self.chans = [inpt_shape[0], *chans]
        self.ksizes = ksizes
        if isinstance(ksizes, int):
            self.ksizes = [ksizes for i in range(len(chans))]
        self.strides = strides
        if isinstance(strides, int):
            self.strides = [strides for i in range(len(chans))]
        self.paddings = paddings
        if isinstance(paddings, int):
            self.paddings = [paddings for i in range(len(chans))]
        self.lnorm = lnorm
        self.actv_layer = actv_layer
        self.drop = drop

        # Conv Layers
        self.shapes = [ inpt_shape[1:] ]
        modules = []
        for i in range(len(chans)):
            modules.append( conv_block(
                self.chans[i],
                self.chans[i+1],
                ksize=self.ksizes[i],
                stride=self.strides[i],
                padding=self.paddings[i],
                actv_layer=self.actv_layer,
                lnorm=self.lnorm and i!=0,
                shape=self.shapes[-1],
                drop=self.drop
            ))
            self.shapes.append( update_shape(
                self.shapes[-1],
                kernel=self.ksizes[i],
                padding=self.paddings[i],
                stride=self.strides[i]
            ))
        self.features = nn.Sequential( *modules )

        # Dense Layers
        self.flat_dim = int(self.chans[-1]*math.prod(self.shapes[-1]))
        modules = [ Flatten() ]
        in_dim = self.flat_dim
        out_dim = self.h_size
        for i in range(self.n_outlayers):
            if i+1 == self.n_outlayers: out_dim = self.outp_dim
            modules.append( nn.LayerNorm(in_dim) )
            torch.nn.init.kaiming_normal_(
                modules[-1].weight,
                nonlinearity='gelu'
            )
            modules.append( nn.Linear(in_dim, out_dim) )
            if i+1 < self.n_outlayers:
                modules.append( self.actv_layer )
            in_dim = self.h_size
        self.dense = nn.Sequential( *modules )
        self.net = nn.Sequential(self.features, self.dense)

    def _init_weights(self, m):
        """ Handled during instantiation """
        pass

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B, C, H, W)
        """
        return self.net(x)


class TreeCNN(nn.Module):
    def __init__(self, n_cnns=1, join_fxn="AvgOverDim", **kwargs):
        """
        This model architecture resembles a tree structure in which
        multiple small CNNs feed into a final layer.

        Args:
            n_cnns: int
                the number of individual CNN networks to instantiate.
            inpt_shape: tuple of ints
            chans: list of ints
            ksizes: int or list of ints
                if single int, will use as kernel size for all layers.
            strides: int or list of ints
                if single int, will use as stride for all layers.
            paddings: int or list of ints
                if single int, will use as padding for all layers.
            lnorm: bool
            outp_dim: int
                the output dimensionality of each CNN and the whole
                TreeCNN
            actv_layer: torch module
            drop: float
                the probability to drop a node in the network
        """
        super().__init__()
        self.n_cnns = n_cnns
        self.cnns = nn.ModuleList([])
        for n in range(n_cnns):
            self.cnns.append( CNN(**kwargs) )
        self.join_fxn = globals()[join_fxn](**kwargs)

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            joined_outps: torch FloatTensor (B, D)
            outps: torch FloatTensor (B, D)
        """
        outps = []
        for cnn in self.cnns:
            outps.append( cnn(x) )
        outps = torch.stack(outps, dim=1)
        self.outps = outps
        return self.join_fxn( outps )


class AvgOverDim(nn.Module):
    """
    Averages over the specified dimension
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (..., dim_n, spec_dim, ...)
        Returns:
            avg: torch FloatTensor (..., dim_n, ...)
                the average over the specified dim
        """
        return x.mean(self.dim)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class AttentionalJoin(nn.Module):
    def __init__(self, outp_dim, *args, **kwargs):
        super().__init__()
        self.dim = outp_dim
        self.attn = Attention(dim=outp_dim, *args, **kwargs)
        self.cls = nn.Parameter(torch.zeros(1,1,outp_dim))

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B,N,D)
        Returns:
            fx: torch FloatTensor (B,D)
                the attended values
        """
        cls = self.cls.repeat(len(x), 1, 1)
        x = torch.cat([x,cls], dim=1)
        fx,_ = self.attn(x)
        return fx[:,-1]


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
