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

import numpy as np

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
    return tuple([int(x) for x in heightwidth])


class Flatten(nn.Module):
    """
    Reshapes the activations to be of shape (B,-1) where B
    is the batch size
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class NullOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Transpose(nn.Module):
    """
    Transposes the activations to be of the argued dimension order.
    Includes the batch dimension
    """
    def __init__(self, order):
        super().__init__()
        """
        Args:
            order: tuple of ints
                the order should include the batch dimension
        """
        self.order = order

    def forward(self, x):
        return x.permute(self.order)


class Reshape(nn.Module):
    """
    Reshapes the activations to be of the argued shape
    """
    def __init__(self, shape):
        super().__init__()
        """
        Args:
            shape: tuple of ints
                the shape should ignore the batch dimension
        """
        self.shape = shape

    def forward(self, x):
        return x.reshape(len(x),*self.shape)


def conv_block(chan_in: int,
               chan_out: int,
               ksize: int=2,
               stride: int=1,
               padding: int=0,
               lnorm: bool=True,
               shape: tuple=None,
               actv_layer=nn.ReLU,
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
        modules.append( nn.GroupNorm(1,chan_in) )
    modules.append( nn.Conv2d(
        chan_in,
        chan_out,
        ksize,
        stride=stride,
        padding=padding
    ))
    torch.nn.init.xavier_uniform_(modules[-1].weight)
    if drop > 0:
        modules.append( nn.Dropout(drop) )
    modules.append( actv_layer() )
    return nn.Sequential( *modules )


class CNN(nn.Module):
    def __init__(self, inpt_shape=(3,32,32),
                       chans=[12,18,24],
                       ksizes=2,
                       strides=1,
                       paddings=0,
                       lnorm=True,
                       out_dim=64,
                       actv_layer=nn.ReLU,
                       drop=0.,
                       n_outlayers=1,
                       h_size=128,
                       output_type="gapooling",
                       is_base=False,
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
            out_dim: int
            actv_layer: torch module
            drop: float
                the probability to drop a node in the network
            n_outlayers: int
                the number of dense layers following the convolutional
                features
            h_size: int
            is_base: bool
                if true, striding is not manipulated and fc layers
                are skipped
            output_type: str
                a string indicating what type of output should be used.

                options: 
                    'gapooling': global average pooling 
                    None: simply flattens features
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.out_dim = out_dim
        self.h_size = h_size
        self.n_outlayers = n_outlayers
        self.chans = [inpt_shape[0], *chans]
        self.ksizes = ksizes
        if isinstance(ksizes, int):
            self.ksizes = [ksizes for i in range(len(chans))]
        self.strides = strides
        if isinstance(strides, int):
            self.strides = [strides for i in range(len(chans))]
            if self.inpt_shape[1] > 32 and not is_base:
                for i in range(min(len(self.strides), 3)):
                    self.strides[-i] = 2
        self.paddings = paddings
        if isinstance(paddings, int):
            self.paddings = [paddings for i in range(len(chans))]
        self.lnorm = lnorm
        self.actv_layer = actv_layer
        self.drop = drop
        self.output_type = output_type.lower()

        # Conv Layers
        self.shapes = [ inpt_shape[-2:] ]
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
                shape=tuple([int(s) for s in self.shapes[-1]]),
                drop=self.drop
            ))
            self.shapes.append( update_shape(
                self.shapes[-1],
                kernel=self.ksizes[i],
                padding=self.paddings[i],
                stride=self.strides[i]
            ))
        self.features = nn.Sequential( *modules )

        if is_base: return
        # Dense Layers
        modules = []
        if self.output_type == "gapooling":
            modules.append( Reshape((self.chans[-1],-1)) )
            modules.append( AvgOverDim(-1) )
            self.flat_dim = self.chans[-1]
        elif self.output_type == "attention":
            modules.append( Reshape((self.chans[-1], -1)) )
            modules.append( Transpose((0,2,1)) )
            modules.append( AttentionalJoin(
                out_dim=self.chans[-1]), pos_enc=False
            )
        else:
            self.flat_dim = int(self.chans[-1]*math.prod(self.shapes[-1]))
        modules.append( Flatten() )
        in_dim = self.flat_dim
        out_dim = self.h_size
        for i in range(self.n_outlayers):
            if i+1 == self.n_outlayers: out_dim = self.out_dim
            modules.append( nn.LayerNorm(in_dim) )
            modules.append( nn.Linear(in_dim, out_dim) )
            torch.nn.init.kaiming_normal_(
                modules[-1].weight,
                nonlinearity='relu'
            )
            if i+1 < self.n_outlayers:
                modules.append( self.actv_layer() )
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
    def __init__(self, n_cnns=1,
                       agg_fxn="AvgOverDim",
                       share_base=False,
                       **kwargs):
        """
        This model architecture resembles a tree structure in which
        multiple small CNNs feed into a final layer.

        Args:
            n_cnns: int
                the number of individual CNN networks to instantiate.
            agg_fxn: str
                the function used to combine the leaf cnns
            inpt_shape: tuple of ints
            chans: list of ints
            ksizes: int or list of ints
                if single int, will use as kernel size for all layers.
            strides: int or list of ints
                if single int, will use as stride for all layers.
            paddings: int or list of ints
                if single int, will use as padding for all layers.
            lnorm: bool
            out_dim: int
                the output dimensionality of each CNN and the whole
                TreeCNN
            actv_layer: torch module
            drop: float
                the probability to drop a node in the network
            h_size: int
                hidden dim of dense layers in cnn final layers
        """
        super().__init__()
        self.share_base = share_base
        self.n_cnns = n_cnns
        self.out_dim = kwargs["out_dim"]
        self.base = NullOp()
        if self.share_base:
            kwgs = {
                **kwargs,
                "chans":kwargs["chans"][:2],
                "is_base": True
            }
            cnn = CNN(**kwgs)
            self.base = cnn.features
            kwargs["inpt_shape"] = [kwgs["chans"][-1], *cnn.shapes[-1]]
        self.cnns = nn.ModuleList([])
        for n in range(n_cnns):
            self.cnns.append( CNN(**kwargs) )
        self.agg_fxn_str = agg_fxn
        self.agg_fxn = globals()[agg_fxn](**kwargs)
        self.leaf_idx = None

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            joined_outps: torch FloatTensor (B, D)
            outps: torch FloatTensor (B, D)
        """
        x = self.base(x)
        outps = []
        if self.leaf_idx is not None:
            outps.append( self.cnns[self.leaf_idx](x) )
        elif self.agg_fxn_str == "AvgOverDim":
            p = 1/self.n_cnns
            outpt = self.cnns[0](x)
            for cnn in self.cnns[1:]:
                outpt += cnn(x)
            return p*outpt
        else:
            for cnn in self.cnns:
                outps.append( cnn(x) )
        outps = torch.stack(outps, dim=1)
        self.outps = outps
        return self.agg_fxn( outps )


class AvgOverDim(nn.Module):
    """
    Averages over the specified dimension
    """
    def __init__(self, dim=1, *args, **kwargs):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv_w = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x):
        B, N, C = x.shape
        kv = self.kv_w(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_w(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class AttentionalJoin(nn.Module):
    """
    Uses a CLS token to extract features using attention from a sequence
    of vectors.
    """
    def __init__(self, out_dim, pos_enc=False, cls=True, *args, **kwargs):
        super().__init__()
        self.dim = out_dim
        self.attn = Attention(dim=out_dim, *args, **kwargs)
        self.cls = None
        if cls: self.cls = nn.Parameter(torch.zeros(1,1,out_dim))
        self.pos_enc = pos_enc
        if self.pos_enc:
            self.pos_enc = PositionalEncoding(
                d_model=out_dim, max_len=5000
            )

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B,N,D)
        Returns:
            fx: torch FloatTensor (B,D)
                the attended values
        """
        B,N,D = x.shape
        if self.pos_enc: x = self.pos_enc(x)
        if self.cls not None:
            cls = self.cls.repeat(B,1,1)
            fx,_ = self.attn(cls, x)
            return fx[...,0,:]
        else:
            fx,_ = self.attn(x, x)
            return fx.mean(-2)


class PositionalEncoding(nn.Module):
    """
    Taken from pytorch tutorial. A simple positonal encoding taken from
    vaswani et al.
    """
    def __init__(self, d_model, dropout= 0.1, max_len= 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = self.get_pe(d_model, max_len)
        self.register_buffer('pe', pe)

    def get_pe(d_model, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [..., seq_len, embedding_dim]
        Returns:
            enc: Tensor, shape [..., seq_len, embedding_dim]
        """
        pe = self.pe
        if x.size(-2) > self.pe.shape[-2]:
            pe = self.get_pe(self.d_model, x.size(-2))
        x = x + pe[:,:x.size(-2)]
        return self.dropout(x)


class RecurrentAttention(nn.Module):
    """
    Applies attention over every k entries to extract features and then
    repeats with the outputs of that step (N//k fewer inputs in the 2nd
    step). Repeats until only 1 output is left which is returned.
    """
    def __init__(self, out_dim, seq_len=4, pos_enc=False, cls=True, *args, **kwargs):
        super().__init__()
        self.dim = out_dim
        self.seq_len = seq_len
        self.attn_join = AttentionalJoin(out_dim, pos_enc,cls=cls, **kwargs)
        self.edge = nn.Parameter(torch.zeros(1,1,out_dim))

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B,N,D)
        Returns:
            fx: torch FloatTensor (B,D)
                the attended values
        """
        B,N,D = x.shape
        fx = x
        while fx.numel() > B*D:
            mod = fx.shape[-2] % self.seq_len
            if mod != 0:
                shape = [1 for _ in range(len(fx.shape[:-1]))]
                edge = self.edge.reshape(*shape, self.dim)
                edge = edge.repeat(*fx.shape[:-2],mod,1)
                fx = torch.cat([fx, edge], dim=-2)
            fx = fx.reshape(-1,self.seq_len,D)
            fx = self.attn_join(fx)
            fx = fx.reshape(B,-1,D)
        return fx.reshape(B,D)



