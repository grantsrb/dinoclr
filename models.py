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
import torchvision.models

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
    Include the batch dimension in your order argument.
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
    Reshapes the activations to be of the argued shape. Ignores the
    batch dimension unless specified in init.
    """
    def __init__(self, shape, ignore_batch_dim=True):
        super().__init__()
        """
        Args:
            shape: tuple of ints
                the shape should ignore the batch dimension
        """
        self.shape = shape
        self.ignore_batch_dim = ignore_batch_dim

    def forward(self, x):
        if self.ignore_batch_dim: return x.reshape(len(x),*self.shape)
        return x.reshape(self.shape)


class ResBlock(nn.Module):
    def __init__(self, chan_in: int,
               chan_out: int,
               ksize: int=2,
               stride: int=1,
               padding: int=0,
               lnorm: bool=True,
               actv_layer=nn.ReLU,
               drop: float=0,
               groups: int=1):
        """
        chan_in: int
        chan_out: int
        ksize: int
        stride: int
        padding: int
        lnorm: bool
        actv_layer: torch module
        drop: float
            dropout probability
        groups: int
            the number of independent convolutions within the layer
        """
        super().__init__()
        self.proj = nn.Conv2d(
          chan_in,chan_out,ksize,stride=stride,padding=padding,groups=groups
        )
        torch.nn.init.xavier_uniform_(self.proj.weight)
        if ksize%2 == 0: padding = (1,0,1,0)
        self.conv1 = conv_block(
            chan_out,
            chan_out,
            ksize=ksize,
            stride=1,
            padding=padding,
            lnorm=lnorm,
            actv_layer=actv_layer,
            drop=drop,
            groups=groups,
            residual=False
        )
        self.conv2 = conv_block(
            chan_out,
            chan_out,
            ksize=ksize,
            stride=1,
            padding=padding,
            lnorm=lnorm,
            actv_layer=actv_layer,
            drop=drop,
            groups=groups,
            residual=False
        )
        self.norm = nn.GroupNorm(groups,chan_out)

    def forward(self, x):
        fx = self.proj(x)
        return self.norm(fx + self.conv2(self.conv1(fx)))

def conv_block(chan_in: int,
               chan_out: int,
               ksize: int=2,
               stride: int=1,
               padding: int or tuple=0,
               lnorm: bool=True,
               actv_layer=nn.ReLU,
               drop: float=0,
               groups: int=1,
               residual=False):
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
        groups: int
            the number of independent convolutions within the layer
        residual: bool
            if true, uses residual style connections
    """
    if residual:
        return ResBlock(
            chan_in=chan_in,
            chan_out=chan_out,
            ksize=ksize,
            stride=stride,
            padding=padding,
            lnorm=lnorm,
            actv_layer=actv_layer,
            drop=drop,
            groups=groups
        )
    modules = []
    if lnorm:
        modules.append( nn.GroupNorm(groups,chan_in) )
    if type(padding) != type(int(1)) and len(padding) > 2:
        modules.append(nn.ConstantPad2d(padding, 0))
        padding = 0
    modules.append( nn.Conv2d(
        chan_in,
        chan_out,
        ksize,
        stride=stride,
        padding=padding,
        groups=groups
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
                       agg_dim=64,
                       actv_layer=nn.ReLU,
                       drop=0.,
                       n_outlayers=1,
                       h_size=128,
                       output_type="gapooling",
                       is_base=False,
                       cls=True,
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
            agg_dim: int
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
                    "attention": attentional join
                    "alt_attn": alternating attention
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.agg_dim = agg_dim
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
            in_dim = self.flat_dim
        elif self.output_type == "attention":
            modules.append( Reshape((self.chans[-1], -1)) )
            modules.append( Transpose((0,2,1)) )
            modules.append( AttentionalJoin(
                agg_dim=self.chans[-1]), pos_enc=False
            )
            in_dim = self.chans[-1]
        elif self.output_type == "alt_attn":
            modules.append( Transpose((0,2,3,1)) )
            modules.append( AlternatingAttention(
                agg_dim=self.chans[-1], seq_len=4, cls=cls
            ))
            in_dim = self.chans[-1]
        else:
            self.flat_dim = int(self.chans[-1]*math.prod(self.shapes[-1]))
            in_dim = self.flat_dim
        modules.append( Flatten() )
        agg_dim = self.h_size
        for i in range(self.n_outlayers):
            if i+1 == self.n_outlayers: agg_dim = self.agg_dim
            modules.append( nn.LayerNorm(in_dim) )
            modules.append( nn.Linear(in_dim, agg_dim) )
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


class GroupedCNN(nn.Module):
    def __init__(self, inpt_shape=(3,32,32),
                       chans=[12,18,24],
                       ksizes=2,
                       strides=1,
                       paddings=0,
                       lnorm=True,
                       agg_dim=64,
                       actv_layer=nn.ReLU,
                       drop=0.,
                       n_outlayers=1,
                       h_size=128,
                       output_type="gapooling",
                       is_base=False,
                       cls=True,
                       groups=1,
                       residual_convs=False,
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
            agg_dim: int
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
            groups: int
                the number of independent cnns to use
            output_type: str
                a string indicating what type of output should be used.

                options: 
                    'gapooling': global average pooling 
                    None: simply flattens features
                    "attention": attentional join
                    "alt_attn": alternating attention
            residual_convs: bool
                resnet style convolutions
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.agg_dim = agg_dim
        self.h_size = h_size
        self.n_outlayers = n_outlayers
        self.chans = [inpt_shape[0], *[c*groups for c in chans]]
        self.ksizes = ksizes
        if isinstance(ksizes, int):
            self.ksizes = [ksizes for i in range(len(chans))]
        self.strides = strides
        if isinstance(strides, int):
            self.strides = [strides for i in range(len(chans))]
            if self.inpt_shape[1] > 32 and not is_base:
                for i in range(min(len(self.strides), 3)):
                    self.strides[-i] = 2
                print("changing striedes to accomodate larger image")
                print("strides:", self.strides)
        self.paddings = paddings
        if isinstance(paddings, int):
            self.paddings = [paddings for i in range(len(chans))]
        self.lnorm = lnorm
        self.actv_layer = actv_layer
        self.drop = drop
        self.groups = groups
        self.output_type = output_type.lower()
        self.residual_convs = residual_convs

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
                drop=self.drop,
                groups=self.groups if i > 0 else 1,
                residual=i>0 and self.residual_convs
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
        n_chans = self.chans[-1]//self.groups
        if self.output_type == "gapooling":
            if self.groups>1:
                modules.append( Reshape((self.groups,n_chans,-1)) )
            else:
                modules.append( Reshape((n_chans,-1)) )
            modules.append( AvgOverDim(-1) )
            self.flat_dim = n_chans*self.groups
            in_dim = self.flat_dim
        elif self.output_type == "attention":
            if self.groups>1:
                modules.append( Reshape((self.groups,n_chans,-1)) )
                modules.append( Transpose((0,1,3,2)) )
                in_dim = self.groups*n_chans
            else:
                modules.append( Reshape((self.chans[-1], -1)) )
                modules.append( Transpose((0,2,1)) )
                in_dim = self.chans[-1]
            modules.append( AttentionalJoin(
                agg_dim=n_chans), pos_enc=False
            )
        elif self.output_type == "alt_attn":
            if self.groups>1:
                H,W = self.shapes[-1]
                modules.append( Reshape((self.groups,n_chans,H,W)) )
                modules.append( Transpose((0,1,3,4,2)) )
                in_dim = self.groups*n_chans
                raise NotImplemented
            else:
                modules.append( Transpose((0,2,3,1)) )
                in_dim = self.chans[-1]
            modules.append( AlternatingAttention(
                agg_dim=self.chans[-1], seq_len=4, cls=cls
            ))
        else:
            self.flat_dim =int(self.chans[-1]*math.prod(self.shapes[-1]))
            in_dim = self.flat_dim
        modules.append( Reshape((-1,1,1)) )
        agg_dim = self.h_size*self.groups
        for i in range(self.n_outlayers):
            if i+1 == self.n_outlayers: agg_dim=self.agg_dim*self.groups
            modules.append( nn.GroupNorm(self.groups, in_dim) )
            modules.append(
                nn.Conv2d(in_dim, agg_dim, 1, groups=self.groups)
            )
            torch.nn.init.xavier_uniform_( modules[-1].weight )
            if i+1 < self.n_outlayers:
                modules.append( self.actv_layer() )
            in_dim = self.h_size*self.groups
        modules.append( Reshape((self.groups,-1)) )
        self.dense = nn.Sequential( *modules )
        self.net = nn.Sequential(self.features, self.dense)

    def _init_weights(self, m):
        """ Handled during instantiation """
        pass

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            fx: torch FloatTensor (B, G, D)
        """
        fx = self.net(x)
        return fx


class GroupedTreeCNN(nn.Module):
    def __init__(self, n_cnns=1,
                       agg_fxn="AvgOverDim",
                       share_base=False,
                       proj_agg=False,
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
            agg_dim: int
                the output dimensionality of each CNN and the whole
                TreeCNN
            actv_layer: torch module
            drop: float
                the probability to drop a node in the network
            h_size: int
                hidden dim of dense layers in cnn final layers
            output_type: str
                method for consolidating and outputting cnn activations
            residual_convs: bool
                if true, each convolution is a residual style convolution
            proj_agg: bool
                if true, the aggregated output is processed by a proj
                layer
        """
        super().__init__()
        if "agg_dim" not in kwargs: kwargs["agg_dim"]=kwargs["out_dim"]
        self.share_base = share_base
        self.n_cnns = n_cnns
        self.agg_dim = kwargs["agg_dim"]
        self.base = NullOp()
        if self.share_base:
            kwgs = {
                **kwargs,
                "groups": 1,
                "chans": kwargs["chans"][:2],
                "is_base": True
            }
            cnn = GroupedCNN(**kwgs)
            self.base = cnn.features
            kwargs["inpt_shape"] = [kwgs["chans"][-1], *cnn.shapes[-1]]
            kwargs["chans"] = kwargs["chans"][2:]
        kwargs["groups"] = self.n_cnns
        self.cnn = GroupedCNN( **kwargs )
        self.agg_fxn_str = agg_fxn
        kwargs["n_cnns"] = self.n_cnns
        self.agg_fxn = globals()[agg_fxn](**kwargs)
        self.leaf_idx = None
        self.proj = NullOp()
        if proj_agg:
            self.proj = nn.Linear(self.agg_dim, self.agg_dim)

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            joined_outps: torch FloatTensor (B, D)
            outps: torch FloatTensor (B, D)
        """
        x = self.base(x)
        if self.leaf_idx is not None:
            fx1 = self.cnn(x)
            x_copy = x.clone().data.detach()
            with torch.no_grad():
                fx2 = self.cnn(x_copy)
            mask = torch.zeros(self.n_cnns,1).to(fx1.get_device())
            mask[self.leaf_idx] = 1
            outps = fx1*mask + fx2*(1-mask)
        else:
            outps = self.cnn(x) # (B, G, D)
        agg = self.agg_fxn( outps )
        return self.proj(agg)


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
            agg_dim: int
                the output dimensionality of each CNN and the whole
                TreeCNN
            actv_layer: torch module
            drop: float
                the probability to drop a node in the network
            h_size: int
                hidden dim of dense layers in cnn final layers
            output_type: str
                method for consolidating and outputting cnn activations
        """
        super().__init__()
        if "agg_dim" not in kwargs: kwargs["agg_dim"]=kwargs["out_dim"]
        self.share_base = share_base
        self.n_cnns = n_cnns
        self.agg_dim = kwargs["agg_dim"]
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
            kwargs["chans"] = kwargs["chans"][2:]
        self.cnns = nn.ModuleList([])
        for n in range(n_cnns):
            self.cnns.append( CNN(**kwargs) )
        self.agg_fxn_str = agg_fxn
        kwargs["n_cnns"] = self.n_cnns
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
            x_copy = x.clone().data.detach()
            for i,cnn in enumerate(self.cnns):
                if i == self.leaf_idx: outps.append( cnn(x) )
                else: outps.append( cnn(x_copy) )
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
        B, M, C = q.shape
        B, N, C = x.shape
        kv = self.kv_w(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_w(q).reshape(B, M, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, M, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class AttentionalJoin(nn.Module):
    """
    Uses a CLS token to extract features using attention from a sequence
    of vectors.
    """
    def __init__(self, agg_dim, pos_enc=False, cls=True, *args, **kwargs):
        super().__init__()
        self.dim = agg_dim
        self.attn = Attention(dim=agg_dim, *args, **kwargs)
        self.cls = None
        if cls: self.cls = nn.Parameter(torch.zeros(1,1,agg_dim))
        self.pos_enc = pos_enc
        if self.pos_enc:
            self.pos_enc = PositionalEncoding( d_model=agg_dim, max_len=5000 )

    # TODO: needs testing
    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (...,N,D)
        Returns:
            fx: torch FloatTensor (...,D)
                the attended values
        """
        og_shape = x.shape
        x = x.reshape(-1, *og_shape[-2:])
        B,N,D = x.shape
        if self.pos_enc: x = self.pos_enc(x)
        if self.cls is not None:
            cls = self.cls.repeat(B,1,1)
            fx,_ = self.attn(cls, x)
            fx = fx[...,0,:]
            if len(og_shape) > 3:
                fx = fx.reshape(*og_shape[:-2], fx.shape[-1])
            return fx
        else:
            fx,_ = self.attn(x, x)
            fx = fx.mean(-2)
            if len(og_shape) > 3:
                fx = fx.reshape(*og_shape[:-2], fx.shape[-1])
            return fx


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

    def get_pe(self, d_model, max_len):
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
    def __init__(self, agg_dim, seq_len=4, pos_enc=False, cls=True, *args, **kwargs):
        super().__init__()
        self.dim = agg_dim
        self.seq_len = seq_len
        self.attn_join = AttentionalJoin(agg_dim, pos_enc,cls=cls, **kwargs)
        self.edge = nn.Parameter(torch.zeros(1,1,agg_dim))

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B,N,D)
        Returns:
            fx: torch FloatTensor (B,D)
                the attended values
        """
        og_shape = x.shape
        if len(og_shape) > 3: x = x.reshape(-1,*og_shape[-2:])
        B,N,D = x.shape
        fx = x
        while fx.numel() > B*D:
            mod = fx.shape[-2] % self.seq_len
            if mod != 0:
                shape = [1 for _ in range(len(fx.shape[:-1]))]
                edge = self.edge.reshape(*shape, self.dim)
                edge = edge.repeat(*fx.shape[:-2],self.seq_len-mod,1)
                fx = torch.cat([fx, edge], dim=-2)
            fx = fx.reshape(-1,self.seq_len,D)
            fx = self.attn_join(fx)
            fx = fx.reshape(B,-1,D)
        fx = fx.reshape(B,D)
        if len(og_shape) > 3: fx = fx.reshape(*og_shape[:-2], D)
        return fx


class AlternatingAttention(nn.Module):
    """
    Alternates applying attention along rows and then along columns.
    """
    def __init__(self, agg_dim,
                       seq_len=4,
                       cls=True,
                       *args, **kwargs):
        """
        agg_dim: int
            the dimensionality of the inputs and attention module
        seq_len: int
            the number of elements to perform attention over
        cls: bool
            if true, each application of attention will yield a single
            output token.
        """
        super().__init__()
        self.dim = agg_dim
        self.seq_len = seq_len
        self.attn_join = AttentionalJoin(agg_dim, pos_enc=True, cls=cls, **kwargs)

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B,H,W,D)
        Returns:
            fx: torch FloatTensor (B,D) or (B,H,W,D)
                the attended values. will maintain
        """
        B,H,W,D = x.shape
        fx = x
        while fx.numel() > B*D:
            mod = fx.shape[-2] % self.seq_len
            if mod != 0:
                pad_left = (self.seq_len-mod)//2
                pad_right = pad_left + int((pad_left*2) != (self.seq_len-mod))
                # Pads fx along W dim
                fx = torch.nn.functional.pad(fx, (0,0,pad_left,pad_right))
            H = fx.shape[-3]
            W = fx.shape[-2]//self.seq_len
            fx = fx.reshape(-1,self.seq_len,D)
            fx = self.attn_join(fx)
            fx = fx.reshape(B,H,W,D).permute(0,2,1,3)
        return fx.reshape(B,D)


class DenseJoin(nn.Module):
    def __init__(self, n_cnns,
                       agg_dim,
                       h_size=256,
                       n_layers=3,
                       lnorm=True,
                       actv_layer=nn.ReLU,
                       *args, **kwargs):
        """
        agg_dim: int
            a bit of a misnomer. this is actually the input dim
        """
        super().__init__()
        self.n_cnns = n_cnns
        self.agg_dim = agg_dim
        self.h_size = h_size
        self.lnorm = lnorm
        self.n_layers = n_layers
        self.actv_layer = actv_layer

        self.inpt_size = self.n_cnns*self.agg_dim
        modules = []
        inpt_size = self.inpt_size
        output_size = self.h_size
        for i in range(self.n_layers):
            if i == (self.n_layers-1): output_size = self.agg_dim
            if self.lnorm: modules.append(nn.LayerNorm(inpt_size))
            modules.append(nn.Linear(inpt_size, output_size))
            modules.append(self.actv_layer())
            inpt_size = self.h_size
        self.dense = nn.Sequential( *modules )

    def forward(self, x):
        """
            x: torch FloatTensor (B, N_CNNS, D)
        Returns:
            fx: torch FloatTensor (B, H)
                the attended values. will maintain
        """
        x = x.reshape(len(x), -1)
        return self.dense(x)


def get_mlp(in_size, h_size, out_size, n_layers=2, lnorm=True):
    """
    Args:
        in_size: int
        h_size: int
        out_size: int
        n_layers: int
        lnorm: bool
            determines if uses layer norm
    """
    modules = [Flatten()]
    if lnorm:
        modules.append( nn.LayerNorm(in_size) )
    modules.append(nn.Linear(in_size, h_size) )
    for i in range(n_layers):
        modules.append(nn.ReLU())
        if lnorm: modules.append( nn.LayerNorm(h_size) )
        out = h_size if i+1 < n_layers else out_size
        modules.append( nn.Linear(h_size, out) )
    return nn.Sequential(*modules)


class ResNet50(nn.Module):
    def __init__(self, inpt_shape,
                       agg_dim=64,
                       h_size=128,
                       n_outlayers=1,
                       lnorm=True,
                       agg_fxn="AvgOverDim",
                       *args, **kwargs):
        """
        This model architecture resembles a tree structure in which
        multiple small CNNs feed into a final layer.

        Args:
            agg_fxn: str
                the function used to combine the leaf cnns
            inpt_shape: tuple of ints
            agg_dim: int
            h_size: int
                hidden dim of dense layers in cnn final layers
            output_type: str
                method for consolidating and outputting cnn activations
            n_outlayers: int
                the number of dense layers following the convolutional
                features
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.h_size = h_size
        self.lnorm = lnorm
        self.n_outlayers = n_outlayers
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        self.agg_dim = agg_dim
        self.agg_fxn_str = agg_fxn
        self.agg_fxn = globals()[agg_fxn](**kwargs)

        # Make MLP
        self.dense = get_mlp(
            in_size=512,
            h_size=self.h_size,
            n_layers=self.n_outlayers,
            out_size=self.agg_dim,
            lnorm=self.lnorm
        )
        self.net = nn.Sequential(
            self.features,
            Reshape((512,-1)),
            Transpose((0,2,1)),
            self.agg_fxn,
            Flatten(),
            self.dense
        )
        temp = torch.zeros(1, *self.inpt_shape)
        temp = self.net(temp)
        print("temp agg", temp.shape)

    def forward(self, x):
        """
        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            outps: torch FloatTensor (B, D)
        """
        return self.net(x)

