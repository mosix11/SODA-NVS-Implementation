import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import sys
from . import nn_utils



def vgg_block_lazy(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def NiN_block_lazy(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())

def NiN_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

class Inception(nn.Module):
    # in_c is the number of input channels to the Inception module
    # c1--c4 are the number of output channels for each branch
    def __init__(self, in_c, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.b2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.b3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.Conv2d(in_c, c4, kernel_size=1)


    def forward(self, X):
        b1 = F.relu(self.b1_1(X))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(X))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(X))))
        b4 = F.relu(self.b4_2(self.b4_1(X)))
        return torch.cat((b1, b2, b3, b4), dim=1)


class InceptionLazy(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
    
    
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1_conv=False, strides=1) -> None:
        super().__init__()

        if strides != 1 and use_1x1_conv == False:
            raise ValueError("""
                using a stride bigger than 1 results in the shape mismatch between output of convolution layers and input 
                while adding them. use use_1x1_conv=True to transform the input shape and match it with the output of the 
                convolution layers
                """)
        if in_channels != out_channels and use_1x1_conv == False:
            raise Exception("""
                If the number of input channels and output channels are different you have to use the 1x1 convolution layer on the
                inputs to match their shape (channel wise in this case) with the output of residual block.
                """)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNextBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, in_channels, out_channels, groups=32, bottleneck_multiplier=1, use_1x1_conv=False, strides=1):
        super().__init__()

        if strides != 1 and use_1x1_conv == False:
            raise Exception("""
                using a stride bigger than 1 results in the shape mismatch between output of convolution layers and input 
                while adding them. use use_1x1_conv=True to transform the input shape and match it with the output of the 
                convolution layers
                """)
        
        if in_channels != out_channels and use_1x1_conv == False:
            raise Exception("""
                If the number of input channels and output channels are different you have to use the 1x1 convolution layer on the
                inputs to match their shape (channel wise in this case) with the output of residual block.
                """)
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise Exception("""
                Number of input channels and output channels must be divisible by the number of groups.
                """)
        bot_channels = int(round(out_channels * bottleneck_multiplier))
        self.conv1 = nn.Conv2d(in_channels, bot_channels, kernel_size=1, stride=1)
    
        self.conv2 = nn.Conv2d(bot_channels, bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.Conv2d(bot_channels, out_channels, kernel_size=1, stride=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if use_1x1_conv:
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.BatchNorm2d(out_channels)
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)
    

# Squeeze and Excitation Block from SE NET
class SEBlock(nn.Module):

    def __init__(self, C, r=16) -> None:
        super().__init__()
        
        self.C = C
        self.r = r
        self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, C//r)
        self.fc2 = nn.Linear(C//r, r)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, X):
        # X.shape = [N, C, H, W] 
        Y = self.globpool(X)
        Y = torch.flatten(Y, 1) 
        Y = self.relu(self.fc1(Y))
        Y = self.sigmoid(self.fc2(Y))
        # Y.shape = [N, C, 1, 1] 
        Y = Y[:, :, None, None]
        return X * Y

# TODO add SEBlock to the residual block
# class ResidualBlockWithSEBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, use_1x1_conv=False, strides=1) -> None:
#         super().__init__()

#         if strides != 1 and use_1x1_conv == False:
#             raise ValueError("""
#                 using a stride bigger than 1 results in the shape mismatch between output of convolution layers and input 
#                 while adding them. use use_1x1_conv=True to transform the input shape and match it with the output of the 
#                 convolution layers
#                 """)
#         if in_channels != out_channels and use_1x1_conv == False:
#             raise Exception("""
#                 If the number of input channels and output channels are different you have to use the 1x1 convolution layer on the
#                 inputs to match their shape (channel wise in this case) with the output of residual block.
#                 """)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         if use_1x1_conv:
#             self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None

#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         Y += X
#         return F.relu(Y)

class DenseBlock(nn.Module):

    def __init__(self, in_channels, num_convs, num_channels) -> None:
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(self.conv_block(in_channels, num_channels))
            in_channels += num_channels
        self.net = nn.Sequential(*layer)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)

class S2SEncoder(nn.Module):
    """The base encoder interface for the Sequence 2 Sequence encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError

class S2SDecoder(nn.Module):
    """The base decoder interface for the Sequence 2 Sequence encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class S2SEncoderDecoder(nn.Module):
    """The base class for the Sequence 2 Sequence encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
    
class AttentionS2SDecoder(S2SDecoder):  #@save
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d, device=queries.device))
        self.attention_weights = nn_utils.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    

class AdditiveAttention(nn.Module): 
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        
        features = queries.unsqueeze(2) + keys.unsqueeze(1)    # These two steps calculate the alignment score vector
        features = torch.tanh(features)                        # or in other words alpha(q, k); 
                                                               # queries are the s_t-1 the hidden state of the decoder and 
                                                               # the keys are the hidden states of the encoder
                                                               
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = nn_utils.masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)
    

class DiffDimDotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, key_dim, dropout):
        super().__init__()
        self.W_q = nn.LazyLinear(key_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        queries = self.W_q(queries)
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d, device=queries.device))
        self.attention_weights = nn_utils.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    # def predict_step(self, batch, device, num_steps,
    #                 save_attention_weights=False):
    #     batch = [a.to(device) for a in batch]
    #     src, tgt, src_valid_len, _ = batch
    #     enc_all_outputs = self.encoder(src, src_valid_len)
    #     dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    #     outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []
    #     for _ in range(num_steps):
    #         Y, dec_state = self.decoder(outputs[-1], dec_state)
    #         outputs.append(Y.argmax(2))
    #         # Save attention weights (to be covered later)
    #         if save_attention_weights:
    #             attention_weights.append(self.decoder.attention_weights)
    #     return torch.cat(outputs[1:], 1), attention_weights

    
class SimpleMultiHeadAttention(nn.Module):
    """Multi-head attention."""
    """This class simply assumes that p_q = p_k = p_v = p_o\h
       where 
       p_q is the projection dimension of transformer head for query
       p_k is the projection dimension of transformer head for key
       p_v is the projection dimension of transformer head for the value
       p_o is the output dimension of the combined heads
       h is the number of the transformer heads
       
       num_hiddens is p_o
    """
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
    
    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    
    
class PositionalEncoding(nn.Module):  
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
    
class PositionWiseFFN(nn.Module): 
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
    
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))