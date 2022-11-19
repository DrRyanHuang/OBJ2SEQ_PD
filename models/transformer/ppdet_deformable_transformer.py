# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.

# https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr


from .ppdet_transformer_position_encoding import PositionEmbedding
from .ppdet_transformer_utils import _get_clones, deformable_attention_core_func
from .ppdet_initializer import linear_init_, constant_, xavier_uniform_, normal_

# __all__ = ['DeformableTransformer']





# from ..layers import MultiHeadAttention
def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    return nn.layer.transformer._convert_attention_mask(attn_mask, dtype)



class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False)
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                constant_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                x=tensor,
                weight=self.in_proj_weight[:, index * self.embed_dim:(index + 1)
                                           * self.embed_dim],
                bias=self.in_proj_bias[index * self.embed_dim:(index + 1) *
                                       self.embed_dim]
                if self.in_proj_bias is not None else None)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape(
            [0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


















class MSDeformableAttention(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1])
        scaling = paddle.arange(
            1, self.num_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights, -1).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        offset_normalizer = value_spatial_shapes.flip([1]).reshape(
            [1, 1, 1, self.num_levels, 1, 2])
        sampling_locations = reference_points.reshape([
            bs, Len_q, 1, self.num_levels, 1, 2
        ]) + sampling_offsets / offset_normalizer

        output = deformable_attention_core_func(
            value, value_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class DeformableTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                               n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                reference_points,
                spatial_shapes,
                src_mask=None,
                pos_embed=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos_embed), reference_points, src,
            spatial_shapes, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src



class DeformableTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super(DeformableTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, H - 0.5, H),
                paddle.linspace(0.5, W - 0.5, W))
            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] *
                                                    H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] *
                                                    W)
            reference_points.append(paddle.stack((ref_x, ref_y), axis=-1))
        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(self,
                src,
                spatial_shapes,
                src_mask=None,
                pos_embed=None,
                valid_ratios=None):
        output = src
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [src.shape[0], spatial_shapes.shape[0], 2])
        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios)
        for layer in self.layers:
            output = layer(output, reference_points, spatial_shapes, src_mask,
                           pos_embed)

        return output





class DeformableTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2 = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Layer):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super(DeformableTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_mask=None,
                query_pos_embed=None):
        output = tgt
        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, reference_points, memory,
                           memory_spatial_shapes, memory_mask, query_pos_embed)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)



class DeformableTransformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 num_queries=300,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=[512, 1024, 2048],
                 num_feature_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 lr_mult=0.1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_num_channels) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_encoder_points, weight_attr, bias_attr)
        self.encoder = DeformableTransformerEncoder(encoder_layer,
                                                    num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_decoder_points, weight_attr, bias_attr)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.reference_points = nn.Linear(
            hidden_dim,
            2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.input_proj = nn.LayerList()
        for in_channels in backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)))
        in_channels = backbone_num_channels[-1]
        for _ in range(num_feature_levels - len(backbone_num_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)))
            in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=-0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        normal_(self.level_embed.weight)
        normal_(self.tgt_embed.weight)
        normal_(self.query_pos_embed.weight)
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_num_channels': [i.channels for i in input_shape], }

    def _get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_ratio_h = paddle.sum(mask[:, :, 0], 1) / H
        valid_ratio_w = paddle.sum(mask[:, 0, :], 1) / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src_feats, src_mask=None):
        srcs = []
        for i in range(len(src_feats)):
            srcs.append(self.input_proj[i](src_feats[i]))
        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for i in range(len_srcs, self.num_feature_levels):
                if i == len_srcs:
                    srcs.append(self.input_proj[i](src_feats[-1]))
                else:
                    srcs.append(self.input_proj[i](srcs[-1]))
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shapes.append([h, w])
            src = src.flatten(2).transpose([0, 2, 1])
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(src_mask.unsqueeze(0), size=(h, w))[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(self._get_valid_ratio(mask))
            pos_embed = self.position_embedding(mask).flatten(2).transpose(
                [0, 2, 1])
            lvl_pos_embed = pos_embed + self.level_embed.weight[level].reshape(
                [1, 1, -1])
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        # [l, 2]
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int64')
        # [b, l, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, mask_flatten,
                              lvl_pos_embed_flatten, valid_ratios)

        # prepare input for decoder
        bs, _, c = memory.shape
        query_embed = self.query_pos_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        tgt = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        reference_points = F.sigmoid(self.reference_points(query_embed))
        reference_points_input = reference_points.unsqueeze(
            2) * valid_ratios.unsqueeze(1)

        # decoder
        hs = self.decoder(tgt, reference_points_input, memory, spatial_shapes,
                          mask_flatten, query_embed)

        return (hs, memory, reference_points)