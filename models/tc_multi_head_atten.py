import paddle
import paddle.nn as nn
from typing import Optional, Tuple
import paddle.nn.functional as F
import math





class Linear(nn.Layer):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    # weight: paddle.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight = self.create_parameter((out_features, in_features), **factory_kwargs)
        if bias:
            # self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias = self.create_parameter((out_features,), **factory_kwargs)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.initializer.KaimingUniform()(self.weight)
        if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # init.uniform_(self.bias, -bound, bound)
            nn.initializer.Uniform()(self.bias)

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)
        
        

# ------------------------------------------------------------------------
# class MultiheadAttention(nn.Layer):
#     r"""Allows the model to jointly attend to information
#     from different representation subspaces as described in the paper:
#     `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
#     Multi-Head Attention is defined as:
#     .. math::
#         \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
#     where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
#     ``forward()`` will use a special optimized implementation if all of the following
#     conditions are met:
#     - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
#       restriction will be loosened in the future.)
#     - inputs are batched (3D) with ``batch_first==True``
#     - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
#     - training is disabled (using ``.eval()``)
#     - ``add_bias_kv`` is ``False``
#     - ``add_zero_attn`` is ``False``
#     - ``batch_first`` is ``True`` and the input is batched
#     - ``kdim`` and ``vdim`` are equal to ``embed_dim``
#     - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
#       nor ``attn_mask`` is passed
#     - autocast is disabled
#     If the optimized implementation is in use, a
#     `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
#     ``query``/``key``/``value`` to represent padding more efficiently than using a
#     padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
#     will be returned, and an additional speedup proportional to the fraction of the input
#     that is padding can be expected.
#     Args:
#         embed_dim: Total dimension of the model.
#         num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
#             across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
#         dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
#         bias: If specified, adds bias to input / output projection layers. Default: ``True``.
#         add_bias_kv: If specified, adds bias to the key and value sequences at axis=0. Default: ``False``.
#         add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at axis=1.
#             Default: ``False``.
#         kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
#         vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#     Examples::
#         >>> # xdoctest: +SKIP
#         >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
#         >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
#     """
#     __constants__ = ['batch_first']
#     bias_k: Optional[paddle.Tensor]
#     bias_v: Optional[paddle.Tensor]

#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         # factory_kwargs = {'device': device, 'dtype': dtype}
#         factory_kwargs = {'dtype': dtype}
#         super(MultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         if not self._qkv_same_embed_dim:
#             self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
#             self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
#             self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)

#         if bias:
#             self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self._reset_parameters()

#     def _reset_parameters(self):
        
#         xavier_uniform_ = nn.initializer.XavierUniform()
#         constant_ = nn.initializer.Constant(0)
#         xavier_normal_ = nn.initializer.XavierNormal()
        
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)

#         if self.in_proj_bias is not None:
#             # constant_(self.in_proj_bias, 0.)
#             # constant_(self.out_proj.bias, 0.)
#             constant_(self.in_proj_bias)
#             constant_(self.out_proj.bias)
            
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)

#     def __setstate__(self, state):
#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if '_qkv_same_embed_dim' not in state:
#             state['_qkv_same_embed_dim'] = True

#         super(MultiheadAttention, self).__setstate__(state)

#     def forward(self, query: paddle.Tensor, key: paddle.Tensor, 
#                 value: paddle.Tensor, key_padding_mask: Optional[paddle.Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[paddle.Tensor] = None,
#                 average_attn_weights: bool = True) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
#         r"""
#     Args:
#         query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
#             or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
#             :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
#             Queries are compared against key-value pairs to produce the output.
#             See "Attention Is All You Need" for more details.
#         key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
#             or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
#             :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
#             See "Attention Is All You Need" for more details.
#         value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
#             ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
#             sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
#             See "Attention Is All You Need" for more details.
#         key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
#             to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
#             Binary and byte masks are supported.
#             For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
#             the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
#         need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
#             Default: ``True``.
#         attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
#             :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
#             :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
#             broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
#             Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
#             corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
#             corresponding position is not allowed to attend. For a float mask, the mask values will be added to
#             the attention weight.
#         average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
#             heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
#             effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
#     Outputs:
#         - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
#           :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
#           where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
#           embedding dimension ``embed_dim``.
#         - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
#           returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
#           :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
#           :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
#           head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.
#         .. note::
#             `batch_first` argument is ignored for unbatched inputs.
#         """
#         is_batched = query.ndim == 3
#         if key_padding_mask is not None:
#             _kpm_dtype = key_padding_mask.dtype
#             if _kpm_dtype != paddle.bool and not paddle.is_floating_point(key_padding_mask):
#                 raise AssertionError(
#                     "only bool and floating types of key_padding_mask are supported")
#         why_not_fast_path = ''
#         if not is_batched:
#             why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.ndim}"
#         elif query is not key or key is not value:
#             # When lifting this restriction, don't forget to either
#             # enforce that the dtypes all match or test cases where
#             # they don't!
#             why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
#         elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
#             why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
#         elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
#             # this case will fail anyway, but at least they'll get a useful error message.
#             why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
#         elif self.training:
#             why_not_fast_path = "training is enabled"
#         elif not self.batch_first:
#             why_not_fast_path = "batch_first was not True"
#         elif self.bias_k is not None:
#             why_not_fast_path = "self.bias_k was not None"
#         elif self.bias_v is not None:
#             why_not_fast_path = "self.bias_v was not None"
#         elif self.add_zero_attn:
#             why_not_fast_path = "add_zero_attn was enabled"
#         elif not self._qkv_same_embed_dim:
#             why_not_fast_path = "_qkv_same_embed_dim was not True"
#         elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
#             why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
#                                  is not supported with NestedTensor input"
#         # elif torch.is_autocast_enabled():
#         elif False:
#             why_not_fast_path = "autocast is enabled"

#         if not why_not_fast_path:
#             tensor_args = (
#                 query,
#                 key,
#                 value,
#                 self.in_proj_weight,
#                 self.in_proj_bias,
#                 self.out_proj.weight,
#                 self.out_proj.bias,
#             )
#             # We have to use list comprehensions below because TorchScript does not support
#             # generator expressions.
#             if torch.overrides.has_torch_function(tensor_args):
#                 why_not_fast_path = "some Tensor argument has_torch_function"
#             elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
#                 why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
#             elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
#                 why_not_fast_path = ("grad is enabled and at least one of query or the "
#                                      "input/output projection weights or biases requires_grad")
#             if not why_not_fast_path:
#                 merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

#                 return torch._native_multi_head_attention(
#                     query,
#                     key,
#                     value,
#                     self.embed_dim,
#                     self.num_heads,
#                     self.in_proj_weight,
#                     self.in_proj_bias,
#                     self.out_proj.weight,
#                     self.out_proj.bias,
#                     merged_mask,
#                     need_weights,
#                     average_attn_weights,
#                     mask_type)

#         any_nested = query.is_nested or key.is_nested or value.is_nested
#         assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
#                                 f"The fast path was not hit because {why_not_fast_path}")

#         if self.batch_first and is_batched:
#             # make sure that the transpose op does not affect the "is" property
#             if key is value:
#                 if query is key:
#                     query = key = value = query.transpose(1, 0)
#                 else:
#                     query, key = [x.transpose(1, 0) for x in (query, key)]
#                     value = key
#             else:
#                 query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

#         if not self._qkv_same_embed_dim:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#                 v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
#         else:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, average_attn_weights=average_attn_weights)
#         if self.batch_first and is_batched:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights

#     def merge_masks(self, 
#                     attn_mask: Optional[paddle.Tensor], 
#                     key_padding_mask: Optional[paddle.Tensor],
#                     query: paddle.Tensor) -> Tuple[Optional[paddle.Tensor], Optional[int]]:
#         r"""
#         Determine mask type and combine masks if necessary. If only one mask is provided, that mask
#         and the corresponding mask type will be returned. If both masks are provided, they will be both
#         expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
#         and mask type 2 will be returned
#         Args:
#             attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
#             key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
#             query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
#         Returns:
#             merged_mask: merged mask
#             mask_type: merged mask type (0, 1, or 2)
#         """
#         mask_type: Optional[int] = None
#         merged_mask: Optional[paddle.Tensor] = None
#         if attn_mask is not None:
#             mask_type = 0
#             merged_mask = attn_mask
#         if key_padding_mask is not None:
#             mask_type = 1
#             merged_mask = key_padding_mask
#         if (attn_mask is not None) and (key_padding_mask is not None):
#             # In this branch query can't be a nested tensor, so it has a shape
#             batch_size, seq_len, _ = query.shape
#             mask_type = 2
#             key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len) \
#                                                         .expand(-1, self.num_heads, -1, -1)
#             attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
#             merged_mask = attn_mask_expanded.logical_or(key_padding_mask_expanded)
#         return merged_mask, mask_type