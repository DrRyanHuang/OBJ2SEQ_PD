# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import paddle
from paddle import nn
from .attention_modules import DeformableEncoderLayer, _get_clones
from .ppdet_deformable_transformer import DeformableTransformerEncoder, DeformableTransformerEncoderLayer

class TransformerEncoder(nn.Layer):
    def __init__(self, args):
        super().__init__()
        encoder_layer = DeformableEncoderLayer(args.ENCODER_LAYER)
        
        # encoder_layer_params = dict(
        #     d_model = args.ENCODER_LAYER.hidden_dim,
        #     n_head  = args.ENCODER_LAYER.nheads,
        #     dim_feedforward = args.ENCODER_LAYER.dim_feedforward,
        #     dropout = args.ENCODER_LAYER.dropout,
        #     activation = args.ENCODER_LAYER.activation,
        #     n_levels = args.ENCODER_LAYER.n_levels,
        #     n_points = args.ENCODER_LAYER.n_points,
        # )
        # encoder_layer = DeformableTransformerEncoderLayer(encoder_layer_params)
        
        self.encoder_layers =  _get_clones(encoder_layer, args.enc_layers) # `_get_clones` 用来给层多复制几倍 
        
    
    def forward(self, tgt, *args, **kwargs):
        # tgt: bs, h, w, c || bs, l, c
        for layer in self.encoder_layers:
            tgt = layer(tgt, *args, **kwargs)
        return tgt


def build_encoder(args):
    return TransformerEncoder(args)