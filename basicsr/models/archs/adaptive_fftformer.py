import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.models.archs.fftformer_arch import fftformer
from basicsr.models.archs.adapter import SpatialPriorModule
from basicsr.models.archs.adapter import InteractionBlock
from timm.models.layers import trunc_normal_
import math
# TODO: find out which version of the MSDeformAttn it's using




class Adaptive_FFTFormer(nn.Module):
    def __init__(self, 
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 pretrain_size=224, 
                 num_heads=12, 
                 conv_inplane=64, 
                 n_points=4,
                 deform_num_heads=6, 
                 init_values=0., 
                 interaction_indexes=None, 
                 with_cffn=True,
                 cffn_ratio=0.25, 
                 deform_ratio=1.0, 
                 add_vit_feature=True, 
                 pretrained=None,
                 use_extra_extractor=True, 
                 with_cp=False, 
                 *args, 
                 **kwargs):
        super(Adaptive_FFTFormer, self).__init__()

        self.pretrained_model = fftformer(inp_channels,
                 out_channels,
                 dim,
                 num_blocks,
                 num_refinement_blocks,
                 ffn_expansion_factor,
                 bias=False)
        
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=dim, with_cp=False)
        
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(dim)
        self.norm2 = nn.SyncBatchNorm(dim)
        self.norm3 = nn.SyncBatchNorm(dim)
        self.norm4 = nn.SyncBatchNorm(dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def load_pretrained_weights(self, weights):
        # TODO: load pretrained weights for fftformer
        return NotImplementedError()
    
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, x):
        return NotImplementedError()

    
