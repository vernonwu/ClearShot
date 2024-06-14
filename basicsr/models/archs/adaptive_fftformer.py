import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.models.archs.fftformer_arch import *
from basicsr.models.archs.adapter import SpatialPriorModule
from basicsr.models.archs.adapter import InteractionBlock
from functools import partial
from basicsr.models.archs.adapter import deform_inputs
from timm.models.layers import trunc_normal_
import math
from basicsr.models.archs.ms_deform_attn import MSDeformAttn

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Adaptive_FFTFormer(fftformer):
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
                 conv_inplane=48,
                 patch_size = 8,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=[[0, 0], [1, 2],[3, 4]],
                 with_cffn= False,
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
        self.patch_size = patch_size
        self.drop_path_rate = 0.2
        self.interaction_indexes = interaction_indexes

        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=dim)
        self.level_embed = nn.Parameter(torch.zeros(3, dim))

        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=dim*2**i, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             with_cffn=with_cffn, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio, extra_extractor= False,
                             patch_merge = (False if i == 0 else True),
                             with_cp=with_cp, with_extractor = (False if i == 2 else True), patch_size = patch_size)
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(dim)
        self.norm2 = nn.SyncBatchNorm(dim*2)
        self.norm3 = nn.SyncBatchNorm(dim*4)
        self.adapter_patch_embed = OverlapPatchEmbed(self.pretrained_model.inp_channels, dim)

        self.blocks = nn.Sequential(*
            [self.encoder_level1,
            self.down1_2, self.encoder_level2,
            self.down2_3, self.encoder_level3]
        )

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        if pretrained is not None:
            self.load_pretrained_weights(pretrained)

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

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
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def load_pretrained_weights(self, weights):
        pretrained_state_dict = torch.load(weights)
        model_state_dict = self.pretrained_model.state_dict()
        for name, param in pretrained_state_dict.items():
            model_state_dict[name].copy_(param)
            model_state_dict[name].requires_grad = False
        self.load_state_dict(model_state_dict)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, input_img):

        c1, c2, c3 = self.spm(input_img)
        c1, c2, c3 = self._add_level_embed(c1, c2, c3)
        c = torch.cat([c1, c2, c3], dim=1)

        x = input_img
        x = self.adapter_patch_embed(x)

        encoder_list = []
        for i, layer in enumerate(self.interactions):
            deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
            indexes = self.interaction_indexes[i]
            bs, dim, H, W = x.shape
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            encoder_list.append(x)

        f1 = self.norm1(encoder_list[0])
        f2 = self.norm2(encoder_list[1])
        f3 = self.norm3(encoder_list[2])

        out_dec_level3 = self.decoder_level3(f3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, f2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, f1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + input_img

        return out_dec_level1