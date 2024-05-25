import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.models.archs.fftformer_arch import *
from basicsr.models.archs.adapter import SpatialPriorModule
from basicsr.models.archs.adapter import InteractionBlock
from basicsr.models.archs.adapter import deform_inputs
from timm.models.layers import trunc_normal_
import math
from basicsr.models.archs.ms_deform_attn import MSDeformAttn
# TODO: find out which version of the MSDeformAttn it's using

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape

        return x,H,W

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
        self.level_embed = nn.Parameter(torch.zeros(3, dim))

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

        # inherit all the layers from the pretrained model
        named_layers = dict(self.pretrained_model.named_children())
        for name, layer in named_layers.items():
            setattr(self, name, layer)

        self.load_pretrained_weights(pretrained)

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
        pretrained_state_dict = torch.load(weights)
        model_state_dict = self.state_dict()
        for name, param in pretrained_state_dict.items():
            model_state_dict[name].copy_(param)
            model_state_dict[name].requires_grad = False
        self.load_state_dict(model_state_dict)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape

        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()

        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        out_dec_level3 = self.decoder_level3(f4)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, f3)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, f2)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + x

        return out_dec_level1


