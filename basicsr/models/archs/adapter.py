import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.ms_deform_attn import MSDeformAttn
from timm.models.layers import DropPath
from einops import rearrange
import torch.utils.checkpoint as cp

_logger = logging.getLogger(__name__)

def flatten(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def unflatten(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Downsample(nn.Module):
    def __init__(self, dim, reduction_factor):
        super().__init__()
        self.downsample = nn.Conv2d(dim, dim, kernel_size=reduction_factor, stride=reduction_factor, padding=0, bias=False)

    def forward(self, x):
        # return cp.checkpoint(self.downsample, x)
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, dim, expansion_factor):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=expansion_factor, stride=expansion_factor, padding=0, bias=False)

    def forward(self, x):
        # return cp.checkpoint(self.upsample, x)
        return self.upsample(x)

class PatchMerging(nn.Module):
    """Patch merging inspired by swin Transformer.
    """
    def __init__(self, embed_dim):
        super(PatchMerging, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim * 4)
        self.conv = nn.Conv1d(embed_dim * 4, embed_dim * 2, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape

        # Reshape to [B, N/4, 4C]
        x = x.contiguous().view(B, N // 4, 4 * C)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Reshape for convolution [B, 4C, N/4]
        x = x.permute(0, 2, 1)

        # Apply convolution
        x = self.conv(x)

        # Reshape back to [B, N/4, 2C]
        x = x.permute(0, 2, 1)

        return x

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // patch_size, w // patch_size)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                                   (h // 16, w // 16),
                                                   (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H // 8, W // 8).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H // 16, W // 16).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 32, W // 32).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=False, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False, patch_merge = False, bias = False, with_extractor = True, patch_size = 4):
        super().__init__()

        self.patch_size = patch_size

        self.injector = Injector(dim=dim - dim*patch_merge//2, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)

        self.with_extractor = with_extractor

        if with_extractor:
            self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                    norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                    cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        else:
            self.extractor = None

        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

        if patch_merge:
            self.patch_merger = PatchMerging(embed_dim = dim//2)
        else:
            self.patch_merger = None

        self.downsample_1 = Downsample(dim=dim - dim*patch_merge//2, reduction_factor= patch_size)
        self.upsample_1 = Upsample(dim=dim - dim*patch_merge//2,expansion_factor=patch_size)

        self.downsample_2 = Downsample(dim=dim, reduction_factor= patch_size)
        self.upsample_2 = Upsample(dim=dim,expansion_factor=patch_size)

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):

        x_down = flatten(self.downsample_1(x))

        x_down = self.injector(query=x_down, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        x = x + self.upsample_1(unflatten(x_down, H // self.patch_size, W // self.patch_size))

        for idx, blk in enumerate(blocks):
            x = blk(x)

        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        if self.with_extractor:

            x_down = flatten(self.downsample_2(x))

            if self.patch_merger is not None:
                H = H // 2
                W = W // 2
                c = self.patch_merger(c)

            c = self.extractor(query=c, reference_points=deform_inputs2[0],
                            feat=x_down, spatial_shapes=deform_inputs2[1],
                            level_start_index=deform_inputs2[2], H=H, W=W)

            if self.extra_extractors is not None:
                for extractor in self.extra_extractors:
                    c = extractor(query=c, reference_points=deform_inputs2[0],
                                feat=x_down, spatial_shapes=deform_inputs2[1],
                                level_start_index=deform_inputs2[2], H=H, W=W)

            x = x + self.upsample_2(unflatten(x_down, H // self.patch_size, W // self.patch_size))

        return x, c

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=48):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c2, c3, c4