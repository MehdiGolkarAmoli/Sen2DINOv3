"""UPerNet components shared verbatim by the foundation-model notebooks."""

import torch
from torch import nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6], out_channels=512):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.out_channels = out_channels

        self.pool_branches = nn.ModuleList()
        for pool_size in pool_sizes:
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            self.pool_branches.append(branch)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pooled_features = [x]
        for branch in self.pool_branches:
            pooled = branch(x)
            pooled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pooled_features.append(pooled)
        x = torch.cat(pooled_features, dim=1)
        x = self.fusion_conv(x)
        return x


class FPNBlock(nn.Module):
    def __init__(self, lateral_channels, fpn_channels=256):
        super().__init__()
        self.lateral_conv = nn.Sequential(
            nn.Conv2d(lateral_channels, fpn_channels, kernel_size=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, top_down_feat, lateral_feat):
        lateral = self.lateral_conv(lateral_feat)
        if lateral.shape[2:] != top_down_feat.shape[2:]:
            lateral = F.interpolate(lateral, size=top_down_feat.shape[2:],
                                    mode='bilinear', align_corners=False)
        fused = top_down_feat + lateral
        out = self.fpn_conv(fused)
        return out


def initialize_upernet_decoder(self, embed_dim, fpn_channels, num_classes):
    """Register the original decoder layers in their original order."""
    self.lateral_projections = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(embed_dim, fpn_channels, kernel_size=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        ) for _ in self.skip_indices
    ])
    self.ppm = PyramidPoolingModule(
        in_channels=embed_dim, pool_sizes=[1, 2, 3, 6], out_channels=512
    )
    self.fpn_top = nn.Sequential(
        nn.Conv2d(512, fpn_channels, kernel_size=1),
        nn.BatchNorm2d(fpn_channels),
        nn.ReLU(inplace=True),
    )
    self.fpn_blocks = nn.ModuleList([
        FPNBlock(lateral_channels=embed_dim, fpn_channels=fpn_channels)
        for _ in range(len(self.skip_indices))
    ])
    self.fpn_fusion = nn.Sequential(
        nn.Conv2d(fpn_channels * len(self.skip_indices), fpn_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(fpn_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(fpn_channels),
        nn.ReLU(inplace=True),
    )
    self.final_conv = nn.Sequential(
        nn.Conv2d(fpn_channels, fpn_channels // 2, kernel_size=3, padding=1),
        nn.BatchNorm2d(fpn_channels // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(fpn_channels // 2, num_classes, kernel_size=1),
    )


def decode_upernet(self, processed_skips, input_size):
    """Apply the original PPM, top-down FPN, fusion, and prediction head."""
    ppm_out = self.ppm(processed_skips[-1])
    fpn_feat = self.fpn_top(ppm_out)
    fpn_outputs = [fpn_feat]
    for level_idx in range(len(self.skip_indices) - 2, -1, -1):
        fpn_feat = F.interpolate(fpn_feat, scale_factor=2, mode='bilinear', align_corners=False)
        lateral_feat = processed_skips[level_idx]
        fpn_feat = self.fpn_blocks[level_idx + 1](fpn_feat, lateral_feat)
        fpn_outputs.append(fpn_feat)
    fpn_outputs = fpn_outputs[::-1]
    target_size = fpn_outputs[0].shape[2:]
    upsampled_fpn = []
    for fpn_out in fpn_outputs:
        if fpn_out.shape[2:] != target_size:
            upsampled = F.interpolate(fpn_out, size=target_size, mode='bilinear', align_corners=False)
        else:
            upsampled = fpn_out
        upsampled_fpn.append(upsampled)
    fused_features = torch.cat(upsampled_fpn, dim=1)
    fused_features = self.fpn_fusion(fused_features)
    if fused_features.shape[2:] != input_size:
        fused_features = F.interpolate(fused_features, size=input_size, mode='bilinear', align_corners=False)
    output = self.final_conv(fused_features)
    return output