"""DOFASegmentationUperNet, extracted from DOFA_UperNet_decoder_peft_512_512.ipynb."""

import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

from .upernet import initialize_upernet_decoder, decode_upernet
from .wavelengths import SENTINEL2_L2A_WAVELENGTHS_UM


class DOFASegmentationUperNet(nn.Module):

    def __init__(self, dofa_backbone, wavelengths=SENTINEL2_L2A_WAVELENGTHS_UM,
                 num_classes=1, use_lora=True,
                 lora_r=32, lora_alpha=32, lora_dropout=0.1, fpn_channels=256):
        super().__init__()

        self.wavelengths = list(wavelengths)

        self.encoder = copy.deepcopy(dofa_backbone)
        self.patch_size = self.encoder.patch_size
        self._pretrain_grid = self.encoder.img_size // self.encoder.patch_size
        self._pos_embed_cache = {}

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,

                target_modules=["attn.qkv", "attn.proj"],
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            print("LoRA applied to DOFA encoder (qkv, proj attention projections)")
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        embed_dim = self._base_encoder().embed_dim

        self.skip_indices = [5, 11, 17, 23]

        initialize_upernet_decoder(self, embed_dim, fpn_channels, num_classes)

    def _base_encoder(self):
        """Return the underlying DOFA module, unwrapping PEFT if needed.
        LoRA replaces Linear layers in-place, so this is the same object
        graph with qkv/proj swapped for LoRA-augmented layers."""
        return self.encoder.get_base_model() if hasattr(self.encoder, "get_base_model") else self.encoder

    def _interpolated_pos_embed(self, grid_size, device, dtype):
        """DOFA ships a fixed sin-cos pos embedding sized for its
        pretraining resolution (224x224 -> 14x14 patches). Bicubically
        interpolate it to match the actual input resolution -- the same
        idea as interpolate_pos_encoding=True for DINOv3."""
        if grid_size in self._pos_embed_cache:
            return self._pos_embed_cache[grid_size].to(device=device, dtype=dtype)

        base_encoder = self._base_encoder()
        pos_embed = base_encoder.pos_embed

        if grid_size == self._pretrain_grid:
            interp = pos_embed
        else:
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            D = patch_pos.shape[-1]
            old = self._pretrain_grid
            patch_pos = patch_pos.reshape(1, old, old, D).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(grid_size, grid_size),
                                       mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, D)
            interp = torch.cat([cls_pos, patch_pos], dim=1)

        self._pos_embed_cache[grid_size] = interp.detach()
        return interp.to(device=device, dtype=dtype)

    def _extract_hidden_states(self, x):
        """Manually replay DOFA's forward_features, but keep every block's
        output instead of only the final pooled embedding -- mirroring the
        output_hidden_states=True interface used for DINOv3, so the UperNet
        decoder below needs no changes."""
        base_encoder = self._base_encoder()

        H, W = x.shape[-2], x.shape[-1]
        assert H == W and H % self.patch_size == 0,\
            f"Expected square input divisible by patch_size={self.patch_size}, got {H}x{W}"
        grid_size = H // self.patch_size

        wavelist = torch.tensor(self.wavelengths, device=x.device, dtype=torch.float32)
        patches, _ = base_encoder.patch_embed(x, wavelist)

        pos_embed = self._interpolated_pos_embed(grid_size, x.device, patches.dtype)
        patches = patches + pos_embed[:, 1:, :]

        cls_token = base_encoder.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1)

        hidden_states = [tokens]
        for block in base_encoder.blocks:
            tokens = block(tokens)
            hidden_states.append(tokens)

        return hidden_states

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor [B, 12, H, W] (12-channel Sentinel-2, fed
               directly into DOFA -- no channel adapter needed)
        Returns:
            Output tensor [B, num_classes, H, W]
        """

        all_states = self._extract_hidden_states(x)

        raw_skips = [all_states[i + 1] for i in self.skip_indices]

        processed_skips = []
        for feat in raw_skips:

            patch_tokens = feat[:, 1:, :]
            B, N, C = patch_tokens.shape
            H = W = int(np.sqrt(N))

            processed_skips.append(patch_tokens.permute(0, 2, 1).reshape(B, C, H, W))

        return decode_upernet(self, processed_skips, x.shape[2:])

    def print_trainable_parameters(self):
        """Print summary of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("Model Parameter Summary (DOFA-LoRA)")
        print("=" * 70)
        print(f"{'Component':<20} {'Total':>15} {'Trainable':>15} {'%':>10}")
        print("-" * 70)

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"{'Encoder (LoRA)':<20} {enc_params:>15,} {enc_trainable:>15,} {enc_trainable / enc_params * 100:>9.2f}%")

        dec_modules = [self.lateral_projections, self.ppm, self.fpn_top,
                       self.fpn_blocks, self.fpn_fusion, self.final_conv]
        dec_params = sum(p.numel() for m in dec_modules for p in m.parameters())
        dec_trainable = sum(p.numel() for m in dec_modules for p in m.parameters() if p.requires_grad)
        print(f"{'Decoder (UperNet)':<20} {dec_params:>15,} {dec_trainable:>15,} {dec_trainable / dec_params * 100:>9.2f}%")

        print("-" * 70)
        print(f"{'TOTAL':<20} {total_params:>15,} {trainable_params:>15,} {trainable_params / total_params * 100:>9.2f}%")
        print("=" * 70)