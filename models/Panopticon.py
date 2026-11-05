"""PanopticonSegmentationUperNet, extracted from Panopticon_ViTB14_UperNet_decoder_peft_518_518.ipynb."""

import copy

import torch
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model

from .upernet import initialize_upernet_decoder, decode_upernet
from .wavelengths import SENTINEL2_L2A_WAVELENGTHS_NM


class PanopticonSegmentationUperNet(nn.Module):
    """Panopticon ViT-B/14 with wavelength inputs, LoRA, hooks, and UPerNet."""

    def __init__(self, panopticon_backbone, in_channels=12, num_classes=1, use_lora=True,
                 lora_r=32, lora_alpha=32, lora_dropout=0.05, fpn_channels=256,
                 channel_wavelengths=None, train_patch_embed=False):
        super().__init__()

        self.use_lora = use_lora
        self.encoder = copy.deepcopy(panopticon_backbone)

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
            print("LoRA applied to Panopticon encoder")
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if train_patch_embed:
            base = self.encoder.get_base_model() if use_lora else self.encoder
            for param in base.model.patch_embed.parameters():
                param.requires_grad = True

        embed_dim = 768
        self.skip_indices = [2, 5, 8, 11]
        self.num_prefix_tokens = 1

        wavelengths = channel_wavelengths if channel_wavelengths is not None else SENTINEL2_L2A_WAVELENGTHS_NM
        assert len(wavelengths) == in_channels, (
            f"channel_wavelengths has {len(wavelengths)} entries but in_channels={in_channels}"
        )
        self.register_buffer(
            'chn_ids_template',
            torch.tensor(wavelengths, dtype=torch.float32).unsqueeze(0)
        )

        initialize_upernet_decoder(self, embed_dim, fpn_channels, num_classes)

        self._captured = {}
        self._hook_handles = []
        self._register_hooks()

    def _get_timm_vit(self):
        base = self.encoder.get_base_model() if hasattr(self.encoder, 'get_base_model') else self.encoder
        return base.model

    def _register_hooks(self):
        vit = self._get_timm_vit()

        def make_hook(idx):
            def hook(module, inp, output):
                self._captured[idx] = output
            return hook

        for idx in self.skip_indices:
            handle = vit.blocks[idx].register_forward_hook(make_hook(idx))
            self._hook_handles.append(handle)

    def _forward_encoder(self, x_dict):

        if hasattr(self.encoder, 'get_base_model'):
            return self.encoder.get_base_model()(x_dict)
        return self.encoder(x_dict)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor [B, 12, H, W] (12-channel Sentinel-2, L2A band order)
        Returns:
            Output tensor [B, num_classes, H, W]
        """

        B = x.shape[0]
        chn_ids = self.chn_ids_template.expand(B, -1).clone()
        x_dict = {'imgs': x, 'chn_ids': chn_ids}

        self._captured.clear()
        _ = self._forward_encoder(x_dict)

        raw_skips = [self._captured[i] for i in self.skip_indices]

        processed_skips = []
        for feat in raw_skips:

            patch_tokens = feat[:, self.num_prefix_tokens:, :]
            Bp, N, C = patch_tokens.shape
            H = W = int(round(N ** 0.5))
            processed_skips.append(patch_tokens.permute(0, 2, 1).reshape(Bp, C, H, W))

        return decode_upernet(self, processed_skips, x.shape[2:])

    def print_trainable_parameters(self):
        """Print summary of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("Model Parameter Summary (Panopticon-LoRA)")
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