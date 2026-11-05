"""MAE-SSL4EO, extracted from MAE_ViT_Large_UperNet.ipynb."""

import numpy as np
from torch import nn
import timm
from torchgeo.models import ViTLarge16_Weights

from .upernet import initialize_upernet_decoder, decode_upernet
from .adapters import MAESpectralAdapter as SentinelSpectralAdapter


def create_mae_vit_large(weights = ViTLarge16_Weights.SENTINEL2_ALL_MAE):

    model = timm.create_model(
        'vit_large_patch16_224',
        pretrained = False,
        in_chans = 13,
        num_classes =0 ,
        global_pool = ''
    )

    model.load_state_dict(weights.get_state_dict(progress= True), strict = False)
    embed_dim = model.embed_dim
    print(f"MAE ViT-Large loaded successully!")
    print(f"Embed dimension: {embed_dim}")
    print(f"Number of blocks: {len(model.blocks)}")

    return model, embed_dim


class MAE_SSL4EO(nn.Module):
    """
    Semantic Segmentation model using MAE-pretrained ViT-Large encoder
    with UperNet decoder for Sentinel-2 imagery.

    Architecture:
    1. Spectral Adapter: 12-ch Sentinel-2 → 13-ch MAE input
    2. MAE ViT-Large encoder (frozen)
    3. UperNet decoder with PPM and FPN
    """

    def __init__(
        self,
        in_channels=12,
        num_classes=1,
        freeze_encoder=True,
        fpn_channels=256,
        mae_weights=ViTLarge16_Weights.SENTINEL2_ALL_MAE
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

        self.adapter = SentinelSpectralAdapter(
            in_channels=in_channels,
            out_channels=13
        )

        self.encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            in_chans=13,
            num_classes=0,
            global_pool='',
        )

        self.encoder.load_state_dict(
            mae_weights.get_state_dict(progress=True),
            strict=False
        )
        print(f"MAE weights loaded from: {mae_weights}")

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen (feature extraction mode)")

        self.embed_dim = self.encoder.embed_dim

        self.skip_indices = [5, 11, 17, 23]

        initialize_upernet_decoder(self, self.embed_dim, fpn_channels, num_classes)

        self.intermediate_features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features from ViT blocks."""
        def get_hook(name):
            def hook(module, input, output):
                self.intermediate_features[name] = output
            return hook

        for idx in self.skip_indices:
            self.encoder.blocks[idx].register_forward_hook(get_hook(f'block_{idx}'))

    def _tokens_to_spatial(self, tokens, has_cls_token=True):
        """
        Convert ViT tokens to spatial feature map.

        Args:
            tokens: [B, N, C] where N = 1 (cls) + H*W (patches)
            has_cls_token: Whether first token is CLS token

        Returns:
            Spatial features [B, C, H, W]
        """
        if has_cls_token:
            patch_tokens = tokens[:, 1:, :]
        else:
            patch_tokens = tokens

        B, N, C = patch_tokens.shape
        H = W = int(np.sqrt(N))

        spatial = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return spatial

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 12, H, W] (12-channel Sentinel-2)

        Returns:
            Output tensor [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        x_adapted = self.adapter(x)

        self.intermediate_features.clear()

        _ = self.encoder(x_adapted)

        processed_skips = []
        for idx in self.skip_indices:
            feat = self.intermediate_features[f'block_{idx}']
            spatial_feat = self._tokens_to_spatial(feat, has_cls_token=True)
            processed_skips.append(spatial_feat)

        return decode_upernet(self, processed_skips, input_size)

    def print_trainable_parameters(self):
        """Print summary of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*70)
        print("Model Parameter Summary (MAE-SSL4EO)")
        print("="*70)
        print(f"{'Component':<25} {'Total':>15} {'Trainable':>15} {'%':>10}")
        print("-" * 70)

        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        adapter_trainable = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"{'Spectral Adapter':<25} {adapter_params:>15,} {adapter_trainable:>15,} {adapter_trainable/max(adapter_params,1)*100:>9.2f}%")

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        enc_status = "Frozen" if self.freeze_encoder else "Trainable"
        print(f"{'Encoder (MAE ViT-L) ' + enc_status:<25} {enc_params:>15,} {enc_trainable:>15,} {enc_trainable/max(enc_params,1)*100:>9.2f}%")

        dec_params = sum(p.numel() for p in self._decoder_parameters())
        dec_trainable = sum(p.numel() for p in self._decoder_parameters() if p.requires_grad)
        print(f"{'Decoder (UperNet)':<25} {dec_params:>15,} {dec_trainable:>15,} {dec_trainable/max(dec_params,1)*100:>9.2f}%")

        print("-" * 70)
        print(f"{'TOTAL':<25} {total_params:>15,} {trainable_params:>15,} {trainable_params/total_params*100:>9.2f}%")
        print("="*70)

    def _decoder_parameters(self):
        """Helper to get decoder parameters."""
        params = []
        for module in [self.lateral_projections, self.ppm, self.fpn_top,
                       self.fpn_blocks, self.fpn_fusion, self.final_conv]:
            params.extend(module.parameters())
        return params