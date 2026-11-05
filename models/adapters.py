"""Spectral adapters, with the DINOv3 projection aligned to manuscript Equation 5."""

from torch import nn


class DinoSpectralAdapter(nn.Module):
    """
    Spectral adapter for Sentinel-2 12-channel input.
    Adapts 12-channel Sentinel-2 data to 3-channel RGB format for DINOv3.
    """
    def __init__(self, in_channels=12, out_channels=3):
        super().__init__()
        self.se_fc1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.se_fc2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        se = x.mean((2, 3), keepdim=True)
        se = self.relu(self.se_fc1(se))
        se = self.sigmoid(self.se_fc2(se))
        x = x * se
        x = self.projection(x)
        return x


class MAESpectralAdapter(nn.Module):

    def __init__(self, in_channels=12, out_channels = 13):
       super().__init__()
       self.in_channels = in_channels
       self.out_channels = out_channels

       reduction = max(in_channels//4, 1)
       self.se_fc1 = nn.Conv2d(in_channels, reduction, kernel_size=1)
       self.relu = nn.ReLU(inplace = True)
       self.se_fc2 = nn.Conv2d(reduction, in_channels, kernel_size= 1)
       self.sigmoid = nn.Sigmoid()

       self.projection = nn.Sequential(
        nn.Conv2d(in_channels, 36, kernel_size = 1),
        nn.GELU(),
        nn.Conv2d(36, out_channels, kernel_size=1)
       )

       self._init_weights()

    def _init_weights(self):

        for m in self.projection.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out',  nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        se = x.mean((2, 3), keepdim=True)
        se = self.relu(self.se_fc1(se))
        se = self.sigmoid(self.se_fc2(se))
        x = x * se

        x = self.projection(x)
        return x