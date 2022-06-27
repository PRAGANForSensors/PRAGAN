import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Sequential(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride))

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        # self.sigmoid = torch.sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return torch.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.sigmoid = torch.sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = torch.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel, kernel, UseBN=False):
        super(CBAM, self).__init__()
        body = []
        body.append(conv(channel, channel, kernel))
        if UseBN==True:
            body.append(nn.BatchNorm2d(num_features=channel))
            # body.append(nn.LayerNorm(normalized_shape=(channel, 256, 256)))
        body.append(nn.GELU())
        body.append(conv(channel, channel, kernel))
        self.body = nn.Sequential(*body)

        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, input):
        x = self.body(input)
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        out = out + input
        return out


