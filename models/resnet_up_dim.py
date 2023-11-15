import torch
import torch.nn as nn
# from models.resnet_build import ResNet, BottleneckBlock
from resnet_build import ResNet, BottleneckBlock
class Resnet_up_dim(nn.Module):
    def __init__(self):
        super(Resnet_up_dim, self).__init__()
        self.block0 = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=512 // 2,
            bottleneck_channels=128,
            out_channels=512,
            num_groups=1,
            norm='BN',
            stride_in_1x1=True,
        )

        self.block1 = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=1024 // 2,
            bottleneck_channels=256,
            out_channels=1024,
            num_groups=1,
            norm='BN',
            stride_in_1x1=True,
        )

        self.block2 = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=2048 // 2,
            bottleneck_channels=512,
            out_channels=2048,
            num_groups=1,
            norm='BN',
            stride_in_1x1=True,
        )

    def forward(self, x):
        x = self.block0(x) # [256] --> [512]
        x = self.block1(x)  # [512] --> [1024]
        x = self.block2(x)  # [1024] --> [2048]
        return x
    
if __name__ == '__main__':
    net = Resnet_up_dim()
    pass