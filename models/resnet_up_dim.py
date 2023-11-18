import torch
import torch.nn as nn
from models.resnet_build import ResNet, BottleneckBlock
# from resnet_build import ResNet, BottleneckBlock
class Resnet_up_dim(nn.Module):
    def __init__(self, output_size=14):
        super(Resnet_up_dim, self).__init__()
        output_size = (output_size+1)//2
        block0 = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=512 // 2,
            bottleneck_channels=128,
            out_channels=512,
            num_groups=1,
            norm='Layernorm',
            stride_in_1x1=True,
            output_size=output_size
        )
        self.block0 = nn.Sequential(
            block0[0],
            block0[1],
            block0[2]
        )
        output_size = (output_size+1)//2
        block1 = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=1024 // 2,
            bottleneck_channels=256,
            out_channels=1024,
            num_groups=1,
            norm='Layernorm',
            stride_in_1x1=True,
            output_size=output_size
        )
        self.block1 = nn.Sequential(
            block1[0],
            block1[1],
            block1[2]
        )
        output_size = (output_size+1)//2
        block2 = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=2048 // 2,
            bottleneck_channels=512,
            out_channels=2048,
            num_groups=1,
            norm='Layernorm',
            stride_in_1x1=True,
            output_size=output_size
        )
        self.block2 = nn.Sequential(
            block2[0],
            block2[1],
            block2[2]
        )

    def forward(self, x):
        x = self.block0(x) # [256] --> [512]
        x = self.block1(x)  # [512] --> [1024]
        x = self.block2(x)  # [1024] --> [2048]
        return x
    
if __name__ == '__main__':
    net = Resnet_up_dim().to('cuda')
    a = torch.ones(1,256,14,14).to('cuda')
    net(a)
    pass