import torch.nn as nn
import torch
from models.resnet_up_dim import Resnet_up_dim
from models.model_vit import ViTModel
import h5py
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.resnet_up = Resnet_up_dim()
        self.layernorm_before = nn.LayerNorm(2048)
        self.dim_down = nn.Sequential(
            nn.Linear(2048,768),
            nn.LayerNorm(768)
        )
        self.vit = ViTModel.from_pretrained('pretrain_weight/vit-base-patch32-224-in21k')
        self.calculate_param = nn.Linear(768,4)
        self.dropout = nn.Dropout(p=0.1)
        
        self.frozen_layer()

    def forward(self, data):
        B, N, D, H, W = data['feature'].shape
        x = self.resnet_up(data['feature'].view(B*N, D, H, W))   # [256,14,14] --> [2048,14,14]
        x = x.mean(dim=[2,3])   # [2048,2,2] --> [2048]
        x = self.layernorm_before(x)
        x = self.dim_down(x)    # [2048] --> [768]
        # x = self.dropout(x)
        x = x.view(B,N,-1)
        x = x + data['position_embedding']
        mask_vit = (1 - torch.matmul(data['mask'].unsqueeze(-1),data['mask'].unsqueeze(-2))) * (-1e6)
        mask_vit = mask_vit[:,None,:,:]
        x = self.vit(x,whole_mask=mask_vit)
        x = self.calculate_param(x[0])
        # x = self.calculate_param(x)
        x = x * data['mask'].unsqueeze(-1).repeat(1,1,4)
        return x

    def frozen_layer(self):
        for name, p in self.named_parameters():
            if 'vit' in name:
                # print('frozen:',name)
                p.requires_grad = False