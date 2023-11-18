import pytorch_lightning as pl
from models.resnet_up_dim import Resnet_up_dim
import torch.nn as nn
import torch
from datasets.structured3d import Structured3D
from torch.utils.data.dataloader import DataLoader
import h5py
from models.model_vit import ViTModel
from utils.loss import Loss
class Detector(pl.LightningModule):
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
        self.h5_data = h5py.File('data/structured3d.h5','r')
        self.dropout = nn.Dropout(p=0.1)
        
        self.loss = Loss()
        self.frozen_layer()

    def forward(self, x, mask, position_embedding):
        B, N, D, H, W = x.shape
        x = self.resnet_up(x.view(B*N, D, H, W))   # [256,14,14] --> [2048,14,14]
        x = x.mean(dim=[2,3])   # [2048,2,2] --> [2048]
        x = self.layernorm_before(x)
        x = self.dim_down(x)    # [2048] --> [768]
        # x = self.dropout(x)
        x = x.view(B,N,-1)
        x = x + position_embedding
        mask_vit = (1 - torch.matmul(mask.unsqueeze(-1),mask.unsqueeze(-2))) * (-1e6)
        mask_vit = mask_vit[:,None,:,:]
        x = self.vit(x,whole_mask = mask_vit)
        x = self.calculate_param(x[0])
        # x = self.calculate_param(x)
        x = x*mask.unsqueeze(-1).repeat(1,1,4)
        return x

    def training_step(self, inputs, batch_idx):
        for key, value in inputs.items():
            inputs[key] = value.to('cuda')
        
        x = self.forward(inputs['feature'], inputs['mask'], inputs['position_embedding'])
        # loss = self.criterion(x, inputs['param'])
        loss = self.loss(x,inputs['param'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, inputs, batch_index):
        for key, value in inputs.items():
            inputs[key] = value.to('cuda')
        
        x = self.forward(inputs['feature'], inputs['mask'], inputs['position_embedding'])
        # loss = self.criterion(x, inputs['param'])
        loss = self.loss(x,inputs['param'])
        self.log('val_loss',loss)

    def test_step(self, inputs, batch_index):
        pass

    def init_weights(self, pretrained=None):
        pass

    def frozen_layer(self):
        for name, p in self.named_parameters():
            if 'vit' in name:
                print('frozen:',name)
                p.requires_grad = False

    # def criterion(self,pred,target):
    #     criterion = Loss()
    #     loss = criterion(pred, target)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr)
        return [optimizer]
    #     dataset = Structured3D(self.h5_data, 'training')
        
    #     self.train_dataloader_loader = DataLoader(dataset = dataset, 
    #                             batch_size=self.args.batch_size_train,
    #                             shuffle=True,
    #                             num_workers=self.args.num_workers,
    #                             drop_last=True,
    #                             collate_fn=dataset.collate_fn)
    #     self.train_dataloader_length = len(self.train_dataloader_loader)
    #     batch_per_epoch = self.train_dataloader_length
    #     t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
    #     warmup_ratio = self.args.warmup_ratio
    #     warmup_iters = int(t_total * warmup_ratio)
    #     optimizer_grouped_parameters = [
    #         {
    #             "params" : [p for n, p in self.resnet_up.named_parameters()],
    #             "weight_decay" : self.args.weight_decay
    #         },
    #         {
    #             "params" : [p for n, p in self.dim_down.named_parameters()],
    #             "weight_decay" : self.args.weight_decay
    #         },
    #         {
    #             "params" : [p for n, p in self.calculate_param.named_parameters()],
    #             "weight_decay" : self.args.weight_decay
    #         }
    #     ]
    #     optimizer = AdamW(optimizer_grouped_parameters,
    #                                  lr=self.args.lr, eps=self.args.adam_eps)
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #             optimizer, warmup_iters, t_total)
    #     return ([optimizer], [lr_scheduler])
    

    def train_dataloader(self):
        dataset = Structured3D(self.h5_data, 'training')
        dataloader = DataLoader(dataset = dataset, 
                                batch_size=self.args.batch_size_train,
                                shuffle=True,
                                num_workers=self.args.num_workers,
                                drop_last=True,
                                collate_fn=dataset.collate_fn)
        # self.train_dataloader_length = len(dataloader)
        return dataloader

    def val_dataloader(self):
        dataset = Structured3D(self.h5_data, 'validation')
        dataloader = DataLoader(dataset = dataset, 
                                batch_size=self.args.batch_size_val,
                                num_workers=self.args.num_workers,
                                collate_fn=dataset.collate_fn)
        return dataloader

    def test_dataloader(self):
        dataset = Structured3D(self.h5_data, 'test')
        dataloader = DataLoader(dataset = dataset, 
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                collate_fn=dataset.collate_fn)
        return dataloader