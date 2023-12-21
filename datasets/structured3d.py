import json
import os
from collections import defaultdict
import h5py
import cv2
import numpy as np
import torchvision.transforms as tf
from torch.utils import data
import torch
from positional_encodings.torch_encodings import PositionalEncoding2D

class Structured3D(data.Dataset):
    def __init__(self, h5_data,phase='training'):
        self.phase = phase

        self.data_set = defaultdict(list)
        for _, img_name in enumerate(h5_data.keys()):
            scene = int(img_name.split('_')[1])
            if scene <= 2999:
                self.data_set['training'].append(img_name)
            elif scene <= 3249:
                self.data_set['validation'].append(img_name)
            else:
                self.data_set['test'].append(img_name)
        self.h5_data = h5_data
        self.output = 14

        position_embedding = PositionalEncoding2D(768)
        self.pos_emb = position_embedding(torch.zeros(1,380, 640,768)).squeeze(0)

    def __getitem__(self, idx):
        img_name = self.data_set[self.phase][idx]
        feature = self.h5_data[img_name]['feature'][()]
        boxes_ceiling = self.h5_data[img_name]['boxes_ceiling'][()]
        boxes_floor = self.h5_data[img_name]['boxes_floor'][()]
        boxes_wall = self.h5_data[img_name]['boxes_wall'][()]

        center_ceiling_w = ((boxes_ceiling[:,2] + boxes_ceiling[:,0]) / 2).reshape(-1,1)
        center_ceiling_h = ((boxes_ceiling[:,3] + boxes_ceiling[:,1]) / 2).reshape(-1,1)
        center_floor_w = ((boxes_floor[:,2] + boxes_floor[:,0]) / 2).reshape(-1,1)
        center_floor_h = ((boxes_floor[:,3] + boxes_floor[:,1]) / 2).reshape(-1,1)
        center_wall_w = ((boxes_wall[:,2] + boxes_wall[:,0]) / 2).reshape(-1,1)
        center_wall_h = ((boxes_wall[:,3] + boxes_wall[:,1]) / 2).reshape(-1,1)

        center_ceiling = np.concatenate([center_ceiling_w, center_ceiling_h], axis=-1)
        center_floor = np.concatenate([center_floor_w, center_floor_h], axis=-1)
        center_wall = np.concatenate([center_wall_w, center_wall_h], axis=-1)

        center = np.concatenate([center_wall, center_floor, center_ceiling], axis=0)
        
        center_int = np.floor(center)
        center_delta = center - center_int
        position_embedding_one = torch.tensor([],dtype=torch.float32)
        for w, h in center_int.astype(np.int32):
            position_embedding_one = torch.cat([position_embedding_one,self.pos_emb[h,w].unsqueeze(0)],dim=0)

        param_ceiling = self.h5_data[img_name]['param_ceiling'][()]
        param_floor = self.h5_data[img_name]['param_floor'][()]
        param_wall = self.h5_data[img_name]['param_wall'][()]

        flag = self.h5_data[img_name]['flag'][()]
        ratio = self.h5_data[img_name]['ratio'][()]

        boxes = np.concatenate([boxes_wall, boxes_floor, boxes_ceiling],axis=0)
        boxes = np.concatenate([np.zeros((boxes.shape[0],1)).astype(np.float32), boxes], axis=1)
        param = np.concatenate([param_wall, param_floor, param_ceiling],axis=0)
        mask = np.ones(param.shape[0]).astype(np.float32)
        # feature, _ = torch.ops.torchvision.roi_pool(torch.tensor(feature), torch.tensor(boxes), ratio, self.output, self.output)
        batch_input = {
            'feature':torch.tensor(feature),
            'param':param,
            'mask':mask,
            'flag':flag,
            'center_int':center_int,
            'center_delta':center_delta,
            'position_embedding_one':position_embedding_one
        }
        return batch_input

    def collate_fn(self, batch):
        MAX_L = 0
        B = len(batch)
        D = 256
        length_one = []
        for batch_one in batch:
            if batch_one['feature'].shape[0] > MAX_L:
                MAX_L = batch_one['feature'].shape[0]
            length_one.append(batch_one['feature'].shape[0])
        feature_batch = torch.zeros(B, MAX_L, D, self.output,self.output)
        param_batch = torch.zeros(B, MAX_L, 4)
        mask_batch = torch.zeros(B, MAX_L)
        flag_batch = torch.zeros(B, 2)
        center_int_batch = torch.zeros(B, MAX_L, 2)
        center_delta_batch = torch.zeros(B, MAX_L, 2)
        position_embedding_batch = torch.zeros(B,MAX_L,768)

        for i, length in enumerate(length_one):
            feature_batch[i,:length] = batch[i]['feature']
            param_batch[i,:length] = torch.tensor(batch[i]['param'])
            mask_batch[i,:length] = torch.tensor(batch[i]['mask'])
            flag_batch[i] = torch.tensor(batch[i]['flag'])
            center_int_batch[i,:length] = torch.tensor(batch[i]['center_int'])
            center_delta_batch[i,:length] = torch.tensor(batch[i]['center_delta'])
            position_embedding_batch[i,:length] = batch[i]['position_embedding_one']
        batch_entry = {
            'feature':feature_batch,
            'param':param_batch,
            'mask':mask_batch,
            'flag':flag_batch,
            'center_int':center_int_batch,
            'center_delta':center_delta_batch,
            'position_embedding':position_embedding_batch
        }
        return batch_entry

    def __len__(self):
        return len(self.data_set[self.phase])

if __name__ == '__main__':
    h5_data = h5py.File('/home/zhangjx/All_model/Non_cubiod/room_layout_transformer/data/structured3d.h5','r')
    train_strcutured3d = Structured3D(h5_data, phase='training')
    val_strcutured3d = Structured3D(h5_data, phase='validation')
    test_strcutured3d = Structured3D(h5_data, phase='test')
    train_dataloader = torch.utils.data.DataLoader(train_strcutured3d,batch_size=48, collate_fn=train_strcutured3d.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_strcutured3d,batch_size=1, collate_fn=val_strcutured3d.collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_strcutured3d,batch_size=1, collate_fn=test_strcutured3d.collate_fn)
    for i,data_one in enumerate(train_dataloader):
        if i % 1000 == 0:
            print(i)
    for i,data_one in enumerate(val_dataloader):
        if i % 1000 == 0:
            print(i)
    for i, data_one in enumerate(test_dataloader):
        if i % 1000 == 0:
            print(i)
    pass