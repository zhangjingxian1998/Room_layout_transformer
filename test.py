from models.model import Model
import h5py
from utils.loss import Loss
import numpy as np
import torch
import random
from utils.utils import parse, build_dataloader, build_optim
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import time

def main(args):

    h5_data = h5py.File('data/structured3d.h5','r')
    loss_f = Loss()
    test_dataloader = build_dataloader(args, h5_data, 'test')
    model = torch.load('/home/zhangjx/All_model/Non_cubiod/room_layout_transformer/log/2023_12_13_10_17_13/BEST.ckpt').to('cuda')
    device = next(model.parameters()).device
    pbar = tqdm(total=len(test_dataloader), ncols=120, desc='test')
    model.eval()
    loss_sum = 0
    for step_i, data in enumerate(test_dataloader):
        for key, value in data.items():
            data[key] = value.to(device)
        output = model(data)
        # output[:,:,:3] = output[:,:,:3] / torch.norm(output[:,:,:3], dim=-1)
        loss = loss_f(output, data['param'], data['mask'])
        
        loss_meter = loss.item()
        loss_sum += loss_meter
        desc_str = 'TEST'
        desc_str += f' | Loss {loss_meter:4f}'
        pbar.set_description(desc_str)
        pbar.update(1)

        pbar.close()
    print(loss_sum/len(test_dataloader))

if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse()
    random.seed(2333)
    torch.manual_seed(2333)
    np.random.seed(2333)
    torch.cuda.manual_seed(2333)
    main(args)