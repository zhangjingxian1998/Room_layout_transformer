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

def main(args, gpu):
    args.gpu = gpu
    args.rank = gpu

    print(f'Process Launching at GPU {gpu}')
    h5_data = h5py.File('data/structured3d.h5','r')
    # model = Model(args).to(args.gpu)
    loss_f = Loss()
    test_dataloader = build_dataloader(args, h5_data, 'training')
    model = torch.load('/home/zhangjx/All_model/Non_cubiod/room_layout_transformer/saved_model_15_0.5572562217712402.ckpt').to(args.gpu)
    device = next(model.parameters()).device
    if args.distributed: # 在分布进程间同步
        dist.barrier()
    for epoch in range(args.epochs):
        pbar = tqdm(total=len(test_dataloader), ncols=120, desc='training')
        model.eval()
        for step_i, data in enumerate(test_dataloader):
            for key, value in data.items():
                data[key] = value.to(device)
            output = model(data)
            output[:,:,:3] = output[:,:,:3] / torch.norm(output[:,:,:3], dim=-1)
            loss = loss_f(output, data['param'])
            
            loss_meter = loss.item()
            desc_str = f'Epoch {epoch}'
            desc_str += f' | Loss {loss_meter:4f}'
            pbar.set_description(desc_str)
            pbar.update(1)

        pbar.close()

if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    random.seed(2333)
    torch.manual_seed(2333)
    np.random.seed(2333)
    torch.cuda.manual_seed(2333)
    main(args, 0)