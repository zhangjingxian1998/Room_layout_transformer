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
def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

def main(args, gpu):
    args.gpu = gpu
    args.rank = gpu

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')
        if args.gpu == 0:
            args.verbose = True
        else:
            args.verbose = False

    print(f'Process Launching at GPU {gpu}')
    h5_data = h5py.File('data/structured3d.h5','r')
    model = Model(args).to(args.gpu)
    loss_f = Loss()
    print(f'Building train loader at GPU {gpu}')
    train_dataloader = build_dataloader(args, h5_data, 'training')
    print(f'Building val loader at GPU {gpu}')
    val_dataloader = build_dataloader(args, h5_data, 'validation')
    optimizer, scheduler = build_optim(args, model.parameters())

    if args.multiGPU:
        if args.distributed:
            model = DDP(model, 
                        device_ids=[args.gpu],
                        # output_device=args.gpu,
                        # find_unused_parameters=True
                                )
    device = next(model.parameters()).device
    if args.distributed: # 在分布进程间同步
        dist.barrier()
    for epoch in range(args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        if args.verbose:
            pbar = tqdm(total=len(train_dataloader), ncols=120, desc='training')
        model.train()
        for step_i, data in enumerate(train_dataloader):
            for key, value in data.items():
                data[key] = value.to(device)
            if args.distributed:
                output = model(data)
                output[:,:,:3] = output[:,:,:3] / torch.norm(output[:,:,:3], dim=-1, keepdim=True)
            else:
                output = model(data)
                output[:,:,:3] = output[:,:,:3] / torch.norm(output[:,:,:3], dim=-1, keepdim=True)
            loss = loss_f(output, data['param'])
            optimizer.zero_grad()
            # if args.distributed: # 在分布进程间同步
            #     dist.barrier()
            # loss = reduce_value(loss)
            # dist.all_reduce(loss)
            
            loss.backward()
            # loss = reduce_value(loss)
            optimizer.step()
            # loss.detach()

            try:
                lr = optimizer.get_lr()[0]
            except AttributeError:
                lr = args.lr
            
            if args.verbose:
                loss_meter = loss.item()
                desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                desc_str += f' | Loss {loss_meter:4f}'
                pbar.set_description(desc_str)
                pbar.update(1)
            # print(f'\nstart_sleep{args.gpu}')
            # time.sleep(10)
            if args.distributed:
                dist.barrier()
            
        if args.verbose:    
            pbar.close()
        if args.verbose:
            torch.save(model.module, f"saved_model_{epoch}_{loss_meter}.ckpt")
        if args.verbose:
            with torch.no_grad():
                model.eval()
                pbar = tqdm(total=len(val_dataloader), ncols=120, desc='validation')
                for step_i, data in enumerate(val_dataloader):
                    for key, value in data.items():
                        data[key] = value.to(device)
                    output = model.module(data)
                    loss = loss_f(output, data['param'])

                    loss_meter = loss.item()
                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                    desc_str += f' | Loss {loss_meter:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    random.seed(2333)
    torch.manual_seed(2333)
    np.random.seed(2333)
    torch.cuda.manual_seed(2333)
    main(args, args.local_rank)