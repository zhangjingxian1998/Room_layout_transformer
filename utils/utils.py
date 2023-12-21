import argparse
from datasets.structured3d import Structured3D
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.optim.lr_scheduler import StepLR
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='train') # train test
    parser.add_argument('--data' ,default='Structured3D')
    parser.add_argument('--batch_size_train',default=64)
    parser.add_argument('--batch_size_val',default=64)
    parser.add_argument('--batch_size_test',default=1)
    parser.add_argument('--num_workers',default=16)
    parser.add_argument('--epochs',default=10)
    parser.add_argument('--weight',default=' ')
    
    parser.add_argument('--gpu_num',default=2)
    parser.add_argument('--test_gpu_id',default=0)

    parser.add_argument('--lr',default=5e-5)
    parser.add_argument('--weight_decay',default=0.0001)
    parser.add_argument('--adam_eps',default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--warmup_ratio',default=0.05)
    parser.add_argument('--gradient_accumulation_steps',default=1)
    parser.add_argument('--clip_grad_norm',default=1)

    parser.add_argument('--multiGPU', action="store_true")
    parser.add_argument('--distributed', action="store_true")
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def build_dataloader(args, h5_data, mode):
    if mode == 'training':
        dataset = Structured3D(h5_data, mode)
        if args.distributed:
            train_sampler = DistributedSampler(dataset)
        else:
            train_sampler = None
        dataloader = DataLoader(dataset = dataset, 
                                    batch_size=args.batch_size_train,
                                    shuffle=(train_sampler is None),
                                    num_workers=args.num_workers,
                                    drop_last=True,
                                    sampler=train_sampler,
                                    # pin_memory=True,
                                    collate_fn=dataset.collate_fn)
    elif mode == 'validation':
        dataset = Structured3D(h5_data, 'validation')
        dataloader = DataLoader(dataset = dataset, 
                                batch_size=args.batch_size_val,
                                num_workers=args.num_workers,
                                collate_fn=dataset.collate_fn)
    elif mode == 'test':
        dataset = Structured3D(h5_data, 'test')
        dataloader = DataLoader(dataset = dataset, 
                                batch_size=1,
                                num_workers=args.num_workers,
                                # pin_memory=True,
                                collate_fn=dataset.collate_fn)
    return dataloader

def build_optim(args, train_dataloader, parameters):
    batch_per_epoch = len(train_dataloader)
    t_total = batch_per_epoch // args.gradient_accumulation_steps * args.epochs
    warmup_ratio = args.warmup_ratio
    warmup_iters = int(t_total * warmup_ratio)

    lr_scheduler = None
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters), 
    #                              lr=args.lr)
    # scheduler_1 = StepLR(optimizer, step_size=3, gamma=0.1)

    optimizer = AdamW(filter(lambda p: p.requires_grad, parameters),
                          lr=args.lr, 
                          eps=args.adam_eps, 
                          weight_decay=args.weight_decay,
                          betas=(args.adam_beta1,args.adam_beta2),
                          no_deprecation_warning=True)
    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, warmup_iters, t_total)
    return optimizer, lr_scheduler