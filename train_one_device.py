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
from torch.utils.tensorboard import SummaryWriter
from packaging import version

def get_time():
    time_now = time.localtime()
    year = str(time_now.tm_year)
    mon = str(time_now.tm_mon)
    day = str(time_now.tm_mday)
    hour = str(time_now.tm_hour)
    min = str(time_now.tm_min)
    sec = str(time_now.tm_sec)
    lis = [year,mon,day,hour,min,sec]
    return '_'.join(lis)
time_now = get_time()
global write_dir
global step_count
step_count = 0
write_dir = './log/' + time_now
writer = SummaryWriter(log_dir=write_dir)

def train(args, 
          model, 
          train_dataloader,
          device,
          loss_f,
          val_dataloader):
    global step_count
    optimizer, scheduler = build_optim(args, train_dataloader, model.parameters())
    for epoch in range(args.epochs):
        pbar = tqdm(total=len(train_dataloader), ncols=120, desc='training')
        model.train()
        loss_sum = 0
        for step_i, data in enumerate(train_dataloader):
            for key, value in data.items():
                data[key] = value.to(device)
            output = model(data)
            # output[:,:,:3] = output[:,:,:3] / torch.norm(output[:,:,:3], dim=-1, keepdim=True)
            loss = loss_f(output, data['param'],data['mask'])
            optimizer.zero_grad()
            
            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm)
            
            optimizer.step()

            if scheduler:
                scheduler.step()
                if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = scheduler.get_last_lr()[0]
                else:
                    lr = scheduler.get_lr()[0]
            else:
                try:
                    lr = optimizer.get_lr()[0]
                except AttributeError:
                    lr = args.lr
            for param in model.parameters():
                param.grad = None
            
            loss_meter = loss.item()
            loss_sum += loss_meter
            desc_str = f'Epoch {epoch} | LR {lr:.10f}'
            desc_str += f' | Loss {loss_meter:4f}'
            pbar.set_description(desc_str)
            pbar.update(1)
        loss_sum = loss_sum / len(train_dataloader)
        pbar.close()
        torch.save(model, write_dir+'/BEST.ckpt')
        writer.add_scalar("train_loss",scalar_value=loss_sum,global_step = step_count)
        # if epoch % 5 == 0:
        loss_sum = 0
        with torch.no_grad():
            model.eval()
            pbar = tqdm(total=len(val_dataloader), ncols=120, desc='validation')
            for step_i, data in enumerate(val_dataloader):
                for key, value in data.items():
                    data[key] = value.to(device)
                output = model(data)
                loss = loss_f(output, data['param'],data['mask'])

                loss_meter = loss.item()
                desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                desc_str += f' | Loss {loss_meter:4f}'
                pbar.set_description(desc_str)
                pbar.update(1)
                loss_sum = loss_sum + loss_meter
        loss_sum = loss_sum / len(val_dataloader)
        writer.add_scalar("val_loss",scalar_value=loss_sum,global_step = step_count)
        step_count+=1
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h5_data = h5py.File('data/structured3d_plane_feature_roi.h5','r')
    model = Model(args).to(device)
    loss_f = Loss()
    train_dataloader = build_dataloader(args, h5_data, 'training')
    val_dataloader = build_dataloader(args, h5_data, 'validation')
    
    train(args,model,train_dataloader,device,loss_f,val_dataloader)
    model.de_frozen_layer()
    train(args,model,train_dataloader,device,loss_f,val_dataloader)


    

if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse()
    random.seed(2333)
    torch.manual_seed(2333)
    np.random.seed(2333)
    torch.cuda.manual_seed(2333)
    main(args)