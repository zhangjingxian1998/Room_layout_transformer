from models.detector import Detector
import random
import torch
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils.utils import parse

def main(args):
    if args.mode == 'train':
        checkpoint_callback = ModelCheckpoint(monitor='train_loss',mode='min',save_top_k=3,save_last=True,
                                              filename='../save_model/{epoch:02d}-{train_loss:.5f}',
                                              )
        model = Detector(args)
        trainer = pl.Trainer(
                    gpus=args.gpu_num,
                    check_val_every_n_epoch=1,
                    strategy='ddp',
                    sync_batchnorm = True,
                    max_epochs = args.epochs,
                    callbacks=[checkpoint_callback],
                    )
        trainer.fit(model)
        pass

    elif args.mode == 'test':
        pass

if __name__ == '__main__':
    args = parse()
    random.seed(2333)
    torch.manual_seed(2333)
    np.random.seed(2333)
    torch.cuda.manual_seed(2333)
    main(args)