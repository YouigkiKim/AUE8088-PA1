"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python test.py --ckpt_file wandb/aue8088-pa1/ygeiua2t/checkpoints/epoch\=19-step\=62500.ckpt
"""
# Python packages
import argparse

# PyTorch & Pytorch Lightning
from lightning import Trainer
from torch.utils.flop_counter import FlopCounterMode
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg
from src.network import MyNetwork, MyNetwork2, MyNetwork3, MyNetwork4, MyNetwork5, MyNetwork6
torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--ckpt_file',
        type = str,
        help = 'Model checkpoint file name')
    args = args.parse_args()
    model_name_arg = args.ckpt_file.split('/')[-1].split('.')[0]
    if model_name_arg == 'mynetwork_1':
        model_name_arg = 'MyNetwork'
    elif model_name_arg == 'mynetwork_2':
        model_name_arg = 'MyNetwork2'
    elif model_name_arg == 'mynetwork_3':
        model_name_arg = 'MyNetwork3'
    elif model_name_arg == 'mynetwork_4':
        model_name_arg = 'MyNetwork4'
    elif model_name_arg == 'mynetwork_5':
        model_name_arg = 'MyNetwork5'
  
    model = SimpleClassifier(
        model_name = model_name_arg,
        num_classes = cfg.NUM_CLASSES,
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = 512,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        benchmark = True,
        inference_mode = True,
        logger = False,
    )

    trainer.validate(model, ckpt_path = args.ckpt_file, datamodule = datamodule)

    # FLOP counter
    x, y = next(iter(datamodule.test_dataloader()))
    flop_counter = FlopCounterMode(model, depth=1)

    with flop_counter:
        model(x)
