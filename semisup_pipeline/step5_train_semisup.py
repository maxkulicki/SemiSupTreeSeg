import os.path as osp
import torch
import torch.nn.parallel
import tqdm
import numpy as np
from collections import defaultdict
from tree_learn.util import (checkpoint_save, init_train_logger, load_checkpoint, get_pointwise_preds, ensemble, get_instances,
                            is_multiple, get_args_and_cfg, build_cosine_scheduler, build_optimizer, assign_remaining_points_nearest_neighbor,
                            point_wise_loss, get_eval_res_components, get_segmentation_metrics, build_dataloader)
from tree_learn.model import TreeLearn
from tree_learn.dataset import TreeDataset
from tree_learn.util.pipeline import generate_tiles
from tools.training.train import train, validate

from TLSpecies.simpleview_pytorch import SimpleView
from TLSpecies.utils.discriminator_dataset import RealPredDataset
from TLSpecies.utils.train import train as train_simpleview
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
import torch.nn as nn
import multiprocessing as mp



TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1


def main():
    mp.set_start_method('spawn')
    torch.cuda.max_memory_allocated(1024*1024*1024*12)

    args, config = get_args_and_cfg()
    logger, writer = init_train_logger(config, args)
    print(config)
    # training objects
    model = TreeLearn(**config.model).cuda()
    
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_cosine_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    dataset = TreeDataset(**config.dataset_train)
    train_data, val_data = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    print(len(train_data), len(val_data))
    train_data.collate_fn = dataset.collate_fn
    val_data.collate_fn = dataset.collate_fn

    train_loader = build_dataloader(train_data, training=True, **config.dataloader.train)
    val_loader = build_dataloader(val_data, training=False, **config.dataloader.test)

    logger.info(f'Load pretrain from {config.pretrain}')
    load_checkpoint(config.model_path, logger, model)

    logger.info('Training')
    for epoch in range(1, config.epochs + 1):
        train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer)
        if is_multiple(epoch, config.validation_frequency):
            optimizer.zero_grad()
            logger.info('Validation')
            validate(config, epoch, model, val_loader, logger, writer)
        writer.flush()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()