import os.path as osp
import torch
import torch.nn.parallel
import tqdm
import numpy as np
import time
import laspy
from collections import defaultdict
from tree_learn.util import (checkpoint_save, init_train_logger, load_checkpoint, get_pointwise_preds, ensemble, get_instances,
                            is_multiple, get_args_and_cfg, build_cosine_scheduler, build_optimizer, assign_remaining_points_nearest_neighbor,
                            point_wise_loss, get_eval_res_components, get_segmentation_metrics, build_dataloader)
from tree_learn.model import TreeLearn
from tree_learn.dataset import TreeDataset
from tree_learn.util.pipeline import generate_tiles
from train import train, validate 

from TLSpecies.simpleview_pytorch import SimpleView
from TLSpecies.utils.discriminator_dataset import RealPredDataset
from TLSpecies.utils.train import train as train_simpleview
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
import torch.nn as nn
import multiprocessing as mp
import copy



TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1


def train_discriminator(model, train_loader, val_loader, params, device):
    if params['loss_fn']=="cross-entropy":
        loss_fn = nn.CrossEntropyLoss()
        print("Using cross-entropy loss...")
    if params['loss_fn']=="smooth-loss":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
        print("Using smooth-loss")

    if type(params['learning_rate']) == list:
        lr = params['learning_rate'][0]
        step_size = params['learning_rate'][1]
        gamma = params['learning_rate'][2]
    else:
        lr = params['learning_rate']

    if params['optimizer']=="sgd":
        print("Optimizing with SGD...")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=params['momentum'])
    elif params['optimizer']=="adam":
        print("Optimizing with AdaM...")
        optimizer = optim.Adam(model.parameters(), lr=lr)


    best_acc = 0

    for epoch in range(params['epoch']):  # loop over the dataset multiple times
        #Training loop============================================
        model.train()
        running_loss = 0.0
        print(train_loader)
        
        for i, data in enumerate(train_loader, 0):
            depth_images = data['depth_images']
            labels = data['labels']

            depth_images = depth_images.to(device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(depth_images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

        model.eval()  
        num_train_correct = 0
        num_train_samples = 0

        num_val_correct = 0
        num_val_samples = 0

        running_train_loss = 0
        running_val_loss = 0

        with torch.no_grad():
            for data in train_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)
                num_train_correct += (predictions == labels).sum()
                num_train_samples += predictions.size(0)

                running_train_loss += loss_fn(scores, labels)

            train_acc = float(num_train_correct)/float(num_train_samples)
            train_loss = running_train_loss/len(train_loader)

            print(f'OVERALL (Train): Got {num_train_correct} / {num_train_samples} with accuracy {train_acc*100:.2f}')

            #Val set eval===============
            all_labels = torch.tensor([]).to(device)
            all_predictions = torch.tensor([]).to(device)

            for data in val_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)

                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                num_val_correct += (predictions == labels).sum()
                num_val_samples += predictions.size(0)

                running_val_loss += loss_fn(scores, labels)

            val_acc = float(num_val_correct)/float(num_val_samples)
            val_loss = running_val_loss/len(val_loader)

            print(f'OVERALL (Val): Got {num_val_correct} / {num_val_samples} with accuracy {val_acc*100:.2f}')
                

            if val_acc >= best_acc:
                best_model_state = copy.deepcopy(model.state_dict())
                best_acc = val_acc

    print('Saving best (val) model...')
    print('Best overall accuracy: {}'.format(best_acc))
    torch.save(best_model_state,
               '{model_dir}/best_model.pth'.format(model_dir=params['model_dir'])
              )
    print('Saved!')

    #return best model
    return best_model_state    

def validate(config, epoch, model, val_loader, logger, writer):  
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, coords, instance_labels = [], [], [], [], [], []
        for batch in tqdm.tqdm(val_loader):

            # forward
            output = model(batch, return_loss=False)
            offset_prediction, semantic_prediction_logit = output['offset_predictions'], output['semantic_prediction_logits']

            batch['coords'] = batch['coords'] + batch['centers']
            semantic_prediction_logits.append(semantic_prediction_logit[batch['masks_sem']])
            semantic_labels.append(batch['semantic_labels'][batch['masks_sem']])
            offset_predictions.append(offset_prediction[batch['masks_sem']])
            offset_labels.append(batch['offset_labels'][batch['masks_sem']])
            coords.append(batch['coords'][batch['masks_sem']]), 
            instance_labels.append(batch['instance_labels'][batch['masks_sem']])

    semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0), torch.cat(semantic_labels, 0)
    offset_predictions, offset_labels = torch.cat(offset_predictions, 0), torch.cat(offset_labels, 0)
    coords, instance_labels = torch.cat(coords, 0), torch.cat(instance_labels).cpu().numpy()

    # split valset into 2 parts along y=0
    mask_y_greater_zero = coords[:, 1] > 0
    mask_y_not_greater_zero = torch.logical_not(mask_y_greater_zero)

    # pointwise eval y_greater_zero
    pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, 
                          config, epoch, writer, logger, 'full')
    
    pointwise_eval(semantic_prediction_logits[mask_y_greater_zero], offset_predictions[mask_y_greater_zero], semantic_labels[mask_y_greater_zero], offset_labels[mask_y_greater_zero], 
                          config, epoch, writer, logger, 'y_greater_zero')

    pointwise_eval(semantic_prediction_logits[mask_y_not_greater_zero], offset_predictions[mask_y_not_greater_zero], semantic_labels[mask_y_not_greater_zero], offset_labels[mask_y_not_greater_zero], 
                          config, epoch, writer, logger, 'y_not_greater_zero')


def pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, config, epoch, writer, logger, eval_name):
    _, offset_loss = point_wise_loss(semantic_prediction_logits.float(), offset_predictions[semantic_labels != NON_TREE_CLASS_IN_DATASET].float(), 
                                                    semantic_labels, offset_labels[semantic_labels != NON_TREE_CLASS_IN_DATASET])
    semantic_prediction_logits, semantic_labels = semantic_prediction_logits.cpu().numpy(), semantic_labels.cpu().numpy()
    
    tree_pred_mask = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)[:, TREE_CLASS_IN_DATASET] >= TREE_CONF_THRESHOLD
    tree_pred_mask = tree_pred_mask.numpy()
    tree_mask = semantic_labels == TREE_CLASS_IN_DATASET

    tp, fp, tn, fn = get_eval_res_components(tree_pred_mask, tree_mask)
    segmentation_res = get_segmentation_metrics(tp, fp, tn, fn)
    acc, prec, rec, f1, fdr, fnr, one_minus_f1, iou, fp_error_rate, fn_error_rate, error_rate = segmentation_res

    writer.add_scalar(f'{eval_name}/acc', acc if not np.isnan(acc) else 0, epoch)
    writer.add_scalar(f'{eval_name}/Offset_MAE', offset_loss, epoch)

    logger.info(f'[VALIDATION] [{epoch}/{config.epochs}] {eval_name}/semantic_acc {acc*100:.2f}, {eval_name}/offset_loss {offset_loss.item():.3f}')


def get_pseudo_labels(dataloader, model, discriminator, config, score_threshold=0.6):
    pointwise_results = get_pointwise_preds(model, dataloader, config.model)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats = pointwise_results
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, backbone_feats, input_feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, backbone_feats, input_feats = data
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET, NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS)
    # assign remaining points
    tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask], NOT_ASSIGNED_LABEL_IN_GROUPING)
    
    discriminator.eval()
    for tree_id in np.unique(instance_preds):
        score = discriminator(coords[instance_preds == tree_id])
        if score < score_threshold:
            instance_preds[instance_preds == tree_id] = NOT_ASSIGNED_LABEL_IN_GROUPING


    inferences = torch.tensor(inferences)
    inferences = inferences.cuda()
    for batch in tqdm.tqdm(inferences):
        with torch.no_grad():
            score = discriminator(batch)
            print("Score:", score)
            if score > score_threshold:
                pseudo_labeled_dataset.append(batch)
    
    pseudo_labeled_dataset = torch.stack(pseudo_labeled_dataset)

def generate_discriminator_dataset(las_path, model, config):
    ground_truth = []
    preds = []

    forest_point_cloud = laspy.read(las_path)

    coords = np.vstack((forest_point_cloud.x, forest_point_cloud.y, forest_point_cloud.z)).T
    instance_gt = forest_point_cloud.treeID
    print("Instance GT")
    print(np.unique(instance_gt))
    for tree_id in np.unique(instance_gt):
        ground_truth.append(coords[instance_gt == tree_id])


    generate_tiles(config.sample_generation, las_path)

    dataset = TreeDataset(**config.dataset_test)
    dataloader = build_dataloader(dataset, training=True, **config.dataloader.train_semi_sup)
    
    pointwise_results = get_pointwise_preds(model, dataloader, config.model)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats = pointwise_results
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, backbone_feats, input_feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, backbone_feats, input_feats = data
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET, NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS)
    # assign remaining points
    tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask], NOT_ASSIGNED_LABEL_IN_GROUPING)
    print("Instance Preds")
    print(np.unique(instance_preds))
    for tree_id in np.unique(instance_preds):
        preds.append(coords[instance_preds == tree_id])

    dataset = RealPredDataset(ground_truth, preds)
    
    return dataset

#def train_semi_supervised(config, epoch, model, optimizer, scheduler, scaler, pseudo_labeled_dataset, logger, writer):
    

def main():
    mp.set_start_method('spawn')
    torch.cuda.max_memory_allocated(1024*1024*1024*8)

    args, config = get_args_and_cfg()
    logger, writer = init_train_logger(config, args)

    # training objects
    model = TreeLearn(**config.model).cuda()

    discriminator = SimpleView(6, 2).cuda()
    
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_cosine_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    train_set = TreeDataset(**config.dataset_train, logger=logger)
    train_loader = build_dataloader(train_set, training=True, **config.dataloader.train)

    val_set = TreeDataset(**config.dataset_test, logger=logger)
    val_loader = build_dataloader(val_set, training=False, **config.dataloader.test)

    semi_supervised_set = TreeDataset(**config.dataset_train_semi_sup, logger=logger)
    semi_supervised_loader = build_dataloader(semi_supervised_set, training=True, **config.dataloader.train_semi_sup)

    # test_set = TreeDataset(**config.dataset_test, logger=logger)
    # test_loader = build_dataloader(test_set, training=False, **config.dataloader.test)
    
    # optionally pretrain or resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif config.pretrain:
        logger.info(f'Load pretrain from {config.pretrain}')
        load_checkpoint(config.pretrain, logger, model)

    # train and val
    # logger.info('Training')
    # for epoch in range(start_epoch, config.epochs + 1):
    #     train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer)
    #     if is_multiple(epoch, config.validation_frequency):
    #         optimizer.zero_grad()
    #         logger.info('Validation')
    #         validate(config, epoch, model, val_loader, logger, writer)
    #     writer.flush()
    torch.cuda.empty_cache()

    #build dataset for training discriminator
    discriminator_dataset = generate_discriminator_dataset(config.generator_data_path, model, config)
    total_size = len(discriminator_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_indices, val_indices = random_split(discriminator_dataset, [train_size, val_size])

# Create data loaders for train and validation sets
    train_dataloader = DataLoader(
        train_indices,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    val_dataloader = DataLoader(
        val_indices,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    # train discriminator model
    logger.info('Training discriminator')
    discriminator_params = {
        'epoch': 3,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'momentum': 0.9,
        'loss_fn': 'cross-entropy',
        'model_dir': 'models/discriminator',
    }
    discriminator = train_discriminator(discriminator, train_dataloader, val_dataloader, discriminator_params, 'cuda')


    # Inference on semi-supervised data, get quality scores
    pseudo_labeled_dataset = get_pseudo_labels(val_dataloader, model, discriminator, config)
    torch.cuda.empty_cache()

    # Train on pseudo labels
    logger.log('Training on pseudo labels')
    for epoch in range(start_epoch, config.epochs + 1):

        train_semi_supervised(config, epoch, model, optimizer, scheduler, scaler, pseudo_labeled_dataset, logger, writer)
        if is_multiple(epoch, config.validation_frequency):
            optimizer.zero_grad()
            logger.info('Validation')
            validate(config, epoch, model, val_loader, logger, writer)
        writer.flush()

if __name__ == '__main__':
    main()