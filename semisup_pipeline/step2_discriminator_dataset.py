import os.path as osp
import os
import torch
import torch.nn.parallel
import tqdm
import numpy as np
import time
import laspy
from collections import defaultdict
from tree_learn.util import (checkpoint_save, init_train_logger, load_checkpoint, get_pointwise_preds, ensemble, get_instances,
                            is_multiple, get_args_and_cfg, build_cosine_scheduler, build_optimizer, assign_remaining_points_nearest_neighbor,
                            point_wise_loss, get_eval_res_components, get_segmentation_metrics, build_dataloader, load_data, save_data,
                            propagate_preds)

from tree_learn.model import TreeLearn
from tree_learn.dataset import TreeDataset
from tree_learn.util.pipeline import generate_tiles
from TLSpecies.utils.discriminator_dataset import RealPredDataset
import multiprocessing as mp
import re

TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1

def center_and_scale(point_cloud):
    centered_cloud = point_cloud-np.mean(point_cloud, axis=0)
    return centered_cloud/np.max(abs(centered_cloud)) * 100

def find_largest_index(folder_path, prefix='Real'):
    # Define the regular expression pattern to match the filenames
    pattern = re.compile(rf'{prefix}_(\d+)\.las')
    max_index = -1
    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    return max_index


def generate_discriminator_dataset(las_path, model, config, save_las_files=False):
    ground_truth = []
    preds = []


    tiles = True

    if tiles:
        for path in os.listdir(las_path):
            if path.endswith('.las') or path.endswith('.laz'):
                print(f"Generating Tiles {path}")
                file_path = osp.join(las_path, path)
                generate_tiles(config.sample_generation, file_path)
        
    dataset = TreeDataset(**config.dataset_test)
    dataloader = build_dataloader(dataset, training=True, **config.dataloader.train_semi_sup)
    print(os.listdir(las_path))

    for path in os.listdir(las_path):
        if path.endswith('.las') or path.endswith('.laz'):
            file_path = osp.join(las_path, path)

            forest_point_cloud = laspy.read(file_path)

            coords = np.vstack((forest_point_cloud.x, forest_point_cloud.y, forest_point_cloud.z)).T
            instance_gt = forest_point_cloud.treeID
            print("Instance GT")
            print(np.unique(instance_gt))
            for tree_id in np.unique(instance_gt):
                if tree_id > 1000:
                    continue
                ground_truth.append(coords[instance_gt == tree_id])
            # tiles = False 
            # if tiles:
            #     print("Generating tiles")
            #     generate_tiles(config.sample_generation, las_path)

    print("Getting pointwise predictions")
    pointwise_results = get_pointwise_preds(model, dataloader, config.model)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats = pointwise_results
    print("Ensembling")
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, backbone_feats, input_feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, backbone_feats, input_feats = data
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET, NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS)
    # assign remaining points
    tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask], NOT_ASSIGNED_LABEL_IN_GROUPING)

    if config.save_full_result:
        print("Saving full results")
        plot_name = os.path.basename(file_path)[:-4]

        propagate=False
        if propagate:
            coords_to_return = load_data(file_path)[:, :3]
            preds_to_return = propagate_preds(coords, instance_preds, coords_to_return, n_neighbors=5)
        else:
            coords_to_return = coords
            preds_to_return = instance_preds
        pointwise_results = {
            'coords': coords,
            'offset_predictions': offset_predictions,
            'offset_labels': offset_labels,
            'semantic_prediction_logits': semantic_prediction_logits,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'backbone_feats': backbone_feats,
            'input_feats': input_feats,
            'instance_preds': instance_preds,
        }
        save_format = 'laz'
        save_dir = "results"
        save_data(np.hstack([coords_to_return, preds_to_return.reshape(-1, 1)]), save_format, plot_name, save_dir)

    print("Instance Preds")
    print(np.unique(instance_preds))
    exit()
    for tree_id in np.unique(instance_preds):
        if tree_id == 0:
            continue
        preds.append(coords[instance_preds == tree_id])

    if save_las_files:
        gt_path = osp.join(config.discriminator_dataset_path, 'Real')
        gt_largest_index = find_largest_index(gt_path, prefix='Real')
        pred_path = osp.join(config.discriminator_dataset_path, 'Pred')
        pred_largest_index = find_largest_index(pred_path, prefix='Pred')
        for i, tree in enumerate(ground_truth):
            tree = center_and_scale(tree)
            gt_tree = laspy.create(point_format=forest_point_cloud.header.point_format, file_version=forest_point_cloud.header.version)
            gt_tree.x = tree[:, 0]
            gt_tree.y = tree[:, 1]
            gt_tree.z = tree[:, 2]
            gt_tree.treeID = np.ones(tree.shape[0]) * i
            gt_tree.write(osp.join(gt_path, f'Real_{i+gt_largest_index}.las'))
            #gt_save_path = osp.join(config.discriminator_dataset_path, 'Real')
            #gt_tree.write(f'{gt_save_path}/Real_{i}.las')
        for i, tree in enumerate(preds):
            tree = center_and_scale(tree)
            pred_tree = laspy.create(point_format=forest_point_cloud.header.point_format, file_version=forest_point_cloud.header.version)
            pred_tree.x = tree[:, 0]
            pred_tree.y = tree[:, 1]
            pred_tree.z = tree[:, 2]
            pred_tree.treeID = np.ones(tree.shape[0]) * i
            pred_tree.write(osp.join(pred_path, f'Pred_{i+pred_largest_index}.las'))
            #pred_tree.write(f'data/wytham_gt_pred/Pred/Pred_{i}.las')

    dataset = RealPredDataset(ground_truth, preds)

    return dataset
    

def main():
    mp.set_start_method('spawn')
    torch.cuda.max_memory_allocated(1024*1024*1024*8)
    args, config = get_args_and_cfg()
    # training objects
    model = TreeLearn(**config.model).cuda()
    model.load_state_dict(torch.load(config.model_path)['net'])

    #build dataset for training discriminator
    discriminator_dataset = generate_discriminator_dataset(config.generator_data_path, model, config, save_las_files=True)
    #save discriminator dataset
    save_path = osp.join(config.discriminator_dataset_path, 'RealPredDataset.pth')
    torch.save(discriminator_dataset, save_path)

if __name__ == '__main__':
    main()