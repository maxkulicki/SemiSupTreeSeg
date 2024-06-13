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
from TLSpecies.utils.discriminator_dataset import RealPredDataset
import multiprocessing as mp
import random

TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1

def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

def save_data(data, save_format, save_name, save_folder, use_offset=True, quality_scores=None):
    if save_format == "las" or save_format == "laz":
        # get points and labels
        assert data.shape[1] == 4
        points = data[:, :3]
        labels = data[:, 3]
        classification = np.ones_like(labels)
        classification[labels == 0] = 2 # terrain according to For-Instance labeling convention (https://zenodo.org/records/8287792)
        classification[labels != 0] = 4 # stem according to For-Instance labeling convention (https://zenodo.org/records/8287792)

        # Create a new LAS file
        header = laspy.LasHeader(version="1.2", point_format=3)
        if use_offset:
            mean_x, mean_y, _ = points.mean(0)
            header.offsets = [mean_x, mean_y, 0]
        else:
            header.offsets = [0, 0, 0]
        
        points = points + header.offsets
        header.scales = [0.001, 0.001, 0.001]
        las = laspy.LasData(header)

        # Set the points and additional fields
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint32))
        las.treeID = labels
        las.classification = classification

        if quality_scores is not None:
            las.add_extra_dim(laspy.ExtraBytesParams(name="quality", type=np.float32))
            las.quality = quality_scores

        # Generate a color for each unique label
        unique_labels = np.unique(labels)
        color_map = {label: generate_random_color() for label in unique_labels}

        # Assign colors based on label
        colors = np.array([color_map[label] for label in labels], dtype=np.uint16)
        colors[classification == 2] = [0, 0, 0]

        # Set RGB colors in the LAS file
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

        # Write the LAS file to disk
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        las.write(save_path)
    elif save_format == "npy":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.save(save_path, data)
    elif save_format == "npz":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.savez_compressed(save_path, points=data[:, :3], labels=data[:, 3])
    elif save_format == "txt":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.savetxt(save_path, data)


def predict_instance_with_quality_scores(segmentation_model, discriminator_model, tile_path, config):
    #generate_tiles(config.sample_generation, las_path)

    dataset = TreeDataset(tile_path)
    dataloader = build_dataloader(dataset, training=True, **config.dataloader.train_semi_sup)
    
    pointwise_results = get_pointwise_preds(segmentation_model, dataloader, config.model)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats = pointwise_results
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, backbone_feats, input_feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, backbone_feats, input_feats = data
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET, NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS)
    
    coords_to_return = load_data(config.forest_path)[:, :3]

    preds_to_return = propagate_preds(coords, instance_preds, coords_to_return, n_neighbors=5)
    

    quality_scores = {}
    for treeID in tqdm(np.unique(preds_to_return)):
        if treeID == 0:
            continue
        tree_coords = coords_to_return[preds_to_return == treeID]
        tree_coords = center_and_scale(tree_coords)
        tree_preds = preds_to_return[preds_to_return == treeID]
        print("TreeID", treeID)
        print(tree_coords.shape, tree_preds.shape)
        depth_images = get_depth_images_from_cloud(tree_coords, image_dim=256)
        quality_score = torch.nn.functional.softmax(discriminator(depth_images.unsqueeze(0).unsqueeze(2)))[0][0].item()
        print(quality_score)
        quality_scores[treeID] = quality_score

    quality_per_point = np.zeros_like(preds_to_return, dtype=np.float64)
    for treeID in np.unique(preds_to_return):
        if treeID == 0:
            continue
        quality_per_point[preds_to_return == treeID] = quality_scores[treeID]

        
    # save
    logger.info(f'{plot_name}: #################### Saving ####################')
    full_dir = os.path.join(results_dir, 'full_forest')
    os.makedirs(full_dir, exist_ok=True)