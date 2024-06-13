import os.path as osp
import torch
import torch.nn.parallel
import tqdm
import numpy as np
from tree_learn.util import (checkpoint_save, init_train_logger, load_checkpoint, get_pointwise_preds, ensemble, get_instances,
                            is_multiple, get_args_and_cfg, build_cosine_scheduler, build_optimizer, assign_remaining_points_nearest_neighbor,
                            point_wise_loss, get_eval_res_components, get_segmentation_metrics, build_dataloader, generate_tiles)
from tree_learn.model import TreeLearn
from tree_learn.dataset import TreeDataset
from TLSpecies.simpleview_pytorch import SimpleView
import os
from TLSpecies.utils import get_depth_images_from_cloud, center_and_scale

TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1

def get_pseudo_labels(dataloader, model, discriminator, config, score_threshold=0.6):
    pointwise_results = get_pointwise_preds(model, dataloader, config.model)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats = pointwise_results
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, backbone_feats, input_feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, backbone_feats, input_feats = data
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET, NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS)
    print(instance_preds)
    print(np.unique(instance_preds))
    print(np.sum(instance_preds == NOT_ASSIGNED_LABEL_IN_GROUPING))
    # assign remaining points
    tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask], NOT_ASSIGNED_LABEL_IN_GROUPING)
    # discriminator.eval()
    # with torch.no_grad():
    #     for tree_id in np.unique(instance_preds):
    #         point_cloud = coords[instance_preds == tree_id]
    #         point_cloud = center_and_scale(point_cloud)
    #         depth_images = get_depth_images_from_cloud(point_cloud).unsqueeze(1).unsqueeze(0).cuda()
    #         score = discriminator(depth_images).softmax(1)[0, 1].item()
    #         print(score)
            # if score < score_threshold:
            #     instance_preds[instance_preds == tree_id] = NOT_ASSIGNED_LABEL_IN_GROUPING

    #print the range of the point coordinates
    # print(coords.min(), coords.max(), coords.mean())
    # coords += np.random.normal(0, 0.1, coords.shape)


    pointwise_results = {
        'points': coords,
        # 'offset_predictions': offset_predictions,
        # 'offset_labels': offset_labels,
        # 'semantic_prediction_logits': semantic_prediction_logits,
        # 'semantic_labels': semantic_labels,
        # 'instance_labels': instance_labels,
        # 'backbone_feats': backbone_feats,
        'feat': input_feats,
        #'instance_label': instance_preds,
        'labels': instance_preds,
    }
    print(np.unique(instance_preds))

    result_path = os.path.join(config.semisup_dir, 'forest/wytham_crop.npz')
    print("Saving pseudo labels to: ", result_path)
    np.savez_compressed(result_path, **pointwise_results)
    
    generate_tiles(config.sample_generation, result_path)

    return pointwise_results

    

def main():
    args, config = get_args_and_cfg()

    model = TreeLearn(**config.model).cuda()
    discriminator = SimpleView(6, 2).cuda()

    model.load_state_dict(torch.load(config.model_path)['net'])
    discriminator.load_state_dict(torch.load(config.discriminator_path))
    
    dataset = TreeDataset(**config.dataset_semi_sup)
    dataloader = build_dataloader(dataset, training=True, **config.dataloader.train_semi_sup)
    # Inference on semi-supervised data, get quality scores
    pseudo_labeled_dataset = get_pseudo_labels(dataloader, model, discriminator, config)

if __name__ == '__main__':
    main()