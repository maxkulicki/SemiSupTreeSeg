import numpy as np
from dendromatics.voxel import voxelate
import laspy
from tqdm import tqdm
import numpy as np


import numpy as np

def voxel_grid_operations(grid1, origin1, grid2, origin2):
    # Convert origins and shapes to numpy arrays for easy manipulation
    origin1 = np.array(origin1)
    origin2 = np.array(origin2)
    shape1 = np.array(grid1.shape)
    shape2 = np.array(grid2.shape)

    # Calculate the bounding box of each grid
    min1 = origin1
    max1 = origin1 + shape1
    min2 = origin2
    max2 = origin2 + shape2

    # Determine the overlap region
    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)

    # Check if there is any overlap
    if np.any(overlap_max <= overlap_min):
        # No overlap
        return 0, 0, 0

    # Determine the common bounding box that includes both grids
    common_min = np.minimum(min1, min2)
    common_max = np.maximum(max1, max2)
    common_shape = common_max - common_min

    # Determine the necessary padding for each grid to align them within the common bounding box
    pad_before1 = min1 - common_min
    pad_after1 = common_max - max1

    pad_before2 = min2 - common_min
    pad_after2 = common_max - max2

    # Padding must be in the form ((pad_before_z, pad_after_z), (pad_before_y, pad_after_y), (pad_before_x, pad_after_x))
    pad1 = [(int(pad_before1[i]), int(pad_after1[i])) for i in range(3)]
    pad2 = [(int(pad_before2[i]), int(pad_after2[i])) for i in range(3)]

    # Pad the grids
    grid1_p = np.pad(grid1, pad1, mode='constant', constant_values=0)
    grid2_p = np.pad(grid2, pad2, mode='constant', constant_values=0)

    # After padding, ensure the shapes are identical
    final_shape = np.maximum(grid1_p.shape, grid2_p.shape)
    grid1_p = np.pad(grid1_p, [(0, max(0, final_shape[i] - grid1_p.shape[i])) for i in range(3)], mode='constant', constant_values=0)
    grid2_p = np.pad(grid2_p, [(0, max(0, final_shape[i] - grid2_p.shape[i])) for i in range(3)], mode='constant', constant_values=0)

    # Calculate intersection and union
    print(np.sum(grid1_p))
    print(np.sum(grid2_p))
    print("Calculating intersection and union")
    intersection = np.sum(np.logical_and(grid1_p, grid2_p))
    print(intersection)
    union = np.sum(np.logical_or(grid1_p, grid2_p))
    print(union)

    # Calculate IoU
    iou = intersection / union if union > 0 else 0

    return intersection, union, iou


def voxelize_point_cloud(points, resolution):
    # Find the minimum and maximum coordinates of the point cloud
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])

    # Calculate the size of the grid
    grid_size = np.ceil((max_x - min_x, max_y - min_y, max_z - min_z)) / resolution
    grid_size = np.clip(grid_size, 1, None).astype(int)

    # Calculate the origin of the grid
    grid_origin = np.array([min_x, min_y, min_z])

    # Create the 3D occupancy array
    occupancy_array = np.zeros(grid_size, dtype=bool)

    # Voxelize the point cloud
    for point in points:
        x_idx = np.clip((point[0] - grid_origin[0]) / resolution, 0, grid_size[0] - 1).astype(int)
        y_idx = np.clip((point[1] - grid_origin[1]) / resolution, 0, grid_size[1] - 1).astype(int)
        z_idx = np.clip((point[2] - grid_origin[2]) / resolution, 0, grid_size[2] - 1).astype(int)
        occupancy_array[x_idx, y_idx, z_idx] = True

    return occupancy_array, grid_origin

def find_closest_iou(grid, origin, grid_dict):

  highest_iou = 0.0
  closest_grid = None
  closest_name = None

  # Loop through each grid in the dictionary
  for name, (other_grid, other_origin) in grid_dict.items():
        # Calculate intersection, union, and IoU using the defined function
        intersection, union, iou = voxel_grid_operations(grid, origin, other_grid, other_origin)
        # Update closest match if current IoU is higher
        if iou > highest_iou:
            highest_iou = iou
            closest_grid = other_grid
            closest_name = name

  print(f"Name: {closest_name}, IoU: {highest_iou}")
  return closest_name, closest_grid, highest_iou

def voxelized_evaluation(gt_point_cloud, pred_point_cloud, resolution=0.1):
    """
    Evaluate the voxelized point cloud
    """
    # bring the point clouds to the same coordinate system by setting the minimum point to (0, 0, 0)
    print(np.min(gt_point_cloud.xyz, axis=0))
    gt_point_cloud.xyz = np.subtract(gt_point_cloud.xyz, np.min(gt_point_cloud.xyz, axis=0))
    # gt_point_cloud.X -= np.min(gt_point_cloud.X)
    # gt_point_cloud.Y -= np.min(gt_point_cloud.Y)
    # gt_point_cloud.Z -= np.min(gt_point_cloud.Z)
    pred_point_cloud.xyz -= np.min(pred_point_cloud.xyz, axis=0)
    # pred_point_cloud.X -= np.min(pred_point_cloud.X)
    # pred_point_cloud.Y -= np.min(pred_point_cloud.Y)
    # pred_point_cloud.Z -= np.min(pred_point_cloud.Z)

    metrics = {}
    ious = []

    pred_voxelized = {}
    print("Voxelizing prediction")
    for tree_id in tqdm(np.unique(pred_point_cloud.treeID)):
        if tree_id == 0:
            continue
        tree_point_cloud = pred_point_cloud[pred_point_cloud.treeID == tree_id]
        #voxelized_tree = voxelate(tree_point_cloud.xyz, resolution, resolution)
        voxelized_tree = voxelize_point_cloud(tree_point_cloud.xyz, resolution)
        pred_voxelized[tree_id] = voxelized_tree
        if tree_id > 20:
            break
        

    #gt_voxelized, _, gt_cloud_to_vox = voxelate(gt_point_cloud.xyz, resolution, resolution)
    #pred_voxelized, _, pred_cloud_to_vox = voxelate(pred_point_cloud.xyz, resolution, resolution)

    print("Evaluating trees")
    for tree_id in tqdm(np.unique(gt_point_cloud.treeID)):
        if tree_id == 0:
            continue
        tree_metrics = {}
        tree_point_cloud = gt_point_cloud[gt_point_cloud.treeID == tree_id]
        #gt_voxelized = voxelate(tree_point_cloud.xyz, resolution, resolution)
        gt_voxelized, origin = voxelize_point_cloud(tree_point_cloud.xyz, resolution)
        _,_,highest_iou = find_closest_iou(gt_voxelized, origin, pred_voxelized)        
        tree_metrics['iou'] = highest_iou
        if tree_id > 20:
            break
        metrics[tree_id] = tree_metrics
        ious.append(highest_iou)

    metrics['mean_iou'] = np.mean(ious)
    metrics['median_iou'] = np.median(ious)
    metrics['std_iou'] = np.std(ious)
    return metrics

if __name__ == '__main__':
    #gt_point_cloud = laspy.read('data/wytham_vox_full/input/wytham_vox0.1.laz')
    #pred_point_cloud = laspy.read('data/wytham_vox_full/wytham_vox0.laz')
    gt_point_cloud = laspy.read('data/tuwien_test/input/tuwien_test.las')
    pred_point_cloud = laspy.read('results/tuwien_test_out.laz')

    #pred_point_cloud.treeID = pred_point_cloud.PredInstance.astype(int)
    print(np.unique(pred_point_cloud.treeID))
    print("Start evaluation")
    metrics = voxelized_evaluation(gt_point_cloud, pred_point_cloud)
    print(metrics)