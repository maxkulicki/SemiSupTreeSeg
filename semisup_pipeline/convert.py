#convert a point cloud from npz to pcd
import open3d as o3d
import numpy as np
import os

#convert to las with instance labels

def convert_npz_to_pcd(npz_path, pcd_path):
    data = np.load(npz_path)
    points = data['points']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud to {pcd_path}")

#get all npz files in a directory and convert them to pcd
def convert_all_npz_to_pcd(npz_dir, pcd_dir):
    for file in os.listdir(npz_dir):
        if file.endswith(".npz"):
            npz_path = os.path.join(npz_dir, file)
            pcd_path = os.path.join(pcd_dir, file.replace('.npz', '.pcd'))
            convert_npz_to_pcd(npz_path, pcd_path)

#convert_npz_to_pcd('TreeLearn/data/semisup/forest/pointwise_results.npz', 'TreeLearn/test.pcd')
convert_all_npz_to_pcd('data/train_semi_sup/tiles/npz', 'data/check')   