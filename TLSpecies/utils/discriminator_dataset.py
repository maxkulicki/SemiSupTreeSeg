from re import X
import os
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2
from .utils import get_depth_images_from_cloud, center_and_scale
from tqdm import tqdm
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class RealPredDataset(Dataset):
    """Dataset for tree species classification from
    point cloud depth projection images"""

    def __init__(self, gt, pred):
        self.species = ['Pred', 'Real']
        self.point_clouds = []
        self.labels = None
        self.file_names = []
        self.ids = []
        #self.meta_frame = pd.DataFrame(columns=meta_frame.columns) #A new dataframe that will only contain the subset of examples from the original that are found in the data directory
        
        self.image_dim = 256
        self.camera_fov_deg = 90
        self.f = 1
        self.camera_dist = 1.4
        self.transforms = ['none']
        
        self.min_rotation = 0
        self.max_rotation = 2*np.pi
        
        self.min_translation = 0
        self.max_translation = 0.93
        
        self.min_scale = 0.71
        self.max_scale = 1.51

        self.labels = torch.zeros(len(gt) + len(pred))

        for i, cloud in tqdm(enumerate(pred), total=len(pred)):
            cloud = center_and_scale(cloud) #Center and scale it
            self.point_clouds.append(torch.from_numpy(cloud))
            self.labels[i] = 0
            self.ids.append('pred' + str(i))

        for i, cloud in tqdm(enumerate(gt), total=len(gt)): 
            idx = i + len(pred)
            cloud = center_and_scale(cloud) #Center and scale it
            self.point_clouds.append(torch.from_numpy(cloud)) #Add the point cloud to the dataset
            self.labels[idx] = 1
            self.ids.append('real' + str(idx))
        
        self.labels = self.labels.long()

        print(self.ids)
        print(self.labels)

        return    
    
    def get_depth_image(self, i, transforms = None):
        if transforms is None:
            transforms = self.transforms
        
        points = self.point_clouds[i]    
            
        if 'rotation' in transforms:
            points = self.random_rotation(points, 
                                          min_rotation=self.min_rotation,
                                          max_rotation=self.max_rotation)
            
        if 'translation' in transforms:
            points = self.random_translation(points,
                                             min_translation=self.min_translation,
                                             max_translation=self.max_translation)
            
        if 'scaling' in transforms:
            points = self.random_scaling(points,
                                         min_scale=self.min_scale,
                                         max_scale=self.max_scale)
            
        
        return torch.unsqueeze(
               get_depth_images_from_cloud(points=points, 
                                                 image_dim=self.image_dim, 
                                                 camera_fov_deg=self.camera_fov_deg, 
                                                 f=self.f, 
                                                 camera_dist=self.camera_dist
                                                 )
                                    , 1)
    
    def set_params(self, 
                   image_dim = None,
                   camera_fov_deg = None,
                   f = None,
                   camera_dist = None,
                   transforms = None,
                   min_rotation = None,
                   max_rotation = None,
                   min_translation = None,
                   max_translation = None,
                   min_scale = None,
                   max_scale = None):
     
        if image_dim:    
            self.image_dim = image_dim        
        if camera_fov_deg:
            self.camera_fov_deg = camera_fov_deg 
        if f:
            self.f = f      
        if camera_dist:
            self.camera_dist = camera_dist      
        if transforms:
            self.transforms = transforms
        if min_rotation:
            self.min_rotation = min_rotation
        if max_rotation:
            self.max_rotation = max_rotation
        if min_translation:
            self.min_translation = min_translation
        if max_translation:
            self.max_translation = max_translation
        if min_scale:
            self.min_scale = min_scale
        if max_scale:
            self.max_scale = max_scale
            
        return
    
    def random_rotation(self,
                        point_cloud,
                        min_rotation=0,
                        max_rotation=2*torch.pi):
          
        theta = torch.rand(1)*(max_rotation - min_rotation) + min_rotation
        
        Rz = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0               ,                 0, 1],
        ]).double()
        
        return torch.matmul(point_cloud, Rz.t())
    
    def random_translation(self,
                           point_cloud,
                           min_translation = 0,
                           max_translation = 0.1):
        
        sign = torch.sign(torch.rand(1) - 0.5)
        tran = torch.rand(3)*(max_translation - min_translation) + min_translation
        
        return point_cloud + sign*tran
    
    def random_scaling(self,
                       point_cloud,
                       min_scale = 0.5,
                       max_scale = 1.5):
        
        
        scale = torch.rand(1)*(max_scale - min_scale) + min_scale
        
        return scale * point_cloud
    
    def __len__(self):
        assert len(self.labels) == len(self.point_clouds)
        #assert len(self.meta_frame) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            num_trees = len(idx)
        elif type(idx) == int:
            num_trees = 1
        else:
            num_trees = len(idx)

        depth_images = torch.zeros(size=(num_trees, 6, 1, self.image_dim, self.image_dim))
        
        if type(idx) == list:
            for i in range(len(idx)):
                depth_images[i] = self.get_depth_image(int(idx[i]))
        elif type(idx) == int:
            depth_images = self.get_depth_image(idx)
        
        labels = self.labels[idx]

        ids = self.ids[idx]
        
        sample = {'depth_images': depth_images, 'labels': labels, 'ids': ids}

        return sample    


    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #         num_trees = len(idx)
    #     else:
    #         num_trees = 1

    #     print('idx: ', idx)

    #     depth_images = torch.zeros(size=(num_trees, 6, 1, self.image_dim, self.image_dim))
        
    #     labels = self.labels[idx]
    #     ids = self.ids[idx]
        
    #     if type(idx) == list:
    #         for i in range(len(idx)):
    #             depth_images[i] = self.get_depth_image(int(idx[i]))
    #     elif type(idx) == int:
    #         depth_images = self.get_depth_image(idx)
        

    #     sample = {'depth_images': depth_images, 'labels': labels, 'ids': ids}

    #     return sample    