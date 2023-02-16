"""
Agent Selection Module for Heterogeneous Collaboration.

Maybe later can use data augment, one sample with different selection setting. 
"""
import numpy as np
import torch


class AgentSelector:
    def __init__(self, args, max_cav):
        self.lidar_ratio = args['lidar_ratio']
        self.ego_modality = args['ego_modality']  # 'random' / 'lidar'/ 'camera'
        self.max_cav = max_cav

        self.preset = None
        if "preset_file" in args:
            self.preset_file = args['preset_file'] # txt file
            self.preset = np.loadtxt(self.preset_file)


    def select_agent(self, i):
        """
        select agent to be equipped with LiDAR / Camera according to the strategy
        1 indicates lidar
        0 indicates camera
        """
        lidar_agent = np.random.choice(2, self.max_cav, p=[1 - self.lidar_ratio, self.lidar_ratio])

        if self.ego_modality == 'lidar':
            lidar_agent[0] = 1

        if self.ego_modality == 'camera':
            lidar_agent[0] = 0
        
        if self.preset:
            lidar_agent = self.preset[i]

        return lidar_agent, 1 - lidar_agent