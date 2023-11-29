# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A model zoo for intermediate fusion.
# Please make sure your pairwise_t_matrix is normalized before using it.


import torch
from torch import nn
from icecream import ic
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.comm_modules.where2comm import Communication
from opencood.models.fuse_modules.where2comm_attn import TransformerFusion
import torch.nn.functional as F

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

def warp_feature(x, record_len, pairwise_t_matrix):
    _, C, H, W = x.shape
    B, L = pairwise_t_matrix.shape[:2]
    split_x = regroup(x, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
        # update each node i
        i = 0 # ego
        neighbor_feature = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W))
        out.append(neighbor_feature)

    out = torch.cat(out, dim=0)
    
    return out

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)
        
        return out

class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        xx : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        return out

class DiscoFusion(nn.Module):
    def __init__(self, feature_dims):
        super(DiscoFusion, self).__init__()
        from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
        self.pixel_weight_layer = PixelWeightLayer(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix):
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        out = []

        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            i = 0 # ego
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)

class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        from opencood.models.sub_modules.convgru import ConvGRU
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W'] # remember to modify for v2xsim dataset
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels] * num_gru_layers,
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)

    def forward(self, x, record_len, normalized_affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]

        split_x = regroup(x, record_len)
        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L,1,H,W)).to(x)
                roi_mask[b,i] = warp_affine_simple(one_tensor, normalized_affine_matrix[b][i, :, :, :],(H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times
        for l in range(self.num_iteration):

            batch_updated_node_features = []
            # iterate each batch
            for b in range(B):

                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = normalized_affine_matrix[b][:N, :N, :, :]

                updated_node_features = []

                # update each node i
                for i in range(N):
                    # (N,1,H,W)
                    mask = roi_mask[b, i, :N, ...]
                    neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                   t_matrix[i, :, :, :],
                                                   (H, W))

                    # (N,C,H,W)
                    ego_agent_feature = batch_node_features[b][i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    #(N,2C,H,W)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, ego_agent_feature], dim=1)
                    # (N,C,H,W)
                    # message contains all feature map from j to ego i.
                    message = self.msg_cnn(neighbor_feature) * mask

                    # (C,H,W)
                    if self.agg_operator=="avg":
                        agg_feature = torch.mean(message, dim=0)
                    elif self.agg_operator=="max":
                        agg_feature = torch.max(message, dim=0)[0]
                    else:
                        raise ValueError("agg_operator has wrong value")
                    # (2C, H, W)
                    cat_feature = torch.cat(
                        [batch_node_features[b][i, ...], agg_feature], dim=0)
                    # (C,H,W)
                    if self.gru_flag:
                        gru_out = \
                            self.conv_gru(cat_feature.unsqueeze(0).unsqueeze(0))[
                                0][
                                0].squeeze(0).squeeze(0)
                    else:
                        gru_out = batch_node_features[b][i, ...] + agg_feature
                    updated_node_features.append(gru_out.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features
        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,C,H,W) -> (B, H, W, C) -> (B,C,H,W)
        out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out

class V2XViTFusion(nn.Module):
    def __init__(self, args):
        super(V2XViTFusion, self).__init__()
        from opencood.models.sub_modules.v2xvit_basic import V2XTransformer
        self.fusion_net = V2XTransformer(args['transformer'])

    def forward(self, x, record_len, normalized_affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]

        regroup_feature, mask = Regroup(x, record_len, L)
        prior_encoding = \
            torch.zeros(len(record_len), L, 3, 1, 1).to(record_len.device)
        
        # prior encoding should include [velocity, time_delay, infra], but it is not supported by all basedataset.
        # it is possible to modify the xxx_basedataset.py and intermediatefusiondataset.py to retrieve these information
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])

        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature_new = []

        for b in range(B):
            ego = 0
            regroup_feature_new.append(warp_affine_simple(regroup_feature[b], normalized_affine_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion. In perfect setting, there is no delay. 
        # it is possible to modify the xxx_basedataset.py and intermediatefusiondataset.py to retrieve these information
        spatial_correction_matrix = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        
        return fused_feature

class When2commFusion(nn.Module):
    def __init__(self, args):
        super(When2commFusion, self).__init__()
        import numpy as np
        from opencood.models.fuse_modules.when2com_fuse import policy_net4, km_generator_v2, MIMOGeneralDotProductAttention, AdditiveAttentin

        self.in_channels = args['in_channels']
        self.feat_H = args['H']
        self.feat_W = args['W']
        self.query_size = args['query_size']
        self.key_size = args['key_size']
        

        self.query_key_net = policy_net4(self.in_channels)
        self.key_net = km_generator_v2(out_size=self.key_size)
        self.query_net = km_generator_v2(out_size=self.query_size)
        # self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)
        self.attention_net = AdditiveAttentin(self.key_size, self.query_size)

    def forward(self, x, record_len, normalized_affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        updated_node_features = []
        for b in range(B):

            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]

            # update each node i
            # (N,1,H,W)
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            query_key_maps = self.query_key_net(neighbor_feature)

            keys = self.key_net(query_key_maps).unsqueeze(0) # [N, C_k]
            query = self.query_net(query_key_maps[0].unsqueeze(0)).unsqueeze(0) # [1, C_q]

            neighbor_feature = neighbor_feature.unsqueeze(0) # [1, N, C, H, W]

            feat_fuse, prob_action = self.attention_net(query, keys, neighbor_feature, sparse=False)

            updated_node_features.append(feat_fuse)

        out = torch.cat(updated_node_features, dim=0)
        
        return out

