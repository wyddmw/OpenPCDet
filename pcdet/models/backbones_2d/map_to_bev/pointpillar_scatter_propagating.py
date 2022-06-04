import torch
import torch.nn as nn
import torch.nn.functional as F
from ....ops.roiaware_pool3d import roiaware_pool3d_utils
from .prop_utils.propagating_module import get_propagatated_feature
from .prop_utils.voting_module import VotingModule
from .prop_utils.prop_module import DeformableSPP


class PointPillarScatterLOC(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        self.voxel_x, self.voxel_y = model_cfg.VOXEL_SIZE[:-1]
        self.x_range, self.y_range = self.model_cfg.POINT_CLOUD_RANGE[0:2]
        self.voting_module = VotingModule()
        self.deform_spp = DeformableSPP()
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_centering_offset = []
        batch_step = []
        #batch_prob = []
        batch_cls = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            centering_offset = torch.zeros(
                2, self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            #propa_prob = torch.zeros(
            #    1, self.nz * self.nx * self.ny,
            #    dtype=pillar_features.dtype,
            #    device=pillar_features.device)
            sparse_cls = torch.zeros(
                1, self.nz * self.nx * self.ny,
                dtype=torch.int,
                device=pillar_features.device) - 1

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            
            # get voxel point_map coordinates
            point_coords_x, point_coords_y = this_coords[:, 3].unsqueeze(dim=0) * self.voxel_x + self.x_range,\
                                              this_coords[:, 2].unsqueeze(dim=0) * self.voxel_y + self.y_range
            point_coords_z = torch.zeros(point_coords_x.shape, device=pillar_features.device)
            pillar_coords = torch.cat((point_coords_x[:, :, None], point_coords_y[:, :, None], point_coords_z[:, :, None]), dim=-1)
            gt_boxes = batch_dict['gt_boxes'][batch_idx, :, :-1]
            gt_boxes[:, 2] = 0
            gt_boxes = gt_boxes.unsqueeze(dim=0)

            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            pillars = pillars.unsqueeze(dim=0)
            
            #offset, prob = self.voting_module(pillars) 
            offset = self.voting_module(pillars)

            spatial_feature[:, indices] = pillars.squeeze(dim=0)
            centering_offset[:, indices] = offset
            #propa_prob[:, indices] = prob
            sparse_cls[:, indices] = roiaware_pool3d_utils.points_in_boxes_gpu(pillar_coords, gt_boxes)
            batch_spatial_features.append(spatial_feature)
            batch_centering_offset.append(centering_offset)
            #batch_prob.append(propa_prob)
            batch_cls.append(sparse_cls)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_centering_offset = torch.stack(batch_centering_offset, 0)
        #batch_prob = torch.stack(batch_prob, 0)
        batch_cls = torch.stack(batch_cls, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)     #
        batch_centering_offset = batch_centering_offset.view(batch_size, 2, self.ny, self.nx).permute(0, 2, 3, 1)
        #batch_prob = batch_prob.view(batch_size, 1, self.ny, self.nx)
        batch_cls = batch_cls.view(batch_size, self.ny, self.nx)
        #propagate_feature = self.deform_spp(batch_spatial_features, batch_centering_offset, batch_prob)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        #batch_dict['spatial_features'] = batch_spatial_features
        #batch_dict['centering_offset'] = batch_centering_offset
        #batch_dict['step'] = batch_step
        #batch_dict['prob'] = batch_prob
        #batch_dict['spatial_features'] = propagate_feature
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['centering_offset'] = batch_centering_offset
        batch_dict['batch_cls_index'] = batch_cls

        return batch_dict
