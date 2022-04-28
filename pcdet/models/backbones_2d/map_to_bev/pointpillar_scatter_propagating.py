import torch
import torch.nn as nn
from .prop_utils.voting_module import VotingModule
from .prop_utils.propagating_module import get_propagatated_feature

class PointPillarScatterLOC(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        self.voting_module = VotingModule()
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_centering_offset = []
        batch_step = []
        batch_prob = []
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
            propa_step = torch.zeros(
                1, self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            propa_prob = torch.zeros(
                1, self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t() 
            spatial_feature[:, indices] = pillars
            offset, step, prob = self.voting_module(pillars.unsqueeze(dim=0))
            centering_offset[:, indices] = offset
            propa_step[:, indices] = step
            propa_step[:, indices] = prob
            batch_spatial_features.append(spatial_feature)
            batch_centering_offset.append(centering_offset)
            batch_step.append(propa_step)
            batch_prob.append(propa_prob)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_centering_offset = torch.stack(batch_centering_offset, 0)
        batch_step = torch.stack(batch_step, 0)
        batch_prob = torch.stack(batch_prob, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)     #
        batch_centering_offset = batch_centering_offset.view(batch_size, 2, self.ny, self.nx).permute(0, 2, 3, 1)
        batch_step = batch_step.view(batch_size, 1, self.ny, self.nx).permute(0, 2, 3, 1)
        batch_prob = batch_prob.view(batch_size, 1, self.ny, self.nx)
        batch_spatial_features = get_propagatated_feature(batch_spatial_features, batch_centering_offset, batch_step, batch_prob)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        #batch_dict['spatial_features'] = batch_spatial_features
        #batch_dict['centering_offset'] = batch_centering_offset
        #batch_dict['step'] = batch_step
        #batch_dict['prob'] = batch_prob
        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict
