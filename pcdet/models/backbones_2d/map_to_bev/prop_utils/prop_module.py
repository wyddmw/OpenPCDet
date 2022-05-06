import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grid_coord(feature):
    H, W = feature.shape[-2:]
    grid = torch.stack(torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
    ), dim=-1).to(feature.device)
    grid.requires_grad = False
    grid = grid.type_as(feature)
    return grid

def normalize_grid(grid):
    # grid shape [N, H, W, 2]
    H, W = grid.shape[1], grid.shape[2]
    grid_h = 2.0 * grid[:, :, :, 0] / (H - 1) - 1.0 
    grid_w = 2.0 * grid[:, :, :, 1] / (W - 1) - 1.0 

    return torch.stack((grid_h, grid_w), dim=-1)

class DeformableSPP(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, feature, offset, weight):
        N, C, H, W = feature.shape
        # offset shape is {N, H, W, 2}
        # this is a hard code here, the stride is set as 0.5 for first version
        stride = torch.ones(N, H, W).to(feature.device) * 0.5 
        grid_coordinate = get_grid_coord(feature).unsqueeze(dim=0).repeat(N, 1, 1, 1)

        target_coordinate = get_grid_coord(feature).unsqueeze(dim=0).repeat(N, 1, 1, 1)
        sample_coordinate = get_grid_coord(feature).unsqueeze(dim=0).repeat(N, 1, 1, 1)
        #target_coordinate[:, :, :, 0] = target_coordinate[:, :, :, 0] + (offset[:, :, :, 0] * stride).round()
        #target_coordinate[:, :, :, 1] = target_coordinate[:, :, :, 1] + (offset[:, :, :, 1] * stride).round()
        target_coordinate[:, :, :, 0] = target_coordinate[:, :, :, 0] + (offset[:, :, :, 0] * stride).int()
        target_coordinate[:, :, :, 1] = target_coordinate[:, :, :, 1] + (offset[:, :, :, 1] * stride).int()

        target_coordinate[:, :, :, 0] = torch.clamp(target_coordinate[:, :, :, 0].clone(), max=H-1, min=0)
        target_coordinate[:, :, :, 1] = torch.clamp(target_coordinate[:, :, :, 1].clone(), max=W-1, min=0)
        target_coordinate = (target_coordinate[:, :, :, 0] * W + target_coordinate[:, :, :, 1]).view(N, -1) 
        grid_coordinate = grid_coordinate.view(N, -1, 2)
        sample_coordinate = sample_coordinate.view(N, -1, 2)
        for i in range(N):
            sample_coordinate[i, target_coordinate[i].long()] = grid_coordinate[i]
        sample_coordinate = sample_coordinate.view(N, H, W, 2)
        sample_coordinate = normalize_grid(sample_coordinate)
        propagated_feature_sampled = F.grid_sample(feature, sample_coordinate, mode='bilinear', padding_mode='zeros', align_corners=True)
        propagated_feature_sampled = feature * (1 - weight) + propagated_feature_sampled * weight 
        return propagated_feature_sampled

