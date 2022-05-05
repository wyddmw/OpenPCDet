import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointPropagation(nn.Module):
    def __init__(self, feature_shape, num_offset=1):
        super().__init__()
        N, C, H, W = feature_shape
        self.num_offset = num_offset
        self.center_offset = nn.Conv2d(C, num_offset*2, 1, 1, bias=False)
        self.step = nn.Conv2d(C, num_offset*2, 1, 1, bias=False)
        self.propagation_weight = nn.Conv2d(C, 1, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        # center_offset, offset, weight = self.predict_step(feature)
        center_offset = self.center_offset(feature)
        step = F.relu(self.step(feature))
        propagation_weight = self.sigmoid(self.propagation_weight(feature))
        
        offset = torch.mul(center_offset, step).permute(0, 2, 3, 1)     # N H W num_coor
        feature_proped = get_propagatated_feature(feature, offset)
        feature_proped = feature * propagation_weight + feature_proped * (1 - propagation_weight)

        return feature_proped


def normalize_grid(grid_coordinate):
    # grid_coordinate [N, H, W, 2]
    H, W = grid_coordinate.shape[1], grid_coordinate.shape[2]
    grid_h = 2.0 * grid_coordinate[:, :, :, 0] / (H - 1) - 1.0
    grid_w = 2.0 * grid_coordinate[:, :, :, 1] / (W - 1) - 1.0
    return torch.stack((grid_h, grid_w), dim=-1)

def get_propagatated_feature(sparse_feature, offset, stride, prob):
    N, C, H, W = sparse_feature.shape
    grid_h = torch.arange(H, dtype=torch.float)
    grid_w = torch.arange(W, dtype=torch.float)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w)
    grid_coordinate = torch.stack((grid_h, grid_w), dim=-1).view(1, H, W, 2).repeat(N, 1, 1, 1).to(offset.device)
    target_coordinate = grid_coordinate.clone()
    sample_coordinate = grid_coordinate.clone()
    offset = offset * stride
    target_coordinate[:, :, :, 0] = torch.round(target_coordinate[:, :, :, 0] + offset[:, :, :, 0])
    target_coordinate[:, :, :, 1] = torch.round(target_coordinate[:, :, :, 1] + offset[:, :, :, 1])
    # deal with outliers
    target_coordinate[:, :, :, 0] = torch.clamp(target_coordinate[:, :, :, 0], max=H-1)
    target_coordinate[:, :, :, 1] = torch.clamp(target_coordinate[:, :, :, 1], max=W-1)
    # get target index
    target_coordinate = (target_coordinate[:, :, :, 0] * W + target_coordinate[:, :, :, 1]).view(N, -1)
    grid_coordinate = grid_coordinate.view(N, -1, 2)
    sample_coordinate = sample_coordinate.view(N, -1, 2)
    for i in range(N):
        sample_coordinate[i, target_coordinate[i].long()] = grid_coordinate[i]
    sample_coordinate = sample_coordinate.view(N, H, W, 2)
    sample_coordinate = normalize_grid(sample_coordinate)
    propagated_feature_sampled = F.grid_sample(sparse_feature, sample_coordinate, mode='bilinear', padding_mode='zeros', align_corners=True)

    propagated_feature_sampled = propagated_feature_sampled * prob + sparse_feature * (1 - prob)
    return propagated_feature_sampled

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def create_grid_canvas(t, dim=0):
    h, w = t.shape[-2:]
    grid = torch.stack(torch.meshgrid(
        torch.arange(h),
        torch.arange(w)), dim=dim)
    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid_canvas(grid):
    h, w = grid.shape[-2:]
    grid_h = 2.0 * grid[:, 0] / max(h-1, 1) - 1.0
    grid_w = 2.0 * grid[:, 1] / max(w-1, 1) - 1.0
    return torch.stack((grid_h, grid_w), dim=0)

def grid_test():
    inp = torch.ones(1, 1, 20, 20)
    # 目的是得到一个 长宽为20的tensor
    out_h = 20
    out_w = 20
    # grid的生成方式等价于用mesh_grid
    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
    grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
    grid = grid.unsqueeze(0)

    outp = F.grid_sample(inp, grid=grid, mode='bilinear')
    print(outp.shape)
    print(inp.shape)
    print(outp - inp)
    # print(outp.shape)  #torch.Size([1, 1, 20, 20])


if __name__ == '__main__':
    feature = torch.randn(1, 32, 128, 256)
    # canvas_transform(feature)
    # grid_test()
    # grid_canvas = create_grid_canvas(feature)
    # print(grid_canvas.shape)
    # grid_canvas = normalize_grid_canvas(grid_canvas)
    # print(grid_canvas.shape)
    pointpropagation = PointPropagation(feature.shape)
    output = pointpropagation.forward(feature)

