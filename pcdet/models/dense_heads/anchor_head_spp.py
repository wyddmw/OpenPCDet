import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingleSPP(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.voxel_size_x = model_cfg['VOXEL_SIZE'][0]
        self.voxel_size_y = model_cfg['VOXEL_SIZE'][1]
        self.x_range = model_cfg['POINT_CLOUD_RANGE'][0]
        self.y_range = model_cfg['POINT_CLOUD_RANGE'][1]
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
    
    def assign_center_target(self, data_dict):
        center_offset_pred = data_dict['centering_offset']
        N, H, W, C = center_offset_pred.shape
        grid_H = self.grid_size[1]  # for pointpillar grid_H is 496
        grid_W = self.grid_size[0]  # for pointpillar grid_W is 432
        gt_center_coords = data_dict['gt_boxes'][:, :, 0:2]
        gt_center_grid_coords_w = (gt_center_coords[:, :, 0] - self.x_range) / self.voxel_size_x
        gt_center_grid_coords_h = (gt_center_coords[:, :, 1] - self.y_range) / self.voxel_size_y
        gt_center_grid = torch.stack((gt_center_grid_coords_h, gt_center_grid_coords_w), dim=-1)
        grid_coords = torch.stack(torch.meshgrid(
            torch.arange(grid_H),
            torch.arange(grid_W),
        ), dim=-1).to(center_offset_pred.device).unsqueeze(dim=0).repeat(N, 1, 1, 1)
        grid_coords = grid_coords.view(N, -1, 2) 
        target_center_offset = torch.zeros(center_offset_pred.shape, device=center_offset_pred.device).view(N, -1, C)       # N, num_grid, 2
        
        cls_index = data_dict['batch_cls_index'].reshape(N, -1).long()
        for i in range(N):
            cur_index = cls_index[i] > -1
            target_center_offset[i][cur_index] = gt_center_grid[i][cls_index[i][cur_index]] - grid_coords[i, cur_index]
        target_center_offset = target_center_offset.view(N, H, W, C)
        return target_center_offset

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        # get centering offset
        centering_offset_pred = self.forward_ret_dict['centering_offset_preds']
        centering_offset_target = self.forward_ret_dict['target_center_offset']
        N, H, W = centering_offset_pred.shape[:-1]
        valid_mask = (self.forward_ret_dict['valid_mask'] > -1).view(N, H, W, 1)
        centering_loss = F.smooth_l1_loss(centering_offset_pred*valid_mask, centering_offset_target*valid_mask, reduce=False).sum() / valid_mask.sum() * 0.1
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss + centering_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            target_center_offset = self.assign_center_target(data_dict)
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            self.forward_ret_dict['centering_offset_preds'] = data_dict['centering_offset']
            self.forward_ret_dict['valid_mask'] = data_dict['batch_cls_index']
            self.forward_ret_dict['target_center_offset'] = target_center_offset

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
