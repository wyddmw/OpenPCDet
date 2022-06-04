import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from pcdet.utils.visualize import draw_bev_gt, draw_bev_pts


class AnchorHeadSingleSPP(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.model_cfg = model_cfg
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
        gt_center_grid = gt_center_grid.int()
        #self.gt_center_grid = gt_center_grid.int()
        grid_coords = torch.stack(torch.meshgrid(
            torch.arange(grid_H),
            torch.arange(grid_W),
        ), dim=-1).to(center_offset_pred.device).unsqueeze(dim=0).repeat(N, 1, 1, 1).view(N, -1, 2).int()
        target_center_offset = torch.zeros(center_offset_pred.shape, device=center_offset_pred.device, dtype=torch.int32).view(N, -1, C)       # N, num_grid, 2
         
        cls_index = data_dict['batch_cls_index'].reshape(N, -1).long()
        #self.center_grid_cls = torch.zeros((N, H, W, 2), device=center_offset_pred.device).view(N, -1, 2).int()
        for i in range(N):
            cur_index = cls_index[i] > -1
            gt_cls_center = gt_center_grid[i][cls_index[i][cur_index]]
            #self.center_grid_cls[i][cur_index] = gt_cls_center
            target_center_offset[i][cur_index] = grid_coords[i][cur_index] - gt_cls_center

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
        centering_loss = F.smooth_l1_loss(centering_offset_pred, centering_offset_target.float()) * 10
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

        draw_bev = False
        if draw_bev:
            import cv2
            import os
            voxel_coords = data_dict['voxel_coords']
            batch_size = data_dict['batch_size']
            visual_dir = './visualize'
            for batch_id in range(batch_size):
                frame_id = data_dict['frame_id'][batch_id]
                frame_pts_path = os.path.join(visual_dir, 'pts_%s.png'%frame_id)
                frame_gt_path = os.path.join(visual_dir, 'gt_%s.png'%frame_id)
                gt_boxes = data_dict["gt_boxes"][batch_id].cpu().numpy() # (K, 7)
                voxel_coord = voxel_coords[voxel_coords[:,0]==batch_id][:,1:].cpu().numpy()[:, ::-1]
                draw_bev_pts(frame_pts_path, voxel_coord, gt_boxes, area_scope = [[0, 69.12], [-39.68, 39.68], [-3, 1]], cmap_color = False, voxel_size = self.model_cfg['VOXEL_SIZE'])
                draw_bev_gt(frame_gt_path, voxel_coord, gt_boxes, area_scope = [[0, 69.12], [-39.68, 39.68], [-3, 1]], cmap_color = False, voxel_size = self.model_cfg['VOXEL_SIZE'])

            draw_center = True
            if draw_center:
                center_dir = os.path.join(visual_dir, 'center_%s.png'%frame_id)
                inside_dir = os.path.join(visual_dir, 'inside_%s.png'%frame_id)

                N, H, W, C = target_center_offset.shape
                for frame_id in range(batch_size):
                    #inside_mask = data_dict['batch_cls_index'][frame_id] != -1
                    inside_mask = target_center_offset[frame_id].sum(-1) != 0
                    inside_mask = inside_mask.reshape(-1).data.cpu().numpy()
                    center_img = np.zeros([H, W, 3], dtype=np.uint8).reshape(-1, 3)
                    inside_img = np.zeros([H, W, 3], dtype=np.uint8).reshape(-1, 3)
                    #center_mask = self.gt_center_grid[frame_id, :, 0] * W + self.gt_center_grid[frame_id, :, 1]
                    center_mask = self.center_grid_cls[frame_id, :, 0] * W + self.center_grid_cls[frame_id, :, 1]
                    center_mask = center_mask.int().data.cpu().numpy() 
                    center_img[center_mask] = (255, 255, 0)
                    inside_img[center_mask] = (255, 255, 0)
                    inside_img[inside_mask] = (255, 255, 255)
                    center_img = center_img.reshape(H, W, 3)
                    inside_img = inside_img.reshape(H, W, 3)
                    cv2.imwrite(center_dir, center_img)
                    cv2.imwrite(inside_dir, inside_img)
                    
        return data_dict
