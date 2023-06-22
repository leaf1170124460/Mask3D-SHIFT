import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from torch.cuda.amp.autocast_mode import autocast


@LOSSES.register_module()
class DepthLoss(nn.Module):

    def __init__(
            self,
            loss_weight=1.0,
            depth_act_mode='monodepth',
            si_weight=1.0,
            sq_rel_weight=1.0,
            abs_rel_weight=1.0,
            downsample_factor=8,
            dbound=(2.0, 58.0, 0.5)
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.depth_act_mode = depth_act_mode
        self.weight = torch.tensor([si_weight, sq_rel_weight, abs_rel_weight], dtype=torch.float32)
        self.downsample_factor = downsample_factor
        self.dbound = dbound
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])

    def forward(self,
                pred_list,
                target_list,
                mask_weight_list,
                reduction_override='mean',
                **kwargs):
        if self.weight is not None and not torch.any(self.weight > 0):
            return (pred_list * self.weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')

        loss = self.get_depth_loss(target_list, pred_list)
        return loss

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(
            -1, self.depth_channels)
        depth_preds = (depth_preds - depth_preds.min()) / (depth_preds.max() - depth_preds.min())
        # fliter out of range depth
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
            -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()
