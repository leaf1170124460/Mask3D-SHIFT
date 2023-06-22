import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.runner import force_fp32
from mmdet.core import (multi_apply, reduce_mean)
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


@HEADS.register_module()
class Detr3DHead_Seg_Depth(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,

                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 with_stuff=False,
                 num_stuff_classes=2,
                 vis_stuff_classes_orders=[10],
                 num_cams=6,
                 num_layers=6,
                 num_cls_fcs=2,
                 mask_post_assign=False,
                 mask_assign_stages=5,
                 seg_stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 code_weights=None,
                 train_cfg=None,
                 with_query_pos=False,
                 loss_ce=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_dice=dict(
                     type='DiceLoss', loss_weight=4.0),
                 loss_depth=dict(
                     type='DepthLoss',
                     loss_weight=5.0,
                     depth_act_mode='sigmoid',
                     si_weight=1.0,
                     sq_rel_weight=1.0,
                     abs_rel_weight=1.0,
                 ),
                 num_channels=256,
                 num_mask_fcs=1,
                 mask_act_cfg=dict(type='ReLU', inplace=True),
                 num_depth_fcs=1,
                 depth_act_cfg=dict(type='ReLU', inplace=True),
                 dbound=(2.0, 58.0, 0.5),
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        if train_cfg:
            if 'refine_assigner' in train_cfg:
                self.train_cfg = train_cfg.copy()
                self.train_cfg['assigner'] = self.train_cfg['refine_assigner']
            else:
                self.train_cfg = train_cfg
        else:
            self.train_cfg = train_cfg
        super(Detr3DHead_Seg_Depth, self).__init__(
            *args, transformer=transformer, train_cfg=self.train_cfg, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self.num_layers = num_layers

        self.mask_post_assign = mask_post_assign
        self.mask_assign_stages = mask_assign_stages
        self.seg_stage_loss_weights = seg_stage_loss_weights

        self.loss_ce = build_loss(loss_ce)
        self.loss_dice = build_loss(loss_dice)

        # depth loss
        self.loss_depth = build_loss(loss_depth)

        # mask mlp
        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(num_channels, num_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), num_channels)[1])
            self.mask_fcs.append(build_activation_layer(mask_act_cfg))

        self.fc_mask = nn.Linear(num_channels, num_channels)

        # depth mlp
        self.depth_fcs = nn.ModuleList()
        for _ in range(num_depth_fcs):
            self.depth_fcs.append(
                nn.Linear(num_channels, num_channels, bias=False))
            self.depth_fcs.append(
                build_norm_layer(dict(type='LN'), num_channels)[1])
            self.depth_fcs.append(build_activation_layer(depth_act_cfg))

        # depth
        self.dbound = dbound
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])

        self.fc_depth = nn.Linear(num_channels, num_channels)

        self.conv_depth = nn.Conv2d(num_channels, self.depth_channels, kernel_size=1, stride=1)

        self.num_cams = num_cams
        self.with_stuff = with_stuff
        self.num_stuff_classes = num_stuff_classes
        self.vis_stuff_classes_orders = vis_stuff_classes_orders
        self.with_query_pos = with_query_pos

        if self.with_stuff:
            self.stuff_query_embedding = nn.Embedding(num_stuff_classes * num_cams,
                                                      self.embed_dims * 2)
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, seg_feats, img_metas, queries=None, references=None, query_pos=None, bboxes=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs = len(img_metas)
        query_embeds = self.query_embedding.weight
        if self.with_stuff:
            stuff_query_embed = self.stuff_query_embedding.weight
            query_embeds = torch.cat([query_embeds, stuff_query_embed], dim=0)

        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_masks_stuff = None

        outputs_classes = []
        outputs_coords = []
        outputs_masks = []

        outputs_depths = []

        queries2img_masks = [self.get_queries2img_mask(inter_references[i], img_metas) for i in
                             range(len(inter_references))]
        queries2img_masks = torch.stack(queries2img_masks)
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_masks.append(self.forward_seg(hs[lvl], seg_feats))
            outputs_depths.append(self.forward_depth(hs[lvl], seg_feats))
        outputs_masks = torch.stack(outputs_masks)
        outputs_depths = torch.stack(outputs_depths)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.with_stuff:
            outputs_coords = outputs_coords[:, :, :self.num_query]
            outputs_masks_stuff = outputs_masks[:, :, :, self.num_query:]
            outputs_masks = outputs_masks[:, :, :, :self.num_query]
            hs = hs[:, :, :self.num_query]
            init_reference = init_reference[:, :self.num_query]
            inter_references = inter_references[:, :, :self.num_query]
            outputs_classes = outputs_classes[:, :, :self.num_query]
            queries2img_masks = queries2img_masks[:, :, :, :self.num_query]
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_mask_preds': outputs_masks,
            'all_depth_preds': outputs_depths,
            'queries2img_masks': queries2img_masks,
            'all_mask_preds_stuff': outputs_masks_stuff,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def forward_seg(self, queries, seg_feats):
        mask_feat = queries
        for mask_fc in self.mask_fcs:
            mask_feat = mask_fc(mask_feat)
        mask_feat = self.fc_mask(mask_feat)
        mask_pred = torch.einsum('bnc,bvchw->bvnhw', mask_feat, seg_feats)
        return mask_pred

    # TODO: torch.einsum
    def forward_depth(self, queries, seg_feats):
        depth_feat = queries
        for depth_fc in self.depth_fcs:
            depth_feat = depth_fc(depth_feat)
        depth_feat = self.fc_depth(depth_feat)
        # v: camera nums
        depth_pred = torch.einsum('bnc,bvchw->bvchw', depth_feat, seg_feats)
        # 
        b, v, c, h, w = depth_pred.shape
        depth_pred = depth_pred.view(-1, c, h, w)
        depth_pred = self.conv_depth(depth_pred)
        depth_pred = depth_pred.view(b, v, -1, h, w)
        return depth_pred

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           mask_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_masks,
                           gt_mask_labels,
                           gt_assigned_masks,
                           queries2img_masks,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        device = gt_bboxes.device
        num_gts = gt_bboxes.size(0)
        H, W = gt_masks[0].shape[1:]
        # TODO: num_cams multi camera index match
        gt_masks_for_cost = torch.zeros([self.num_cams, num_gts, H, W], device=device)
        mask_weights = torch.zeros([self.num_cams, num_gts], device=device)
        for i in range(num_gts):
            for vi in range(self.num_cams):
                if gt_assigned_masks[i, vi] != -1:
                    gt_masks_for_cost[vi, i] = gt_masks[vi][gt_assigned_masks[i, vi]]
                    mask_weights[vi, i] = 1

        assign_result = self.assigner.assign(bbox_pred, cls_score, mask_pred, gt_bboxes,
                                             gt_labels, gt_masks_for_cost, mask_weights, gt_mask_labels,
                                             gt_assigned_masks, queries2img_masks, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # velocity weigth to zero
        bbox_weights[:, 7:9] = 0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        mask_targets = torch.zeros_like(mask_pred)
        mask_targets[:, pos_inds] = gt_masks_for_cost[:, sampling_result.pos_assigned_gt_inds]
        mask_targets_weights = torch.zeros([self.num_cams, num_bboxes], device=device)
        mask_targets_weights[:, pos_inds] = mask_weights[:, sampling_result.pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, mask_targets, mask_targets_weights)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    mask_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_mask_labels_list,
                    gt_assigned_masks_list,
                    queries2img_masks_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list, mask_targets_list, mask_targets_weights_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, mask_preds_list,
            gt_labels_list, gt_bboxes_list, gt_masks_list, gt_mask_labels_list, gt_assigned_masks_list,
            queries2img_masks_list,
            gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, mask_targets_list, mask_targets_weights_list)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    mask_preds,
                    mask_preds_stuff,
                    depth_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_mask_labels_list,
                    gt_assigned_masks_list,
                    gt_masks_stuff,
                    gt_mask_labels_stuff,
                    queries2img_masks,
                    gt_depth_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        queries2img_masks_list = [queries2img_masks[i] for i in range(num_imgs)]

        gt_assigned_masks_list = [gt_assigned_masks_list[i] for i in range(num_imgs)]

        # depth_preds_list = [depth_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, mask_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_masks_list, gt_mask_labels_list,
                                           gt_assigned_masks_list, queries2img_masks_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, mask_targets_list, mask_targets_weights_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        mask_targets = torch.cat(mask_targets_list, 1)
        mask_weights = torch.cat(mask_targets_weights_list, 1)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        mask_preds = mask_preds.reshape(
            [mask_preds.shape[1], mask_preds.shape[0] * mask_preds.shape[2], mask_preds.shape[3], mask_preds.shape[4]])

        pos_mask_preds = mask_preds[mask_weights.bool()]
        pos_mask_targets = mask_targets[mask_weights.bool()]
        num_pos_masks = pos_mask_preds.shape[0]

        if self.with_stuff:
            pos_mask_preds_stuff = []
            pos_mask_targets_stuff = []
            for bi in range(len(gt_masks_stuff)):
                for vi in range(self.num_cams):
                    labels = gt_mask_labels_stuff[bi][vi]
                    masks = gt_masks_stuff[bi][vi][labels < self.num_classes + self.num_stuff_classes]
                    labels = labels[labels < self.num_classes + self.num_stuff_classes]
                    ids = (labels - self.num_classes) * self.num_cams + vi
                    pos_mask_preds_stuff.append(mask_preds_stuff[bi, vi, ids])
                    pos_mask_targets_stuff.append(masks)
            pos_mask_preds_stuff = torch.cat(pos_mask_preds_stuff, dim=0)
            pos_mask_targets_stuff = torch.cat(pos_mask_targets_stuff, dim=0)
            pos_mask_preds = torch.cat([pos_mask_preds, pos_mask_preds_stuff], dim=0)
            pos_mask_targets = torch.cat([pos_mask_targets, pos_mask_targets_stuff], dim=0)

        loss_ce = self.loss_ce(pos_mask_preds, pos_mask_targets)

        loss_dice = self.loss_dice(pos_mask_preds, pos_mask_targets)

        # loss_depth = self.loss_depth(
        #     pred_list=depth_preds_list,
        #     target_list=gt_depth_list,
        #     mask_weight_list=[(gt_depth > 0.).float() for gt_depth in gt_depth_list]
        # )

        gt_depth = torch.stack(gt_depth_list)
        # depth_preds = F.interpolate(depth_preds, gt_depth.shape[2:], mode='bilinear')

        loss_depth = self.loss_depth(
            pred_list=depth_preds,
            target_list=gt_depth,
            mask_weight_list=(gt_depth > 0.).float()
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_ce = torch.nan_to_num(loss_ce)
        loss_dice = torch.nan_to_num(loss_dice)

        loss_depth = torch.nan_to_num(loss_depth)
        return loss_cls, loss_bbox, loss_ce, loss_dice, loss_depth

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_masks_list,
             gt_mask_labels_list,
             gt_assigned_masks_list,
             gt_masks_stuff,
             gt_mask_labels_stuff,
             gt_depth_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        all_mask_preds = preds_dicts['all_mask_preds']
        num_dec_layers = len(all_cls_scores)
        all_mask_preds_stuff = [None for _ in range(num_dec_layers)]
        if self.with_stuff:
            all_mask_preds_stuff = preds_dicts['all_mask_preds_stuff']
        queries2img_masks = preds_dicts['queries2img_masks']

        all_depth_preds = preds_dicts['all_depth_preds']

        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_mask_labels_list = [gt_mask_labels_list for _ in range(num_dec_layers)]
        all_gt_assigned_masks = [gt_assigned_masks_list for _ in range(num_dec_layers)]
        all_gt_masks_stuff_list = [gt_masks_stuff for _ in range(num_dec_layers)]
        all_gt_mask_labels_stuff_list = [gt_mask_labels_stuff for _ in range(num_dec_layers)]

        all_gt_depth_list = [gt_depth_list for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_ce, losses_dice, losses_depth = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_mask_preds, all_mask_preds_stuff,
            all_depth_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list, all_gt_mask_labels_list, all_gt_assigned_masks,
            all_gt_masks_stuff_list, all_gt_mask_labels_stuff_list, queries2img_masks,
            all_gt_depth_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox, enc_pos_inds_list = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_mask_ce'] = losses_ce[-1]
        loss_dict['loss_mask_dice'] = losses_dice[-1]

        loss_dict['loss_depth'] = losses_depth[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_ce_i, loss_dice_i, loss_depth_i in zip(losses_cls[:-1],
                                                                                 losses_bbox[:-1], losses_ce[:-1],
                                                                                 losses_dice[:-1], losses_depth[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_mask_ce'] = loss_ce_i
            loss_dict[f'd{num_dec_layer}.loss_mask_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_depth'] = loss_depth_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

    # def get_seg_masks(self,masks_per_img,labels_per_img,scores_per_img,img_meta):
    #     H,W=img_meta['ori_shape'][:2]
    #     masks_per_img = F.interpolate(
    #         masks_per_img.unsqueeze(0).sigmoid(),
    #         size=img_meta['img_shape'][0][:2],
    #         mode='bilinear',
    #         align_corners=False).squeeze(0)
    #     seg_masks=masks_per_img[:,14:H+14,:W]
    #     seg_masks=seg_masks>0.5
    #     num_classes=self.num_classes
    #     segm_result=[[] for _ in range(num_classes)]
    #     mask_preds=seg_masks.cpu().numpy()
    #     mask_labels=labels_per_img.cpu().numpy()
    #     num_ins=mask_preds.shape[0]
    #     cls_scores=scores_per_img.sigmoid().cpu().numpy()
    #     for idx in range(num_ins):
    #         if cls_scores[idx]>0.3:
    #             segm_result[mask_labels[idx]].append(mask_preds[idx])

    #     return segm_result

    # def get_seg_results(self,preds_dict,img_metas):

    #     masks=preds_dict['all_mask_preds'][-1]
    #     cls_scores=preds_dict['all_cls_scores'][-1]
    #     queries2img_masks=preds_dict['queries2img_masks'][-1]
    #     B=len(masks)
    #     num_cam=len(masks[0])
    #     num_classes=cls_scores.shape[-1]
    #     results=[]
    #     for bi in range(B):
    #         results.append([])
    #         for vi in range(num_cam):
    #             cls_score_per_img=cls_scores[bi,queries2img_masks[bi,vi]]
    #             num_queries=len(cls_score_per_img)

    #             scores_per_img,topk_indices=cls_score_per_img.flatten(0,1).topk(num_queries,sorted=True)
    #             mask_indices=topk_indices//num_classes
    #             labels_per_img=topk_indices%num_classes
    #             masks_per_img=masks[bi,vi,queries2img_masks[bi,vi]]
    #             if num_queries==0:
    #                 single_result=[[] for _ in range(num_classes)]
    #             else:
    #                 single_result=self.get_seg_masks(masks_per_img,labels_per_img,scores_per_img,img_metas[bi])
    #             results[bi].append(single_result)

    #     return results

    def get_seg_results(self, preds_dict, img_metas):

        masks = preds_dict['all_mask_preds'][-1]
        cls_scores = preds_dict['all_cls_scores'][-1]
        queries2img_masks = preds_dict['queries2img_masks'][-1]
        B = len(masks)
        num_cam = len(masks[0])
        # H,W=masks.shape[-2:]
        H, W = img_metas[0]['input_shape']
        results = []
        for bi in range(B):
            result = []
            for vi in range(num_cam):

                panoptic_seg = masks.new_full((H, W), self.num_classes, dtype=torch.long)
                if self.with_stuff:
                    panoptic_seg[panoptic_seg == self.num_classes] = self.num_classes + self.num_stuff_classes
                    for ci in self.vis_stuff_classes_orders:
                        stuff_mask = preds_dict['all_mask_preds_stuff'][-1][
                            bi, vi, (ci - self.num_classes) * num_cam + vi].sigmoid()
                        stuff_mask = F.interpolate(stuff_mask[None, None], (H, W), mode='bilinear')[0, 0]
                        panoptic_seg[stuff_mask >= 0.5] = ci

                cls_score_per_img = cls_scores[bi, queries2img_masks[bi, vi]].sigmoid()
                scores_per_img, labels_per_img = cls_score_per_img.max(dim=1)
                mask_per_img = masks[bi, vi, queries2img_masks[bi, vi]]

                if len(scores_per_img) == 0:
                    result.append(panoptic_seg)
                    continue
                mask_per_img = F.interpolate(mask_per_img[None], (H, W), mode='bilinear')[0]
                cur_prob_masks = scores_per_img.view(-1, 1, 1) * mask_per_img.sigmoid()
                cur_mask_ids = cur_prob_masks.argmax(0)

                sorted_inds = torch.argsort(-scores_per_img)
                current_segment_id = 0

                for k in sorted_inds:
                    if scores_per_img[k] < 0.3:
                        continue

                    mask = cur_mask_ids == k
                    mask_area = mask.sum().item()
                    original_area = (mask_per_img[k] >= 0.5).sum().item()

                    if mask_area > 0 and original_area > 0:
                        if mask_area / original_area < 0.6:
                            continue
                        mask = mask & (mask_per_img[k] >= 0.5)
                        panoptic_seg[mask] = labels_per_img[k] + 100 * current_segment_id
                        current_segment_id += 1

                result.append(panoptic_seg)
            result = torch.stack(result)
            results.append(result)
            results = torch.stack(results).cpu().numpy()
        return results

    def get_depth_results(self, preds_dict, img_metas):
        depths = preds_dict['all_depth_preds'][-1]
        b, v, c, h, w = depths.shape
        H, W = img_metas[0]['input_shape']
        # interpolate    
        depths = depths.view(-1, c, h, w)
        results = F.interpolate(depths, (H, W), mode='bilinear')
        results = results.view(b, v, c, H, W)
        return results

    def get_queries2img_mask(self, reference_points, img_metas):
        pc_range = self.pc_range
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)
        reference_points = reference_points.clone()
        # reference_points_x * (x_max - x_min) + x_min
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # reference_points_y * (y_max - y_min) + y_min
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # reference_points_z * (z_max - z_min) + z_min
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        # reference_points (B, num_queries, 4)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        B, num_query = reference_points.size()[:2]
        num_cam = lidar2img.size(1)
        reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
        eps = 1e-5
        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        reference_points_cam = (reference_points_cam - 0.5) * 2
        mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 1:2] > -1.0)
                & (reference_points_cam[..., 1:2] < 1.0))
        mask = mask.view(B, num_cam, num_query)
        mask = torch.nan_to_num(mask)

        return mask
