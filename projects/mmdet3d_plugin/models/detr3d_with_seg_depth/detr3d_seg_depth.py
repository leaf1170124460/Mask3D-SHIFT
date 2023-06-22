import copy
import random

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class Detr3D_Seg_Depth(MVXTwoStageDetector):
    """Detr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 mask_assign_stride=8,
                 img_feats_levels=None,
                 ):
        super(Detr3D_Seg_Depth,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.mask_assign_stride = mask_assign_stride
        self.img_feats_levels = img_feats_levels

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            # if img.dim() == 5 and img.size(0) == 1:
            #     img.squeeze_()
            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        img_feats = img_feats_reshaped
        img_feats = [img_feats[li] for li in self.img_feats_levels]
        if self.mask_assign_stride == 8:

            seg_feats = img_feats[0]
        else:
            seg_feats = img_feats[1]
        return img_feats, seg_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          seg_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_masks=None,
                          gt_mask_labels=None,
                          gt_masks_stuff=None,
                          gt_mask_labels_stuff=None,
                          gt_bboxes_ignore=None,
                          gt_assigned_masks=None,
                          gt_depth=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, seg_feats, img_metas)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_masks, gt_mask_labels, gt_assigned_masks, gt_masks_stuff,
                       gt_mask_labels_stuff, gt_depth, outs]

        losses = self.pts_bbox_head.loss(*loss_inputs)

        losses.update(losses)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_mask_labels=None,
                      gt_masks=None,
                      gt_assigned_masks=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      gt_depth=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if isinstance(img, list):
            H, W = img[0].shape[-2:]
        else:
            H, W = img.shape[-2:]
        assign_H, assign_W = H // self.mask_assign_stride, W // self.mask_assign_stride
        gt_masks_thing, gt_masks_stuff, gt_mask_labels_thing, gt_mask_labels_stuff = self.adjust_gt_masks(gt_masks,
                                                                                                          gt_mask_labels,
                                                                                                          assign_H,
                                                                                                          assign_W)
        img_feats, seg_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, seg_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks,
                                            gt_mask_labels=gt_mask_labels, gt_assigned_masks=gt_assigned_masks,
                                            gt_masks_stuff=gt_masks_stuff, gt_mask_labels_stuff=gt_mask_labels_stuff,
                                            gt_depth=gt_depth)
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)
        # if num_augs == 1:
        #     img = [img] if img is None else img
        #     return self.simple_test(None, img_metas[0], img[0], **kwargs)
        # else:
        #     return self.aug_test(None, img_metas, img, **kwargs)

    def simple_test_pts(self, x, seg_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, seg_feats, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # get seg results
        seg_results = self.pts_bbox_head.get_seg_results(outs, img_metas)
        # get depth results
        depth_results = self.pts_bbox_head.get_depth_results(outs, img_metas)
        return bbox_results, seg_results, depth_results

    def simple_test(self, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_feats, seg_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts, seg_results, depth_results = self.simple_test_pts(
            img_feats, seg_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox, seg_result, depth_result in zip(bbox_list, bbox_pts, seg_results, depth_results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['seg_result'] = seg_result
            result_dict['depth_result'] = depth_result
        results = [bbox_list, img, img_metas]

        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def adjust_gt_masks(self, gt_masks, gt_mask_labels, assign_H, assign_W):
        gt_masks_thing, gt_masks_stuff, gt_mask_labels_thing, gt_mask_labels_stuff = [], [], [], []
        for b_i in range(len(gt_masks)):
            gt_masks_thing.append([])
            gt_masks_stuff.append([])
            gt_mask_labels_thing.append([])
            gt_mask_labels_stuff.append([])

            for v_i in range(len(gt_masks[b_i])):
                gt_masks[b_i][v_i] = \
                    F.interpolate(gt_masks[b_i][v_i][None].float(), (assign_H, assign_W), mode='bilinear',
                                  align_corners=False)[0]
                gt_masks_thing[b_i].append(gt_masks[b_i][v_i][gt_mask_labels[b_i][v_i] <= 9])
                gt_masks_stuff[b_i].append(gt_masks[b_i][v_i][gt_mask_labels[b_i][v_i] > 9])
                gt_mask_labels_thing[b_i].append(gt_mask_labels[b_i][v_i][gt_mask_labels[b_i][v_i] <= 9])
                gt_mask_labels_stuff[b_i].append(gt_mask_labels[b_i][v_i][gt_mask_labels[b_i][v_i] > 9])
        return gt_masks_thing, gt_masks_stuff, gt_mask_labels_thing, gt_mask_labels_stuff

    # def show_results(self, data, result, out_dir, show=False, score_thr=None):
    #     PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
    #                (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
    #                (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
    #                (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
    #                (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    #                (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
    #                (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
    #                (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
    #                (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
    #                (134, 134, 103), (145, 148, 174), (255, 208, 186),
    #                (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
    #                (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
    #                (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
    #                (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
    #                (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
    #                (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
    #                (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
    #                (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
    #                (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
    #                (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
    #                (191, 162, 208), (255, 255, 128), (147, 211, 203),
    #                (150, 100, 100), (168, 171, 172), (146, 112, 198),
    #                (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
    #                (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
    #                (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
    #                (106, 154, 176),
    #                (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
    #                (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
    #                (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
    #                (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
    #                (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
    #                (146, 139, 141),
    #                (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
    #                (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
    #                (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
    #                (102, 102, 156), (250, 141, 255)]
    #     img_metas=data['img_metas'][0]._data[0]
    #     for batchindex in range(len(img_metas)):
    #         gt_masks=data['gt_masks'][0].data[0][batchindex]
    #         gt_bboxes=data['gt_bboxes_3d'][0].data[0][batchindex]
    #         gt_assigned_masks=data['gt_assigned_masks'][0].data[batchindex]
    #         for view_idx, masklist in enumerate(result[batchindex]['seg_masks']):
    #             imgname = img_metas[batchindex]['filename'][view_idx].split('/')[-1]
    #             img_path=img_metas[batchindex]['filename'][view_idx]
    #             ori_img=mmcv.imread(img_path)
    #             gt_img=ori_img.copy()
    #             pngname = imgname.replace('jpg', 'png')
    #             H, W, C = img_metas[batchindex]['img_shape'][view_idx]
    #             H = 900
    #             W = 1600
    #             segimage = np.zeros((H, W, 3))
    #             bkmask = np.ones((H, W))

    #             colorbox = []

    #             for i in range(len(masklist)):
    #                 for j in range(len(masklist[i])):
    #                     color = random.choice(PALETTE)
    #                     while color in colorbox:
    #                         color = random.choice(PALETTE)
    #                     colorbox.append(color)

    #                     segimage[masklist[i][j]] = color
    #                     bkmask[masklist[i][j]] = 0

    #             alpha=0.7
    #             segimage=cv2.addWeighted(ori_img,alpha,segimage.astype(np.uint8),1-alpha,0)
    #             bboxes=result[batchindex]['pts_bbox']['boxes_3d']
    #             pos_idx=result[batchindex]['pts_bbox']['scores_3d']>0.3
    #             bboxes=bboxes[pos_idx]
    #             lidar2img=img_metas[batchindex]['lidar2img'][view_idx]
    #             vis_image=draw_lidar_bbox3d_on_img(bboxes,segimage,lidar2img,img_metas=None)

    #             for mi,mask in enumerate(gt_masks[view_idx]):
    #                 if mi not in gt_assigned_masks[:,view_idx]:
    #                     continue
    #                 color = np.array([random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
    #                 mask = np.array(mask[:, :, None]).repeat(3, axis=2)
    #                 gt_img=np.where(mask,gt_img*alpha+(1-alpha)*color,gt_img)

    #             gt_img=draw_lidar_bbox3d_on_img(gt_bboxes,gt_img,lidar2img,img_metas=None)

    #             vis_image=np.concatenate([gt_img,vis_image])
    #             vis_image=cv2.resize(vis_image,(1800,1600))
    #             cv2.imshow('visualize',vis_image)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    def show_results(self, data, result, out_dir, show=False, score_thr=None):
        PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                   (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                   (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                   (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                   (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                   (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                   (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                   (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                   (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                   (134, 134, 103), (145, 148, 174), (255, 208, 186),
                   (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                   (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                   (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
                   (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
                   (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
                   (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
                   (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
                   (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
                   (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
                   (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
                   (191, 162, 208), (255, 255, 128), (147, 211, 203),
                   (150, 100, 100), (168, 171, 172), (146, 112, 198),
                   (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
                   (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
                   (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
                   (106, 154, 176),
                   (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
                   (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
                   (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
                   (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
                   (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
                   (146, 139, 141),
                   (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
                   (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
                   (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
                   (102, 102, 156), (250, 141, 255)]
        img_metas = data['img_metas'][0]._data[0]
        for batchindex in range(len(img_metas)):
            if 'gt_bboxes_3d' in data:
                with_gt = True
                gt_masks = data['gt_masks'][0].data[0][batchindex]
                gt_mask_labels = data['gt_mask_labels'][0].data[0][batchindex]
                gt_bboxes = data['gt_bboxes_3d'][0].data[0][batchindex]
                gt_assigned_masks = data['gt_assigned_masks'][0].data[batchindex]
            else:
                with_gt = False
            seg_result = result[batchindex]['seg_result']
            for vi in range(len(seg_result)):
                img_path = img_metas[batchindex]['filename'][vi]
                ori_img = mmcv.imread(img_path)
                gt_img = ori_img.copy()
                H, W, C, _ = img_metas[batchindex]['ori_shape']
                seg = seg_result[vi]
                inds = np.unique(seg)
                segimage = np.zeros((seg.shape[0], seg.shape[1], 3))
                colorbox = []
                for idx in inds:
                    if self.pts_bbox_head.with_stuff:
                        if idx == self.pts_bbox_head.num_classes + self.pts_bbox_head.num_stuff_classes:
                            continue
                    else:
                        if idx == self.pts_bbox_head.num_classes:
                            continue
                    color = random.choice(PALETTE)
                    while color in colorbox:
                        color = random.choice(PALETTE)
                    colorbox.append(color)
                    segimage[seg == idx] = color
                pad_H, pad_W = img_metas[batchindex]['pad_shape'][vi][:2]
                segimage = cv2.resize(segimage, (pad_W, pad_H))
                segimage = segimage[14:H + 14, :W]

                alpha = 0.7
                segimage = cv2.addWeighted(ori_img, alpha, segimage.astype(np.uint8), 1 - alpha, 0)
                bboxes = result[batchindex]['pts_bbox']['boxes_3d']
                pos_idx = result[batchindex]['pts_bbox']['scores_3d'] > 0.5
                bboxes = bboxes[pos_idx]
                lidar2img = img_metas[batchindex]['lidar2img'][vi]
                vis_image = draw_lidar_bbox3d_on_img(bboxes, segimage, lidar2img, img_metas=None)

                if with_gt:
                    for mi, mask in enumerate(gt_masks[vi]):
                        label = gt_mask_labels[vi][mi].item()
                        if not (
                                self.pts_bbox_head.with_stuff and label in self.pts_bbox_head.vis_stuff_classes_orders) and mi not in gt_assigned_masks[
                                                                                                                                      :,
                                                                                                                                      vi]:
                            continue
                        color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
                        mask = np.array(mask[:, :, None]).repeat(3, axis=2)
                        gt_img = np.where(mask, gt_img * alpha + (1 - alpha) * color, gt_img)

                    gt_img = draw_lidar_bbox3d_on_img(gt_bboxes, gt_img, lidar2img, img_metas=None)

                    vis_image = np.concatenate([gt_img, vis_image])
                    # vis_image=cv2.resize(vis_image,(2*H,W))
                cv2.imshow('visualize', vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            try:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                         (corners[end, 0], corners[end, 1]), color, thickness,
                         cv2.LINE_AA)
            except:
                pass
    return img.astype(np.uint8)
