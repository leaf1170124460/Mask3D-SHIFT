import os
import os.path as osp

import cv2
import numpy as np
import torch

from ..datasets.shift.mmdet3d_dataset import SHIFTDataset


def format_depth_results(depth_results, img_metas, out_dir, depth_bound):
    out_depth_dir = osp.join(out_dir, 'depth')
    if not osp.exists(out_depth_dir):
        os.makedirs(out_depth_dir)

    depth_results_list = []
    for bi in range(len(depth_results)):
        new_depth_results = []
        for vi in range(len(depth_results[bi])):
            # bin to depth
            depth = depth_results[bi][vi]
            depth_image = torch.argmax(depth, dim=0)

            # depth to bgr depth
            depth_image = np.array(depth_image.cpu(), dtype=np.int32)
            depth_image = depth_image * SHIFTDataset.DEPTH_FACTOR

            bgr_shape = tuple(list(depth_image.shape) + [3])
            bgr_depth_map = np.zeros(bgr_shape, dtype=np.uint8)
            for i in range(3):
                bgr_depth_map[..., 2 - i] = depth_image % 256
                depth_image //= 256

            # save bgr depth
            video_name = img_metas[vi]['video_name'][0]
            image_name = img_metas[vi]['image_name'][0]
            current_out_depth_dir = osp.join(out_depth_dir, video_name)
            if not osp.exists(current_out_depth_dir):
                os.mkdir(current_out_depth_dir)

            # only keep one frame
            if image_name.startswith('00000100'):
                out_depth_path = osp.join(current_out_depth_dir,
                                          image_name.replace('img', 'depth').replace('.jpg', '.png'))

                cv2.imwrite(out_depth_path, bgr_depth_map)

                new_depth_results.append(out_depth_path)

        depth_results_list.append(new_depth_results)
        return depth_results_list
