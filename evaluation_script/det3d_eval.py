from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pyquaternion
import tqdm
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricDataList
from scalabel.label.typing import Box3D, Config, Frame

TP_METRICS = ["trans_err", "scale_err", "orient_err"]


class DetectionMetrics:
    """Stores average precision and true positive metric results. Provides properties to summarize."""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """Calculates the mean over distance thresholds for each label."""
        return {
            class_name: np.mean(list(d.values()))
            for class_name, d in self._label_aps.items()
        }

    @property
    def mean_ap(self) -> float:
        """Calculates the mean AP by averaging over distance thresholds and classes."""
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """Calculates the mean true positive error across all classes for each metric."""
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:
            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    def serialize(self):
        return {
            "mean_dist_aps": self.mean_dist_aps,
            "mean_ap": self.mean_ap,
            "tp_errors": self.tp_errors,
            "tp_scores": self.tp_scores,
        }


def cam_to_lidar(box3d: Box3D):
    """Change box3d from camera coordinate to lidar coordinate.

    Coordinates in LiDAR:

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0


    Coordinates in camera:

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    """
    rot_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    location = np.array(box3d.location)
    x_size, y_size, z_size = box3d.dimension
    yaw = box3d.orientation[1]

    location = location[np.newaxis, :] @ rot_mat.transpose()
    dimension = [x_size, z_size, y_size]
    yaw = -yaw - np.pi / 2
    quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw)
    return (location[0].tolist(), dimension, list(quat))


def evaluate_det_3d(
    gt_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    dist_ths: List[float] = [0.5, 1.0, 2.0],
    dist_th_tp: float = 1.0,
):
    """Evaluate 3D detection using NuScenes detection metrics."""
    gt_boxes = EvalBoxes()
    pred_boxes = EvalBoxes()
    for frame in tqdm.tqdm(gt_frames):
        boxes = []
        for label in frame.labels:
            if label.box3d is not None:
                location, dimensions, orientation = cam_to_lidar(label.box3d)
                boxes.append(
                    DetectionBox(
                        f"{frame.name}_{frame.videoName}",
                        location,
                        dimensions,
                        orientation,
                        detection_name=label.category,
                        detection_score=label.score if label.score is not None else 1.0,
                    )
                )
        gt_boxes.add_boxes(f"{frame.name}_{frame.videoName}", boxes)
    for frame in tqdm.tqdm(pred_frames):
        boxes = []
        for label in frame.labels:
            if label.box3d is not None:
                location, dimensions, orientation = cam_to_lidar(label.box3d)
                boxes.append(
                    DetectionBox(
                        f"{frame.name}_{frame.videoName}",
                        location,
                        dimensions,
                        orientation,
                        detection_name=label.category,
                        detection_score=label.score if label.score is not None else 1.0,
                    )
                )
        pred_boxes.add_boxes(f"{frame.name}_{frame.videoName}", boxes)

    # Run evaluation
    class_names = [category.name for category in config.categories]
    metric_data_list = DetectionMetricDataList()
    for class_name in class_names:
        for dist_th in dist_ths:
            md = accumulate(
                gt_boxes,
                pred_boxes,
                class_name,
                center_distance,
                dist_th,
            )
            metric_data_list.set(class_name, dist_th, md)

    metrics = DetectionMetrics(class_names)
    for class_name in class_names:
        # Compute APs.
        for dist_th in dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, min_recall=0.1, min_precision=0.1)
            metrics.add_label_ap(class_name, dist_th, ap)

        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, dist_th_tp)]
            tp = calc_tp(metric_data, min_recall=0.1, metric_name=metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    metrics = metrics.serialize()
    return metrics
