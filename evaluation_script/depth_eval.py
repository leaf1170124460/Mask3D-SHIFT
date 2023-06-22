"""SHIFT depth evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .common import Evaluator


class DepthEvaluator(Evaluator):
    METRICS = ["abs_err", "silog", "rmse_log"]

    def __init__(self, min_depth: float = 1.0, max_depth: float = 80.0) -> None:
        """Initialize the depth evaluator."""
        self.min_depth = min_depth
        self.max_depth = max_depth
        super().__init__()

    def mean_absolute_error(self, pred, target):
        """Compute the mean absolute error.
        Args:
            pred (np.array): Prediction depth map, in shape (H, W).
            target (np.array): Target depth map, in shape (H, W).
        Returns:
            float: Mean absolute error.
        """
        mask = (target > self.min_depth) & (target < self.max_depth)
        return np.mean(np.abs(pred[mask] - target[mask]))

    def silog(self, pred, target, eps=1e-6):
        """Compute the scale-invariant log error of KITTI.
        Args:
            pred (np.array): Prediction depth map, in shape (H, W).
            target (np.array): Target depth map, in shape (H, W).
            eps (float, optional): Epsilon. Defaults to 1e-6.
        Returns:
            float: Silog error.
        """
        mask = (target > self.min_depth) & (target < self.max_depth)
        err = np.log(pred[mask] + eps) - np.log(target[mask] + eps)
        return np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    def rmse_log(self, pred, target, eps=1e-6):
        """Compute the scale-invariant log error of KITTI.
        Args:
            pred (np.array): Prediction depth map, in shape (H, W).
            target (np.array): Target depth map, in shape (H, W).
            eps (float, optional): Epsilon. Defaults to 1e-6.
        Returns:
            float: Silog error.
        """
        mask = (target > self.min_depth) & (target < self.max_depth)
        err = (np.log(target[mask] + eps) - np.log(pred[mask] + eps)) ** 2
        return np.sqrt(err.mean())

    def crop(self, mask):
        return mask[:740, :]

    def process(self, prediction: np.array, target: np.array) -> None:
        """Process a batch of data.
        Args:
            prediction (np.array): Prediction depth map.
            target (np.array): Target depth map.
        """
        prediction = self.crop(prediction)
        target = self.crop(target)
        mae = self.mean_absolute_error(prediction, target)
        silog = self.silog(prediction, target)
        rmse_log = self.rmse_log(prediction, target)
        self.metrics["abs_err"].append(mae)
        self.metrics["silog"].append(silog)
        self.metrics["rmse_log"].append(rmse_log)

    def preprocess(self, data: np.array) -> np.array:
        if len(data.shape) == 3 and data.shape[2] == 3:
            data = data.astype(np.float32)
            data = data[:, :, 0] + data[:, :, 1] * 256 + data[:, :, 2] * 256 * 256
            data /= 16777.216
        else:
            data = data.astype(np.float32)
            data /= 3.1875
        return data

    def evaluate(self) -> Dict[str, float]:
        """Evaluate all predictions according to given metric.
        Returns:
            dict[str, float]: Evaluation results.
        """
        return {metric: np.nanmean(self.metrics[metric]) for metric in self.metrics}
