"""SHIFT base evaluation."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import tqdm
from PIL import Image


class Evaluator:
    """Abstract evaluator class."""

    METRICS: List[str] = []

    def __init__(self) -> None:
        """Initialize evaluator."""
        self.reset()

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self.metrics = {metric: [] for metric in self.METRICS}

    def process(self, *args: Any) -> None:  # type: ignore
        """Process a batch of data."""
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        """Evaluate all predictions according to given metric.
        Returns:
            dict[str, float]: Evaluation results.
        """
        return {metric: np.nanmean(values) for metric, values in self.metrics.items()}

    def preprocess(self, data: np.array) -> np.array:
        """Preprocess data before evaluation.

        Args:
            data (np.array): Data to be processed.
        Returns:
            np.array: Processed data.
        """
        return data

    def process_from_folder(
        self,
        pred_folder_path: str,
        target_folder_path: str,
        max_num_seqs: int = -1,
        used_seqs=None,
    ) -> Dict[str, float]:
        """Process all predictions in a folder of images.

        Args:
            pred_folder_path (str): Path to folder containing predictions.
            target_folder_path (str): Path to folder containing targets.

        Returns:
            dict[str, float]: Evaluation results.
        """
        self.reset()
        seqs = sorted(os.listdir(target_folder_path))
        if max_num_seqs > 0:
            seqs = seqs[:max_num_seqs]
        for seq_name in tqdm.tqdm(seqs):
            if used_seqs is not None and seq_name not in used_seqs:
                continue
            for frame_name in sorted(
                os.listdir(os.path.join(target_folder_path, seq_name))
            ):
                if not frame_name.endswith(".png"):
                    continue
                try:
                    pred = np.array(
                        Image.open(os.path.join(pred_folder_path, seq_name, frame_name))
                    )
                    target = np.array(
                        Image.open(
                            os.path.join(target_folder_path, seq_name, frame_name)
                        )
                    )
                    pred = self.preprocess(pred)
                    target = self.preprocess(target)
                    self.process(pred, target)
                except Exception as e:
                    print(f"Error when evaluating {seq_name}/{frame_name}: {e}")
                    # append 0 to metrics
                    for metric in self.METRICS:
                        self.metrics[metric].append(0.0)
        return self.evaluate()
