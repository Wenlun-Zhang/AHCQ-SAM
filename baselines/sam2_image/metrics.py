from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


def coco_area_bucket(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


@dataclass
class MeanIoUMeter:
    values: list[float] = field(default_factory=list)
    by_size: dict[str, list[float]] = field(
        default_factory=lambda: {"small": [], "medium": [], "large": []}
    )

    def update(self, iou: float, area: float) -> None:
        self.values.append(iou)
        self.by_size[coco_area_bucket(area)].append(iou)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def summary(self) -> dict[str, float | int]:
        return {
            "instances": len(self.values),
            "mIoU": self._mean(self.values),
            "mIoU_small": self._mean(self.by_size["small"]),
            "mIoU_medium": self._mean(self.by_size["medium"]),
            "mIoU_large": self._mean(self.by_size["large"]),
            "num_small": len(self.by_size["small"]),
            "num_medium": len(self.by_size["medium"]),
            "num_large": len(self.by_size["large"]),
        }

