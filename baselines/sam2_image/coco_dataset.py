from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class CocoInstance:
    image_id: int
    ann_id: int
    category_id: int
    bbox_xyxy: np.ndarray
    area: float
    gt_mask: np.ndarray


@dataclass(frozen=True)
class CocoImageRecord:
    image_id: int
    file_name: str
    image_path: Path
    instances: list[CocoInstance]


def _load_coco_api():
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for COCO image evaluation. "
            "Install it in the SAM2 environment with: pip install pycocotools"
        ) from exc
    return COCO


class CocoBoxPromptDataset:
    """COCO instances as GT-box prompts and GT masks.

    The dataset groups annotations by image so SAM2 image embeddings are computed
    once per image and reused for all instance box prompts in that image.
    """

    def __init__(
        self,
        image_root: str | Path,
        ann_file: str | Path,
        category_ids: Iterable[int] | None = None,
        min_area: float = 1.0,
        include_crowd: bool = False,
    ) -> None:
        COCO = _load_coco_api()
        self.image_root = Path(image_root)
        self.ann_file = Path(ann_file)
        self.coco = COCO(str(self.ann_file))
        self.min_area = min_area
        self.include_crowd = include_crowd
        self.category_ids = list(category_ids) if category_ids else None

        image_ids = sorted(self.coco.getImgIds(catIds=self.category_ids or []))
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def _valid_ann(self, ann: dict) -> bool:
        if not self.include_crowd and ann.get("iscrowd", 0):
            return False
        if ann.get("area", 0.0) < self.min_area:
            return False
        if "segmentation" not in ann or not ann["segmentation"]:
            return False
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        return w > 0 and h > 0

    def _load_record(self, image_id: int) -> CocoImageRecord | None:
        info = self.coco.loadImgs([image_id])[0]
        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id],
            catIds=self.category_ids or [],
            iscrowd=None if self.include_crowd else False,
        )
        anns = [ann for ann in self.coco.loadAnns(ann_ids) if self._valid_ann(ann)]

        instances: list[CocoInstance] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            gt_mask = self.coco.annToMask(ann).astype(bool)
            if not gt_mask.any():
                continue
            instances.append(
                CocoInstance(
                    image_id=image_id,
                    ann_id=ann["id"],
                    category_id=ann["category_id"],
                    bbox_xyxy=bbox_xyxy,
                    area=float(ann["area"]),
                    gt_mask=gt_mask,
                )
            )

        if not instances:
            return None

        return CocoImageRecord(
            image_id=image_id,
            file_name=info["file_name"],
            image_path=self.image_root / info["file_name"],
            instances=instances,
        )

    def iter_image_ids(self, image_ids: Iterable[int]) -> Iterator[CocoImageRecord]:
        for image_id in image_ids:
            record = self._load_record(image_id)
            if record is not None:
                yield record

    def __iter__(self) -> Iterator[CocoImageRecord]:
        yield from self.iter_image_ids(self.image_ids)
