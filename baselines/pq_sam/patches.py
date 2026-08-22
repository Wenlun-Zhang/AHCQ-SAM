from __future__ import annotations

import cv2
import torch
from mmdet.core import bbox2result


def _limit_calib_prompts(results, output_boxes, calib, max_calib_prompts):
    if calib and max_calib_prompts is not None and output_boxes.shape[0] > max_calib_prompts:
        _, keep = torch.topk(results[0]["scores"], k=max_calib_prompts)
        output_boxes = output_boxes[keep]
    return output_boxes


def _format_outputs(wrapper, output_boxes, mask_pred, sam_score, results):
    if wrapper.best_in_multi_mask:
        sam_score, max_iou_idx = torch.max(sam_score, dim=1)
        mask_pred = mask_pred[torch.arange(mask_pred.size(0)), max_iou_idx]
    else:
        mask_pred = mask_pred.squeeze(1)
        sam_score = sam_score.squeeze(-1)

    label_pred = results[0]["labels"]
    score_pred = results[0]["scores"]
    if output_boxes.shape[0] != label_pred.shape[0]:
        score_pred = score_pred[: output_boxes.shape[0]]
        label_pred = label_pred[: output_boxes.shape[0]]

    mask_pred_binary = (mask_pred > wrapper.predictor.model.mask_threshold).float()
    if wrapper.use_sam_iou:
        det_scores = score_pred * sam_score
    else:
        mask_scores_per_image = (mask_pred * mask_pred_binary).flatten(1).sum(1) / (
            mask_pred_binary.flatten(1).sum(1) + 1e-6
        )
        det_scores = score_pred * mask_scores_per_image

    mask_pred_binary = mask_pred_binary.bool()
    bboxes = torch.cat([output_boxes, det_scores[:, None]], dim=-1)
    bbox_results = bbox2result(bboxes, label_pred, wrapper.num_classes)
    mask_results = [[] for _ in range(wrapper.num_classes)]
    for j, label in enumerate(label_pred):
        mask = mask_pred_binary[j].detach().cpu().numpy()
        mask_results[label].append(mask)
    return [(bbox_results, mask_results)]


def _pq_simple_test(
    self,
    img,
    img_metas,
    rescale=True,
    ori_img=None,
    calib=False,
    get_det_results=False,
    max_calib_prompts=None,
):
    assert rescale
    assert len(img_metas) == 1
    with torch.no_grad():
        results = self.det_model.simple_test(img, img_metas, rescale)
    if get_det_results:
        return results

    output_boxes = _limit_calib_prompts(results, results[0]["boxes"], calib, max_calib_prompts)

    if ori_img is None:
        image_path = img_metas[0]["filename"]
        ori_img = cv2.imread(image_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    self.predictor.set_image(ori_img)

    transformed_boxes = self.predictor.transform.apply_boxes_torch(output_boxes, ori_img.shape[:2])
    transformed_boxes = transformed_boxes.to(self.predictor.device)

    if calib:
        if not self.save_position_encoding:
            try:
                self.predictor.predict_pe()
                self.save_position_encoding = True
            except Exception:
                pass
        return self.predictor.predict_calib(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=self.best_in_multi_mask,
            return_logits=True,
        )

    mask_pred, sam_score, _ = self.predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=self.best_in_multi_mask,
        return_logits=True,
    )
    return _format_outputs(self, output_boxes, mask_pred, sam_score, results)


def _pq_only_forward_sam(self, det_results_and_torch_img, rescale=True, calib=False, max_calib_prompts=None):
    results = det_results_and_torch_img[0]
    torch_img = det_results_and_torch_img[1]
    assert rescale

    output_boxes = _limit_calib_prompts(results, results[0]["boxes"], calib, max_calib_prompts)

    self.predictor.set_torch_image(torch_img[0], torch_img[1])

    transformed_boxes = self.predictor.transform.apply_boxes_torch(output_boxes, torch_img[1])
    transformed_boxes = transformed_boxes.to(self.predictor.device)

    if calib:
        if not self.save_position_encoding:
            self.predictor.predict_pe()
            self.save_position_encoding = True
        return self.predictor.predict_calib(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=self.best_in_multi_mask,
            return_logits=True,
        )

    mask_pred, sam_score, _ = self.predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=self.best_in_multi_mask,
        return_logits=True,
    )
    return _format_outputs(self, output_boxes, mask_pred, sam_score, results)


def install_pq_sam_runtime_patches():
    from projects.instance_segment_anything.models.det_wrapper_instance_sam import DetWrapperInstanceSAM

    DetWrapperInstanceSAM.simple_test = _pq_simple_test
    DetWrapperInstanceSAM.only_forward_sam = _pq_only_forward_sam
