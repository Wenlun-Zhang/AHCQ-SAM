from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


SIS_STATS = {"compressions": 0, "synthesized_memories": 0}


def reset_sis_stats() -> None:
    SIS_STATS["compressions"] = 0
    SIS_STATS["synthesized_memories"] = 0


def get_sis_stats() -> dict[str, int]:
    return dict(SIS_STATS)


def maybe_apply_sis(inference_state: dict[str, Any], sis_config) -> None:
    if sis_config is None or not getattr(sis_config, "enabled", False):
        return
    capacity = int(getattr(sis_config, "capacity", 0))
    topk = int(getattr(sis_config, "topk", 3))
    if capacity <= 0 or topk <= 1:
        return

    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        compressed = compress_non_cond_memory(
            obj_output_dict["non_cond_frame_outputs"],
            capacity,
            topk,
        )
        if compressed:
            SIS_STATS["compressions"] += 1
            SIS_STATS["synthesized_memories"] += 1


def compress_non_cond_memory(
    non_cond_outputs: dict[int, dict[str, Any]],
    capacity: int,
    topk: int,
) -> bool:
    frame_ids = sorted(non_cond_outputs)
    if len(frame_ids) <= capacity:
        return False

    latest_frame = frame_ids[-1]
    reference = non_cond_outputs[latest_frame]
    reference_vec = memory_vector(reference)
    if reference_vec is None:
        prune_oldest(non_cond_outputs, capacity)
        return False

    candidate_ids = [
        frame_id
        for frame_id in frame_ids[:-1]
        if memory_vector(non_cond_outputs[frame_id]) is not None
    ]
    if len(candidate_ids) < topk:
        prune_oldest(non_cond_outputs, capacity)
        return False

    similarities = []
    for frame_id in candidate_ids:
        vec = memory_vector(non_cond_outputs[frame_id])
        sim = F.cosine_similarity(reference_vec, vec, dim=0)
        similarities.append((frame_id, float(sim.cpu())))
    selected = sorted(similarities, key=lambda item: item[1], reverse=True)[:topk]
    selected_ids = [frame_id for frame_id, _score in selected]
    scores = torch.tensor([score for _frame_id, score in selected], dtype=torch.float32)
    weights = torch.softmax(scores, dim=0)

    synthetic = synthesize_outputs(
        [non_cond_outputs[frame_id] for frame_id in selected_ids],
        weights,
    )
    synthetic_frame = max(selected_ids)
    for frame_id in selected_ids:
        non_cond_outputs.pop(frame_id, None)
    non_cond_outputs[synthetic_frame] = synthetic

    prune_oldest(non_cond_outputs, capacity)
    return True


def memory_vector(output: dict[str, Any]) -> torch.Tensor | None:
    features = output.get("maskmem_features")
    if not torch.is_tensor(features) or features.numel() == 0:
        return None
    return features.detach().float().flatten()


def synthesize_outputs(outputs: list[dict[str, Any]], weights: torch.Tensor) -> dict[str, Any]:
    base = outputs[0]
    return {
        "maskmem_features": weighted_tensor(
            [output.get("maskmem_features") for output in outputs],
            weights,
        ),
        "maskmem_pos_enc": base.get("maskmem_pos_enc"),
        "pred_masks": weighted_tensor(
            [output.get("pred_masks") for output in outputs],
            weights,
        ),
        "obj_ptr": weighted_tensor(
            [output.get("obj_ptr") for output in outputs],
            weights,
        ),
        "object_score_logits": weighted_tensor(
            [output.get("object_score_logits") for output in outputs],
            weights,
        ),
    }


def weighted_tensor(values: list[Any], weights: torch.Tensor) -> Any:
    tensors = [value for value in values if torch.is_tensor(value)]
    if len(tensors) != len(values):
        return values[0]
    device = tensors[0].device
    dtype = tensors[0].dtype
    result = torch.zeros_like(tensors[0], dtype=torch.float32)
    for tensor, weight in zip(tensors, weights.to(device=device)):
        result = result + tensor.to(device=device, dtype=torch.float32) * weight
    return result.to(dtype=dtype)


def prune_oldest(non_cond_outputs: dict[int, dict[str, Any]], capacity: int) -> None:
    while len(non_cond_outputs) > capacity:
        oldest = min(non_cond_outputs)
        non_cond_outputs.pop(oldest, None)
