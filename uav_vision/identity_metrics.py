"""
Identity metrics for tracking evaluated on a fixed set of detections: IDF1, IDP, IDR and ID switches.

These are the standard multi-object tracking identity scores (Ristani et al., 2016, for IDF1;
CLEAR MOT for ID switches), computed CONDITIONED ON DETECTIONS. Every box scored is a box the
detector produced and a person labelled; a person the detector never boxed is not counted at all.
That is the honest scope when the ground truth is drawn over the detector's own output, and it
must be stated with the number: a full MOT IDF1 would also charge the missed people, and would be
lower.

Two sequences per box, in the same order: the ground-truth identity a person gave it, and the id
the system under test gave it. A box the system left without an id (a tracker that never
confirmed it) is not a hypothesis: it adds to the misses and to nothing else. Boxes whose ground
truth is not an identity -- not a person, or undecidable -- are left out before scoring and counted
separately, because an identity score has nothing to say about them.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np

NO_IDENTITY = frozenset({"x", "?"})


def identity_scores(
    truth: Sequence[Optional[Hashable]],
    predicted: Sequence[Optional[Hashable]],
    order: Sequence[float],
) -> Dict[str, float]:
    """
    Scores predicted ids against ground-truth identities over the same boxes.

    Args:
        truth: ground-truth identity per box. None, or a label in NO_IDENTITY, leaves the box out.
        predicted: id the system assigned per box. None means the system gave it no id.
        order: a time key per box (frame index or timestamp), used only to count switches.

    Returns:
        idf1, idp, idr: identity F1, precision and recall, from the one-to-one matching between
            true identities and predicted ids that maximises the boxes they share.
        id_switches: times the id on a true identity changed from its previous assigned id,
            walking that identity's boxes in time order and skipping boxes with no id.
        boxes, boxes_with_id, identities, predicted_ids, excluded.
    """
    from scipy.optimize import linear_sum_assignment

    if not (len(truth) == len(predicted) == len(order)):
        raise ValueError("truth, predicted and order must describe the same boxes")

    keep = [i for i, g in enumerate(truth) if g is not None and g not in NO_IDENTITY]
    excluded = len(truth) - len(keep)
    gts = sorted({truth[i] for i in keep}, key=str)
    preds = sorted({predicted[i] for i in keep if predicted[i] is not None}, key=str)
    gi = {g: k for k, g in enumerate(gts)}
    pi = {p: k for k, p in enumerate(preds)}

    shared = np.zeros((len(gts), len(preds)), dtype=np.int64)
    for i in keep:
        if predicted[i] is not None:
            shared[gi[truth[i]], pi[predicted[i]]] += 1

    n_truth = len(keep)
    n_pred = int(shared.sum())
    idtp = 0
    if shared.size:
        rows, cols = linear_sum_assignment(-shared)
        idtp = int(shared[rows, cols].sum())
    idfn = n_truth - idtp
    idfp = n_pred - idtp

    switches = 0
    by_identity: Dict[Hashable, List[int]] = {}
    for i in keep:
        by_identity.setdefault(truth[i], []).append(i)
    for boxes in by_identity.values():
        last = None
        for i in sorted(boxes, key=lambda j: order[j]):
            p = predicted[i]
            if p is None:
                continue
            if last is not None and p != last:
                switches += 1
            last = p

    denominator = 2 * idtp + idfp + idfn
    return {
        "idf1": 2 * idtp / denominator if denominator else 0.0,
        "idp": idtp / n_pred if n_pred else 0.0,
        "idr": idtp / n_truth if n_truth else 0.0,
        "id_switches": switches,
        "boxes": n_truth,
        "boxes_with_id": n_pred,
        "identities": len(gts),
        "predicted_ids": len(preds),
        "excluded": excluded,
    }
