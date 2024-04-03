import numpy as np

from del_sort.deep_sort import linear_assignment


def iou(bbox, candidates):
    """
    计算交并比
    :param bbox: ndarray
        一个格式为（左上角x，左上角y，宽度，高度）的检测框
    :param candidates: ndarray
        后弦检测框的矩阵（每行一个），格式与bbox相同
    :return: ndarray
        在[0，1]区间内，交并比越大，意味着重合部分越大
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
    np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
    np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    计算轨迹与检测之间的距离
    :param tracks: List[deep_sort.track.Track]
        所有目标跟踪轨迹的列表
    :param detections: List[deep_sort.detection.Detection]
        所有目标检测的列表
    :param track_indices: Optional[List[int]]
        应匹配的轨迹索引列表。默认为所有的"tracks"
    :param detection_indices: Optional[List[int]]
        应匹配的检查索引列表。默认为所有的"detections"
    :return: ndarray
        一个形状为len(track_indices),len(detection_indices)的成本矩阵
        其中(i,j)是"1-iou(tracks[track_indices[i]],detections[detection_indices[j]])"
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFINITY_COST
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
