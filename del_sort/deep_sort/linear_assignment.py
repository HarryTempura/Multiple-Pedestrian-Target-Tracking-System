import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment

from del_sort.deep_sort import kalman_filter
from main import opt

INFINITY_COST = 1e+5


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    """
    最小成本匹配(解决线性分配问题)
    :param distance_metric:Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        用于计算轨迹和检测之间距离度量的函数，返回值是一个N*M的成本矩阵
    :param max_distance: float
        门控门限。成本大于这个的关联被忽略
    :param tracks: List[track.Track]
        当前时间步骤的预测轨迹列表
    :param detections: List[detection.Detection]
        当前时间步骤的检测列表
    :param track_indices: List[int]
        将"cost_matrix"中的行映射到"tracks"中的行的轨迹索引列表（见上文描述）
    :param detection_indices: List[int]
        将"cost_matrix"中的列映射到"detection"中的列的检测索引列表（见上文描述）
    :return: (List[(int,int)],List[int],List[int])
        一个三元组，包含匹配上的和未匹配上的轨迹和检测
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # 没有可以匹配的对象

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    cost_matrix_ = cost_matrix.copy()

    indices = linear_assignment(cost_matrix_)
    matches, unmatched_tracks, unmatched_detections = [], [], []

    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None
):
    """
    运行级联匹配
    :param distance_metric: Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        用于计算轨迹和检测之间距离度量的函数，返回值是一个N*M的成本矩阵
    :param max_distance: float
        门控门限。成本大于这个的关联被忽略
    :param cascade_depth: int
        级联深度，应为最大轨迹的年龄
    :param tracks: List[track.Track]
        当前时间步骤的预测轨迹列表
    :param detections: List[detection.Detection]
        当前时间步骤的检测列表
    :param track_indices: Optional[List[int]]
        将"cost_matrix"中的行映射到"tracks"中的行的轨迹索引列表（见上文描述）
    :param detection_indices: Optional[List[int]]
        将"cost_matrix"中的列映射到"detection"中的列的检测索引列表（见上文描述）
    :return: (List[(int,int)],List[int],List[int])
        一个三元组，包含匹配上的和未匹配上的轨迹和检测
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    if opt.woC:
        track_indices_l = [
            k for k in track_indices
            # if tracks[k].time_since_update == 1 + level
        ]
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections
        )
        matches += matches_l
    else:
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # 未匹配的检测为空
                break

            track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
            if len(track_indices_l) == 0:  # 在这个步骤没有可以匹配的对象
                continue

            matches_l, _, unmatched_detections = min_cost_matching(
                distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections
            )
            matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFINITY_COST, only_position=False
):
    """
    根据卡尔曼滤波得到的状态分布，验证成本矩阵中不可行的条目（门限）
    :param cost_matrix: ndarray
        N*M维的成本矩阵，其中N是轨迹索引的数量，M是检测索引的数量
        (i,j)是"tracks[track_indices[i]]"和"detections[detection_indices[j]]"之间的关联成本
    :param tracks: List[track.Track]
        当前时间步骤的预测轨迹列表
    :param detections: List[detection.Detection]
        当前时间步骤的检测列表
    :param track_indices: List[int]
        将"cost_matrix"中的行映射到"tracks"中的行的轨迹索引列表（见上文描述）
    :param detection_indices: List[int]
        将"cost_matrix"中的列映射到"detection"中的列的检测索引列表（见上文描述）
    :param gated_cost: Optional[float]
        成本矩阵中不可进行关联的条目被设定成这个值。默认是一个非常大的值
    :param only_position: Optional[bool]
        若为"True"在门控矩阵中只考虑状态分布的(x,y)位置。默认值为"False"
    :return: ndarray
        修改后的成本矩阵
    """
    assert not only_position
    gating_threshold = kalman_filter.chi2inv95[4]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = track.kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        if opt.MC:
            cost_matrix[row] = opt.MC_lambda * cost_matrix[row] + (1 - opt.MC_lambda) * gating_distance

    return cost_matrix
