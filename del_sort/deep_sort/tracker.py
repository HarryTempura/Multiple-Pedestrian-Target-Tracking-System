import numpy as np

from del_sort.deep_sort import linear_assignment, iou_matching
from del_sort.deep_sort.track import Track
from main import opt


class Tracker(object):
    def __init__(self, metric, update_ms, max_iou_distance=0.7, max_age=30, n_init=3):
        """
        多目标跟踪器
        :param metric: nn_matching.NearestNeighborDistanceMetric
            测量轨迹关联的距离度量
        :param max_iou_distance: float
            最大IoU距离
        :param max_age: int
            最大生命周期
        :param n_init: int
            轨迹被确认之前的连续检查次数。如果在第一个"n_int"中发生miss，则跟踪状态标记为删除
        """
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.n_init = n_init
        self.tracks = []  # 当前所有被跟踪目标的轨迹
        self._next_id = 1  # 轨迹ID编号

        if update_ms is None:
            self.max_age = max_age
        else:
            self.max_age = opt.agePower * update_ms

    def predict(self):
        """向前传播一个时间步长的轨迹状态分布,这个函数应该在每个时间步骤，即更新之前被调用一次"""
        for track in self.tracks:
            track.predict()

    def camera_update(self, video, frame):
        """
        更新每个跟踪目标的摄像头状态（咱们摄像头不动）
        :param video: 当前序列名称
        :param frame: 当前帧的索引
        :return:
        """
        for track in self.tracks:
            track.camera_update(video, frame)

    def update(self, detections):
        """
        执行测量更新和跟踪管理
        :param detections: List[detection.Detection]
            当前时间步骤的检测列表
        :return:
        """
        # 运行级联匹配
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # 更新轨迹集
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 更新距离度量
        if opt.LOCAL:
            active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
            features, targets = [], []
            features_a, features_b, features_c, features_d = [], [], [], []
            for track in self.tracks:
                if not track.is_confirmed():
                    continue
                features += track.features

                features_a += track.feature_heads
                features_b += track.feature_cloth
                features_c += track.feature_pants
                features_d += track.feature_shoes

                targets += [track.track_id for _ in track.features]
                if not opt.EMA:
                    track.features = []
            self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
            self.metric.partial_fit(np.asarray(features_a), np.asarray(targets), active_targets, 'a')
            self.metric.partial_fit(np.asarray(features_b), np.asarray(targets), active_targets, 'b')
            self.metric.partial_fit(np.asarray(features_c), np.asarray(targets), active_targets, 'c')
            self.metric.partial_fit(np.asarray(features_d), np.asarray(targets), active_targets, 'd')
        else:
            active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
            features, targets = [], []
            for track in self.tracks:
                if not track.is_confirmed():
                    continue
                features += track.features
                targets += [track.track_id for _ in track.features]
                if not opt.EMA:
                    track.features = []
            self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """
        将当前时间步骤的检测结果和之前跟踪的目标进行匹配
        :param detections: List[detection.Detection]
            当前时间步骤的检测列表
        :return: (List[(int, int)], List[int], List[int])
            一个三元组，包含匹配的检测和轨迹，未匹配的轨迹，未匹配的检测
        """

        def gated_metric(tracks, dets, track_indices, detection_indices):
            """
            门控矩阵
            :param tracks: List[track.Track]
                当前时间步骤的预测轨迹列表
            :param dets: List[detection.Detection]
                当前时间步骤的检测列表
            :param track_indices: List[int]
                应该用于匹配的tracks索引的列表
            :param detection_indices: List[int]
                应该用于匹配的detections索引的列表
            :return: ndarray
                检测的特征和Track的id之间的距离矩阵（代价矩阵）
            """
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices, detection_indices
            )

            return cost_matrix

        def local_metric(tracks, dets, track_indices, detection_indices):
            """
            局部特征的门控矩阵
            Args:
                tracks: 轨迹列表
                dets: 检测列表
                track_indices: 应该用于匹配的tracks索引的列表
                detection_indices: 应该用于匹配的dets索引的列表

            Returns:

            """
            features_a = np.array([dets[i].feature_heads for i in detection_indices])  # 头部
            features_b = np.array([dets[i].feature_cloth for i in detection_indices])  # 上衣
            features_c = np.array([dets[i].feature_pants for i in detection_indices])  # 裤子
            features_d = np.array([dets[i].feature_shoes for i in detection_indices])  # 脚部
            targets = np.array([tracks[i].track_id for i in track_indices])

            cost_matrix_a = self.metric.distance(features_a, targets, 'a')
            cost_matrix_b = self.metric.distance(features_b, targets, 'b')
            cost_matrix_c = self.metric.distance(features_c, targets, 'c')
            cost_matrix_d = self.metric.distance(features_d, targets, 'd')

            cost_matrix = \
                (1 - opt.CLOTH - opt.PANTS) / 2 * cost_matrix_a + opt.CLOTH * cost_matrix_b + \
                opt.PANTS * cost_matrix_c + (1 - opt.CLOTH - opt.PANTS) / 2 * cost_matrix_d

            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices, detection_indices
            )

            return cost_matrix

        # 将轨迹集分成确认和未确认的轨迹
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 使用外观特征关联确认的轨迹
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks
        )

        if opt.LOCAL:
            # 使用局部特征关联确认轨迹
            local_track_candidates = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
            unmatched_detections_l = unmatched_detections.copy()
            unmatched_detections = [k for k in unmatched_detections_l if detections[k].confidence < 0.6]
            high_confidence = [k for k in unmatched_detections_l if detections[k].confidence >= 0.6]
            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.matching_cascade(
                local_metric, opt.MAX_LOCAL, self.max_age, self.tracks,
                detections, local_track_candidates, unmatched_detections
            )

            # 使用IoU将剩余的轨迹与未确认的轨迹联系起来
            iou_track_candidates = \
                unconfirmed_tracks + [k for k in unmatched_tracks_b if self.tracks[k].time_since_update == 1]
            unmatched_tracks_b = [k for k in unmatched_tracks_b if self.tracks[k].time_since_update != 1]
            unmatched_detections += high_confidence
            matches_c, unmatched_tracks_c, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections
            )

            matches = matches_a + matches_b + matches_c
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b + unmatched_tracks_c))
        else:
            # 使用IoU将剩余的轨迹与未确认的轨迹联系起来
            iou_track_candidates = \
                unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections
            )

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):  # 在没有匹配到现有任何轨迹的情况下，为新的检测创建一个新的轨迹
        """
        在没有匹配到现有任何轨迹的情况下，为新的检测创建一个新的轨迹
        :param detection: detection.Detection
            当前时间步骤的检测序列中未匹配到轨迹的检测
        :return:
        """
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age, detection.feature, detection.confidence,
            detection.feature_heads, detection.feature_cloth, detection.feature_pants, detection.feature_shoes
        ))
        self._next_id += 1
