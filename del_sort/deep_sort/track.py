import numpy as np

from del_sort.deep_sort.kalman_filter import KalmanFilter
from del_sort.deep_sort.nn_matching import cost
from main import opt


class TrackState(object):
    """用于表示每个目标轨迹的状态：暂定，确认和删除"""
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track(object):
    def __init__(
            self, detection, track_id, n_init, max_age, feature=None, score=None,
            feature_heads=None, feature_cloth=None, feature_pants=None, feature_shoes=None
    ):
        """
        一个单一的目标轨迹，其状态空间为(x,y,a,h)和相关的速度，其中(x,y)是检测框的中心，a是长宽比，h是高度
        :param detection: ndarray
            检测框的坐标(x,y,a,h)，中心位置(x,y)，长宽比a，高度h
        :param track_id: int
            一个唯一的轨迹标识符
        :param n_init: int
            轨迹被确认之前的连续检查次数。如果在第一个"n_int"中发生miss，则跟踪状态标记为删除
        :param max_age: int
            最大生命周期
        :param feature: Optional[ndarray]
            该轨迹的特征检测向量来源，如果不为None，这个特征将被添加到feature列表中
        :param score: 类型不明
            置信度得分
        """
        self.track_id = track_id
        self.hits = 1  # 跟踪器在跟踪目标时成功预测目标的帧数
        self.age = 1  # 跟踪器已经存在的帧数，包括当前帧
        self.time_since_update = 0  # 如果当前帧没有成功的预测这个值会增加
        self.state = TrackState.Tentative  # 目标进入待定状态
        self.features = []  # 用于存储提取的特征向量的空列表
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)
        if opt.LOCAL:
            self.feature_heads = []
            if feature_heads is not None:
                feature_heads /= np.linalg.norm(feature_heads)
                self.feature_heads.append(feature_heads)
            self.feature_cloth = []
            if feature_cloth is not None:
                feature_cloth /= np.linalg.norm(feature_cloth)
                self.feature_cloth.append(feature_cloth)
            self.feature_pants = []
            if feature_pants is not None:
                feature_pants /= np.linalg.norm(feature_pants)
                self.feature_pants.append(feature_pants)
            self.feature_shoes = []
            if feature_shoes is not None:
                feature_shoes /= np.linalg.norm(feature_shoes)
                self.feature_shoes.append(feature_shoes)
        self.scores = []
        if score is not None:
            self.scores.append(score)
        self._n_init = n_init
        self._max_age = max_age
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(detection)  # 初始化卡尔曼滤波器，返回状态均值和协方差矩阵

    def to_tlwh(self):
        """
        转换检测框格式
        :return: ndarray
            检测框
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """
        转换检测框格式
        :return: ndarray
            检测框
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self):
        """使用卡尔曼滤波预测步骤将状态分布传播到当前时间步骤"""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1  # 距离上次更新的时间

    @staticmethod
    def get_matrix(dict_frame_matrix, frame):  # 返回给定帧的矩阵
        eye = np.eye(3)
        matrix = dict_frame_matrix[frame]
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def camera_update(self, video, frame):  # 更新跟踪目标的摄像头状态
        dict_frame_matrix = opt.ecc[video]
        frame = str(int(frame))
        if frame in dict_frame_matrix:
            matrix = self.get_matrix(dict_frame_matrix, frame)
            x1, y1, x2, y2 = self.to_tlbr()
            x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
            x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
            w, h = x2_ - x1_, y2_ - y1_
            cx, cy = x1_ + w / 2, y1_ + h / 2
            self.mean[:4] = [cx, cy, w / h, h]

    def myAlpha(self, confidence):
        """
        置信度相关的动态加权
        :param confidence: float
            检测的置信度
        :return score: float
            检测得分
        """
        original_min = 0.6
        original_max = 1.0
        target_min = 0.1
        target_max = 0.3
        if confidence < original_min:
            return target_min
        scale_factor = (target_max - target_min) / (original_max - original_min)
        normalized_value = target_min + (confidence - original_min) * scale_factor
        score = float(normalized_value)
        return score

    def update(self, detection):
        """
        执行卡尔曼滤波测量更新步骤并更新特征缓存
        :param detection: detection.Detection
            相关的检测
        :return:
        """
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection.to_xyah(), detection.confidence
        )
        feature = detection.feature / np.linalg.norm(detection.feature)
        feature_heads = detection.feature_heads / np.linalg.norm(detection.feature_heads)
        feature_cloth = detection.feature_cloth / np.linalg.norm(detection.feature_cloth)
        feature_pants = detection.feature_pants / np.linalg.norm(detection.feature_pants)
        feature_shoes = detection.feature_shoes / np.linalg.norm(detection.feature_shoes)
        if opt.EMA:
            # 改成我自己的特征更新方式了
            alpha = self.myAlpha(detection.confidence)
            smooth_feat = (1 - alpha) * feature + alpha * self.features[-1]
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features = [smooth_feat]  # 全局
            if opt.LOCAL:
                if 1 - cost(self.feature_heads[-1], detection.feature_heads) * 1000 + detection.confidence > 1.2:
                    smooth_feat = opt.EMA_alpha * self.feature_heads[-1] + (1 - opt.EMA_alpha) * feature_heads
                    smooth_feat /= np.linalg.norm(smooth_feat)
                    self.feature_heads = [smooth_feat]  # 头部
                if 1 - cost(self.feature_cloth[-1], detection.feature_cloth) * 1000 + detection.confidence > 1.2:
                    smooth_feat = opt.EMA_alpha * self.feature_cloth[-1] + (1 - opt.EMA_alpha) * feature_cloth
                    smooth_feat /= np.linalg.norm(smooth_feat)
                    self.feature_cloth = [smooth_feat]  # 衣服
                if 1 - cost(self.feature_pants[-1], detection.feature_pants) * 1000 + detection.confidence > 1.2:
                    smooth_feat = opt.EMA_alpha * self.feature_pants[-1] + (1 - opt.EMA_alpha) * feature_pants
                    smooth_feat /= np.linalg.norm(smooth_feat)
                    self.feature_pants = [smooth_feat]  # 裤子
                if 1 - cost(self.feature_shoes[-1], detection.feature_shoes) * 1000 + detection.confidence > 1.2:
                    smooth_feat = opt.EMA_alpha * self.feature_shoes[-1] + (1 - opt.EMA_alpha) * feature_shoes
                    smooth_feat /= np.linalg.norm(smooth_feat)
                    self.feature_shoes = [smooth_feat]  # 脚部
            # 这是原来的更新方式
            # smooth_feat = opt.EMA_alpha * self.features[-1] + (1 - opt.EMA_alpha) * feature
            # smooth_feat /= np.linalg.norm(smooth_feat)
            # self.features = [smooth_feat]
        else:
            self.features.append(feature)
        # time_now = time.time()
        # self.feature_loader.update(feature, detection.confidence, time_now)
        # self.features.append(self.feature_loader.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """将轨迹标记为missed（在当前的时间步骤中没有关联）"""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """
        检查轨迹状态是否为待定
        :return: bool
            待定则返回True
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """
        检查轨迹状态是否为确定
        :return: bool
            确定则返回True
        """
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """
        检查轨迹状态是否为删除
        :return: bool
            删除则返回True
        """
        return self.state == TrackState.Deleted
