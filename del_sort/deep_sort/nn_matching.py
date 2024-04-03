import numpy as np


def _pdist(a, b):
    """
    计算"a"和"b"中点之间的成对平方距离
    :param a: array_like
        一个由N个维度为M的样本组成的N*M矩阵
    :param b: array_like
        一个由L个维度为M的样本组成的L*M矩阵
    :return: ndarray
        一个大小为len(a),len(b)的矩阵，使得元素(i,j)包含a[i]和b[j]之间的平方距离
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """
    计算"a"和"b"中各点之间的成对余弦距离
    :param a: array_like
        一个由N个维度为M的样本组成的N*M矩阵
    :param b: array_like
        一个由L个维度为M的样本组成的L*M矩阵
    :param data_is_normalized: Optional[bool]
        如果是True，则假设a和b中的行是单位长度的向量 否则，a和b被明确的归一化为长度1
    :return: ndarray
        大小为len(a),len(b)的矩阵，使得元素(i,j)包含a[i]和b[j]之间的平方距离
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """
    计算最近邻距离的辅助函数，使得欧氏距离计算样本点矩阵x和查询点矩阵y之间的距离
    :param x: ndarray
        一个由N个行向量（样本点）组成的矩阵
    :param y: ndarray
        一个由M个行向量（样本点）组成的矩阵
    :return: ndarray
        一个长度为M的向量，其中包含每个条目的最小欧氏距离，即与样本的最小欧氏距离
    """
    # x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
    # y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """
    计算欧式距离的辅助函数
    :param x: ndarray
        一个由N个行向量（样本点）组成的矩阵
    :param y: ndarray
        一个由M个行向量（样本点）组成的矩阵
    :return: ndarray
        一个长度为M的向量，其中包含每个条目的最小余弦距离
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):
        """
        一个最近邻距离度量，对于每个目标，返回与迄今为止观察到的任何样本最近的距离
        :param metric: str
            表示距离度量方式，只能是字符串"euclidean"或"cosine"
        :param matching_threshold: float
            表示匹配阈值。距离大于该阈值的样本将被视为无效匹配
        :param budget: Optional[int]
            表示每个类别最多保留的样本数。当到达该预算时，将删除最旧的样本。
        """
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}  #: Dict[int -> List[ndarray]] 该字典将存储检测到的样本数据
        self.samples_a, self.samples_b, self.samples_c, self.samples_d = {}, {}, {}, {}

    def partial_fit(self, features, targets, active_targets, local=None):
        """
        用新数据更新距离度量
        :param features: ndarray
            N*M的矩阵，其中N是特征数量，M是每个特征的维数
        :param targets: ndarray
            与features中每个特征对应的目标标识符的整数数组
        :param active_targets: List[int]
            一个包含当前场景中存在的目标标识符的列表
        :param local: string
            表示位置
        :return:
        """
        if local == 'a':
            for feature, target in zip(features, targets):
                self.samples_a.setdefault(target, []).append(feature)
                if self.budget is not None:
                    self.samples_a[target] = self.samples_a[target][-self.budget:]
            self.samples_a = {k: self.samples_a[k] for k in active_targets}
        elif local == 'b':
            for feature, target in zip(features, targets):
                self.samples_b.setdefault(target, []).append(feature)
                if self.budget is not None:
                    self.samples_b[target] = self.samples_b[target][-self.budget:]
            self.samples_b = {k: self.samples_b[k] for k in active_targets}
        elif local == 'c':
            for feature, target in zip(features, targets):
                self.samples_c.setdefault(target, []).append(feature)
                if self.budget is not None:
                    self.samples_c[target] = self.samples_c[target][-self.budget:]
            self.samples_c = {k: self.samples_c[k] for k in active_targets}
        elif local == 'd':
            for feature, target in zip(features, targets):
                self.samples_d.setdefault(target, []).append(feature)
                if self.budget is not None:
                    self.samples_d[target] = self.samples_d[target][-self.budget:]
            self.samples_d = {k: self.samples_d[k] for k in active_targets}
        else:
            for feature, target in zip(features, targets):
                self.samples.setdefault(target, []).append(feature)
                if self.budget is not None:
                    self.samples[target] = self.samples[target][-self.budget:]
            self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets, local=None):
        """
        计算特征和目标之间的距离
        :param features: ndarray
            一个由N个维度为M的特征组成的N*M矩阵
        :param targets: List[int]
            一个与给定特征相匹配的目标列表
        :param local: string
            表示位置
        :return: ndarray
            返回一个形状为(len(targets), len(features))的成本矩阵，其中元素(i,j)包含target[i]和features[j]之间最近的平方距离
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            if local == 'a':
                cost_matrix[i, :] = self._metric(self.samples_a[target], features)
            elif local == 'b':
                cost_matrix[i, :] = self._metric(self.samples_b[target], features)
            elif local == 'c':
                cost_matrix[i, :] = self._metric(self.samples_c[target], features)
            elif local == 'd':
                cost_matrix[i, :] = self._metric(self.samples_d[target], features)
            else:
                cost_matrix[i, :] = self._metric(self.samples[target], features)

        return cost_matrix


def cost(x, y):
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    return _nn_cosine_distance(x, y)
