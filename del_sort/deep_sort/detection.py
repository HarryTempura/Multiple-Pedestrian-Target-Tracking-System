import numpy as np


class Detection(object):
    def __init__(self, tlwh, confidence, feature):
        """
        检测图像信息
        :param tlwh:
        :param confidence:
        :param feature:
        """
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=float)

    def to_tlbr(self):
        """
        转换检测框格式为(左上，右下)
        :return:
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        转换检测框格式为(中心坐标，长宽比，高度)
        :return:
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
