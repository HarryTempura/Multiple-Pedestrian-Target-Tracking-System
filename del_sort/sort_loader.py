import os
import cv2
import numpy as np
import torch

from main import opt


class SORTLoader:
    def __init__(self, feat_extractor, processor):
        """
        检测信息加载器
        :param feat_extractor:
        :param processor:
        """
        self.feat_extractor = feat_extractor
        self.processor = processor

    def gather_sequence_info(self, video_file: str, detection_file: str) -> dict:
        """
        收集序列信息
        :param video_file: 视频文件路径
        :param detection_file: 检测信息路径
        :return:
        """
        groundtruth = None

        # 读取视频基本信息
        video = cv2.VideoCapture(video_file)

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        images = {}
        for i in range(frame_count - 1):
            ret, frame = video.read()
            if ret:
                images[i + 1] = frame

        # 加载检测结果
        detections = None
        if detection_file:
            detections = np.genfromtxt(detection_file, delimiter=',', dtype=float)
            detections = np.array(detections[detections[:, 0].argsort()])

        # 获取每一帧检测信息
        frame_detections = {}
        for detection in detections:
            frame = int(detection[0])
            if frame in frame_detections:
                frame_detections[frame].append(detection[2:7])
            else:
                frame_detections[frame] = detection[2:7]

        # TODO: 未实现
        # 构建包含所有信息的字典
        seq_info = {
            "sequence_name": os.path.basename(video_file),  # 视频文件名作为序列名称
            "image_filenames": images,  # 图像列表
            "detections": detections,  # 检测信息
            "groundtruth": groundtruth,  # 如果有真实信息，可以添加
            "image_size": (height, width),  # 图像尺寸
            "min_frame_idx": 1,  # 最小帧索引
            "max_frame_idx": frame_count,  # 最大帧索引
            "feature_dim": feature_dim,  # 特征维度
            "update_ms": fps  # 更新时间
        }

        return seq_info

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().byte().to(opt.device, dtype=torch.float32)

        return image_tensor

    def _get_feat(self, image):
        feat = self.feat_extractor(image).detach().cpu().numpy()

        return feat
