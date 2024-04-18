import os
import cv2
import numpy as np

from del_sort.application_util import preprocessing, visualization
from del_sort.deep_sort import nn_matching
from del_sort.deep_sort.detection import Detection
from del_sort.deep_sort.tracker import Tracker
from main import opt


def gather_sequence_info(sequence_dir, detection_file):
    """
    收集序列信息，图像文件名、检测结果、地面实况（如果有）。
    :param sequence_dir: str
        MOTChallenge序列目录的路径
    :param detection_file: str
        检测文件路径
    :return: Dict
        一个包含序列各种信息的字典
    """
    # 构建图像文件路径
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f) for f in os.listdir(image_dir)}

    # 构建地面实况文件路径
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    # 加载检测结果（如果提供）
    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)

    # 加载地面实况（如果存在）
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    # 获取第一帧图像的大小
    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    # 获取帧号范围
    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    # 获取帧率信息（如果提供）
    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)
        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    # 获取特征维度
    feature_dim = detections.shape[1] - 10 if detections is not None else 0

    # 构建包含所有信息的字典
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }

    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """
    从原始检测矩阵中创建给定的帧的索引的检测
    :param detection_mat: ndarray
        检测矩阵
    :param frame_idx: int
        视频中帧的编号
    :param min_height: Optional[int]
        最小检测框高度，用于过滤较小的检测框
    :return: List[tracker.Detection]
        一个检测列表
    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx  # 创建切片掩码 用以保留对应帧范围内的元素
    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[7:2055]  # 添加局部特征
        feature_heads, feature_cloth, feature_pants, feature_shoes = \
            row[2055:4103], row[4103:6151], row[6151:8199], row[8199:]
        if bbox[3] < min_height:
            continue
        # detection_list.append(Detection(bbox, confidence, feature))
        detection_list.append(Detection(
            bbox, confidence, feature, feature_heads, feature_cloth, feature_pants, feature_shoes
        ))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence, nms_max_overlap, min_detection_height,
        max_cosine_distance, nn_budget, display, data_loader):
    """
    在一个特定的序列上运行多目标检测器
    :param sequence_dir: str
        序列文件夹的路径
    :param detection_file: str
        检测结果文件夹路径
    :param output_file: str
        跟踪结果保存路径
    :param min_confidence: float
        最小检测置信度
    :param nms_max_overlap: float
        非极大值抑制的重叠阈值
    :param min_detection_height: int
        最小检测框高度
    :param max_cosine_distance: float
        最大余弦距离阈值
    :param nn_budget: Optional[int]
        外观描述符库的大小限制
    :param display: bool
        是否可视化
    :param data_loader: Loader
        加载器
    :return: 将结果直接写入文件 没有返回
    """
    # 在这个位置替换成我自己的数据集信息
    # seq_info = gather_sequence_info(sequence_dir, detection_file)
    seq_info = data_loader.gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    # tracker = Tracker(metric, int(seq_info["update_ms"]))
    tracker = Tracker(metric, 1)  # TODO:轨迹的最大生命周期
    results = []

    def frame_callback(vis, frame_idx):
        """
        在视频序列中检测和跟踪目标
        :param vis: application_util.visualization.NoVisualization()
            可视化对象
        :param frame_idx: int
            当前帧的索引
        :return:
        """
        # print("Processing frame %05d" % frame_idx)
        # 加载图像并产生检测结果
        detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # 运行非极大值抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # 更新跟踪器
        if opt.ECC:
            tracker.camera_update(sequence_dir.split('/')[-1], frame_idx)
        tracker.predict()
        tracker.update(detections)

        # 更新可视化
        if display:
            image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            # vis.draw_detections(detections)  # 展示检测框
            vis.draw_trackers(tracker.tracks)  # 展示跟踪框

        # 存储结果
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # 运行跟踪器
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # 存储结果
    if opt.res_save:
        f = open(output_file, 'w')
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (row[0], row[1], row[2], row[3], row[4], row[5]), file=f)


def bool_string(input_string):  # 将字符串转换成布尔值
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return input_string == "True"
