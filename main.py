#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from pages.login import LoginWindow
from utils import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将 ROOT 添加到 PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对


def parse_opt():
    """
    解析 YOLOv5 检测的命令行参数，设置推理选项和模型配置。
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default=ROOT / "yolov5/yolov5s.pt", help="模型路径或 triton URL"
    )
    parser.add_argument(
        "--data", type=str, default=ROOT / "yolov5/data/coco128.yaml", help="（可选）dataset.yaml 路径"
    )
    parser.add_argument(
        "--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="推理大小 h，w"
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=1000, help="每张图像的最大检测次数")
    parser.add_argument("--device", default="", help="CUDA 设备，即 0 或 0、1、2、3 或 CPU")
    parser.add_argument("--view-img", action="store_true", help="显示结果")
    parser.add_argument("--save-txt", action="store_true", help="将结果保存到 *.txt")
    parser.add_argument("--save-csv", action="store_true", help="以 CSV 格式保存结果")
    parser.add_argument("--save-conf", action="store_true", help="在 --save-txt 标签中保存置信度")
    parser.add_argument("--save-crop", action="store_true", help="保存裁剪的预测框")
    parser.add_argument("--nosave", action="store_true", help="不保存图像/视频")
    parser.add_argument("--classes", nargs="+", type=int, default=0, help="按类筛选：--类 0 或 --类 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="与类别无关的 NMS")
    parser.add_argument("--augment", action="store_true", help="增强推理")
    parser.add_argument("--visualize", action="store_true", help="可视化要素")
    parser.add_argument("--update", action="store_true", help="更新全部 models")
    parser.add_argument("--project", default=ROOT / "yolov5/runs/detect", help="将结果保存到 project/name")
    parser.add_argument("--name", default="exp", help="将结果保存到 project/name")
    parser.add_argument("--exist-ok", action="store_true", help="现有 project/name 正常，不递增")
    parser.add_argument("--line-thickness", default=100, type=int, help="边界框厚度（像素）")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="隐藏标签")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="隐藏置信度")
    parser.add_argument("--half", action="store_true", help="使用 FP16 半精度推理")
    parser.add_argument("--dnn", action="store_true", help="使用 OpenCV DNN 进行 ONNX 推理")
    parser.add_argument("--vid-stride", type=int, default=1, help="视频帧速率步幅")

    parser.add_argument("--NSA", action="store_true", help="NSA 卡尔曼滤波器")
    parser.add_argument("--EMA", action="store_true", help="EMA 特征更新测略")
    parser.add_argument("--MC", action="store_true", help="与外观和运动成本相匹配")
    parser.add_argument("--woC", action="store_true", help="将匹配级联替换为香草（普通）匹配")
    parser.add_argument("--AFLink", action="store_true", help="无外观链接")
    parser.add_argument("--GSI", action="store_true", help="高斯平滑插值")

    opts = parser.parse_args()
    opts.imgsz *= 2 if len(opts.imgsz) == 1 else 1  # expand
    return opts


opt = parse_opt()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = LoginWindow()
    window.show()
    logger.info('启动成功')

    app.exec()
