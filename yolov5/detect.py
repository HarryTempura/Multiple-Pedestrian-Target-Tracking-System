# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
对图像、视频、目录、球体、YouTube、网络摄像头、流等运行 YOLOv5 检测推理。

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
    increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,
)
from yolov5.utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将 ROOT 添加到 PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对


@smart_inference_mode()
def run(
        weights=ROOT / "yolov5s.pt",  # model path or triton URL
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # 保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # 下载

    # 目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增量运行
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 制作目录

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 数据加载器
    bs = 1  # 批量大小
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 定义 CSV 文件的路径
        csv_path = save_dir / "predictions.csv"

        # 创建或追加到 CSV 文件
        def write_to_csv(image_name, prediction, confidence):
            """
            将图像的预测数据写入 CSV 文件，如果文件存在，则追加该文件。
            :param image_name:
            :param prediction:
            :param confidence:
            :return:
            """
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # 过程预测
        for i, det in enumerate(pred):  # 每张图片
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # 到路径
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 WHWH
            imc = im0.copy() if save_crop else im0  # 对于save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 将框从 img_size 重新调整为 im0 大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # 写入文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # 将 bbox 添加到图像
                        c = int(cls)  # Integer 类
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # 流式传输结果
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1毫秒

            # 保存结果（带有检测的图像）
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印时间（仅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型（用于修复 SourceChangeWarning）


def parse_opt(video_path=(ROOT / "data/images")):
    """
    解析 YOLOv5 检测的命令行参数，设置推理选项和模型配置。
    :param video_path: 默认视频路径
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="模型路径或 triton URL")
    parser.add_argument("--source", type=str, default=video_path, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="（可选）dataset.yaml 路径")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="推理大小 h，w")
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
    parser.add_argument("--classes", nargs="+", type=int, help="按类筛选：--类 0 或 --类 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="与类别无关的 NMS")
    parser.add_argument("--augment", action="store_true", help="增强推理")
    parser.add_argument("--visualize", action="store_true", help="可视化要素")
    parser.add_argument("--update", action="store_true", help="更新全部 models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="将结果保存到 project/name")
    parser.add_argument("--name", default="exp", help="将结果保存到 project/name")
    parser.add_argument("--exist-ok", action="store_true", help="现有 project/name 正常，不递增")
    parser.add_argument("--line-thickness", default=3, type=int, help="边界框厚度（像素）")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="隐藏标签")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="隐藏置信度")
    parser.add_argument("--half", action="store_true", help="使用 FP16 半精度推理")
    parser.add_argument("--dnn", action="store_true", help="使用 OpenCV DNN 进行 ONNX 推理")
    parser.add_argument("--vid-stride", type=int, default=1, help="视频帧速率步幅")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def start(video_path):
    opt = parse_opt(video_path)
    main(opt)


def main(opt):
    """
    使用给定选项执行 YOLOv5 模型推理，在运行模型之前检查需求。
    :param opt:
    :return:
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
