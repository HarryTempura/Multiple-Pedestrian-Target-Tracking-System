import warnings
import torch

from os.path import join

import del_sort.AFLink.config as cfg

from del_sort.AFLink.AppFreeLink import AFLink
from del_sort.AFLink.dataset import LinkData
from del_sort.AFLink.model import PostLinker
from del_sort.GSI.GSI import gaussian_interpolation
from del_sort.deep_sort.deep_sort_app import run
from del_sort.processor import Processor
from del_sort.sort_loader import SORTLoader
from main import opt

warnings.filterwarnings("ignore")


def start(video_file, detection_file):
    feat_extractor = Extractor(cfg)  # TODO: 这个位置定义特征提取器
    processor = Processor(cfg)
    data_loader = SORTLoader(feat_extractor, processor)

    if opt.AFLink:  # 启用AFLink
        model = PostLinker()
        model.load_state_dict(torch.load(opt.path_AFLink))
        dataset = LinkData('', '')

    for i, seq in enumerate(opt.sequences, start=1):  # enumerate是Python中的枚举函数
        print('Processing the {}th video {} ...'.format(i, seq))
        path_save = join(opt.dir_save, seq + '.txt')

        run(sequence_dir=video_file,
            detection_file=detection_file,
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=True,
            data_loader=data_loader)

        if not opt.res_save:
            print('The {}th video {} result not saved!'.format(i, seq))
        else:
            print('The {}th video {} result has been saved.'.format(i, seq))

        if opt.AFLink:  # 启用AFLink
            linker = AFLink(
                path_in=path_save,
                path_out=path_save,
                model=model,
                dataset=dataset,
                thrT=(0, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
                thrS=75,
                thrP=0.05  # 0.10 for CenterTrack, FairMOT, TransTrack.
            )
            linker.link()

        if opt.GSI:  # 启用高斯平滑
            gaussian_interpolation(path_in=path_save, path_out=path_save, interval=20, tau=10)
