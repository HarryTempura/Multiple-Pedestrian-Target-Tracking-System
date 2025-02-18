import os
from glob import glob

from . import DATASET_REGISTRY
from .bases import ImageDataset

__all__ = ['PRAI', ]


@DATASET_REGISTRY.register()
class PRAI(ImageDataset):
    dataset_dir = "PRAI-1581"
    dataset_name = 'prai'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir, 'images')

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        img_paths = glob(os.path.join(train_path, "*.jpg"))
        for img_path in img_paths:
            split_path = img_path.split('/')
            img_info = split_path[-1].split('_')
            pid = self.dataset_name + "_" + img_info[0]
            camid = self.dataset_name + "_" + img_info[1]
            data.append([img_path, pid, camid])
        return data
