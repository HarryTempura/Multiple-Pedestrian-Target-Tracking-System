import cv2


class Processor:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def resize(self, image):
        """
        规范输入图片的尺寸
        :param image:
        :return:
        """
        target_height, target_width = self.cfg.INPUT.SIZE
        image = cv2.resize(image, (target_width, target_height))

        return image
