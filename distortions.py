import torchvision.transforms as transforms

import numpy as np
import cv2


def pil2array(image):
    if image.mode == 'L':
        return np.array(image.getdata(), np.uint8).reshape(image.size + (1,))
    elif image.mode == 'RGB':
        return np.array(image.getdata(), np.uint8).reshape(image.size + (3,))


class Distortion(object):
    def __init__(self, distortion_type, image_size, level, levels_dict):
        self.type = distortion_type
        self.level = levels_dict[self.type][level - 1]
        self.transform = None
        self.image_size = image_size
        self.to_numpy = transforms.Lambda(
            lambda image: pil2array(image)
        )
        self.reshape = transforms.Lambda(
            lambda image: image.reshape(self.image_size)
        )

    def gaussian_noise(self):
        gaussian_noise = np.random.normal(0, self.level, self.image_size)
        return transforms.Lambda(lambda image: np.add(image, gaussian_noise))

    def motion_blur(self):
        kernel_motion_blur = np.zeros((self.level, self.level))
        kernel_motion_blur[int((self.level - 1)/2), :] = np.ones(self.level)
        kernel_motion_blur /= self.level
        return transforms.Lambda(
            lambda image: cv2.filter2D(image, -1, kernel_motion_blur)
        )

    def combination(self):
        reshape = transforms.Lambda(
            lambda image: image.reshape(self.image_size)
        )
        return transforms.Compose(
            [self.motion_blur(),
             reshape,
             self.gaussian_noise()]
        )

    def none(self):
        return transforms.Lambda(
            lambda image: image
        )

    def trans(self):
        if self.type == 'motion_blur':
            self.transform = transforms.Compose(
                [self.to_numpy,
                 self.motion_blur(),
                 self.reshape]
            )
        elif self.type == 'gaussian_noise':
            self.transform = transforms.Compose(
                [self.to_numpy,
                 self.gaussian_noise(),
                 self.reshape]
            )
        elif self.type == 'combination':
            self.transform = transforms.Compose(
                [self.to_numpy,
                 self.combination(),
                 self.reshape]
            )
        elif self.type == 'none':
            self.transform = transforms.Compose(
                [self.to_numpy,
                 self.none(),
                 self.reshape]
            )
        return self.transform
