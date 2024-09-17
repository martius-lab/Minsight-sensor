import torch
from skimage import io
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

import matplotlib.tri as tri
import cv2


def resize(img, scale):
    if scale != 100:
        width = int(img.shape[1] * scale / 100)
        height = int(img.shape[0] * scale / 100)
        dim = (width, height)
        down_size = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return cv2.resize(down_size, (410, 308), interpolation=cv2.INTER_AREA)
    else:
        return img


def get_max_min(dir):

    max_min_dict = np.load(
        os.path.join(dir, "groundtruth_max_min.npy"), allow_pickle=True
    ).item()
    return max_min_dict.get("max"), max_min_dict.get("min")


class Postprocessor:

    def __init__(self, use_gpu, params):

        self.rescale = params.rescale
        self.data_dir = os.path.join(params.data_path, params.sensor)

        X3 = cv2.imread(os.path.join(self.data_dir, "NoContact_avg.png"))
        self.X3 = TF.to_tensor(X3)

        self.skeleton_surface = np.load(os.path.join(self.data_dir, "beam_nodes.npy"))
        self.indentation_info = torch.from_numpy(
            np.load(os.path.join(self.data_dir, "indentation_info.npy"))
        )
        self.col_ind = torch.from_numpy(
            np.load(os.path.join(self.data_dir, "column_idx_mapping.npy"))
        )

        if params.force_map == True:
            self.scale = torch.tensor(1.0 / 4096.0)
            self.const = torch.tensor(0.0)

        else:
            max_, min_ = get_max_min(self.data_dir)
            self.max_ = np.hstack((max_[12:15], max_[6:9]))
            self.min_ = np.hstack((min_[12:15], min_[6:9]))
            self.scale = torch.from_numpy((self.max_ - self.min_)).float()
            self.const = torch.from_numpy(self.min_).float()
        if use_gpu:
            self.scale = self.scale.cuda()
            self.const = self.const.cuda()
            self.X3 = self.X3.cuda()

    def undo_rescale(self, output):
        if self.rescale:
            return torch.add(output * self.scale, self.const)
        else:
            return output

    def preprocess_for_loss(self, outputs, labels):
        if self.force_map:
            raise NotImplementedError()

        return self.labels_weight * (outputs), self.labels_weight * (labels)

    def transform(self, force):
        if torch.is_tensor(force):
            force_ = torch.zeros((3, 1600))
        else:
            force_ = np.zeros((3, 1600))
        force_[0] = force[0].reshape((40, 40)).flatten()
        force_[1] = force[1].reshape((40, 40)).flatten()
        force_[2] = force[2].reshape((40, 40)).flatten()

        return force_
