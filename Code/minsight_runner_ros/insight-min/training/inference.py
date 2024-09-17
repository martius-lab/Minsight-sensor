import numpy as np
import cv2


import torch
import torchvision.transforms.functional as TF
import os
from torch.autograd import Variable
import sys

sys.path.append("/home/iandrussow/tactile_sensing/minsight_runner/insight-min/training")
from model import get_model
from utils import load_checkpoint
from dataset import Postprocessor


class MinsightSensor:

    def __init__(self, data_path, use_gpu, params):

        self.force_map = params.force_map
        self.params = params

        self.use_gpu = use_gpu

        self.X3 = cv2.imread(os.path.join(data_path, "NoContact_avg.png"))
        self.X4 = TF.to_tensor(cv2.imread((os.path.join(data_path, "gradient1.png"))))
        self.X5 = TF.to_tensor(cv2.imread((os.path.join(data_path, "gradient2.png"))))

        self.postprocessor = Postprocessor(use_gpu, params)

        # use gpu or not
        torch.cuda.empty_cache()
        print("use_gpu:{}".format(self.use_gpu))

        model, optimizer_ft, exp_lr_scheduler = get_model(self.params)

        checkpoint_path = os.path.join(self.params.working_dir, "checkpoint.pt")
        self.model = load_checkpoint(
            checkpoint_path, model, optimizer_ft, exp_lr_scheduler, self.use_gpu
        )

        if self.use_gpu:
            self.model = self.model.cuda()

        print("Using fully trained model")
        self.model.eval()

    def reset_no_contact_img(self, img):
        self.X3 = img

    def preprocess(self, image):

        X13 = TF.to_tensor(cv2.subtract(image, self.X3))

        if self.force_map:
            X = torch.cat([X13, self.X4[0][None, :, :], self.X5[0][None, :, :]], 0)
        else:
            X = torch.cat([X13], 0)

        return X

    def inference(self, input):

        # wrap them in Variable
        input = Variable(input)
        if self.use_gpu:
            input = input.cuda()

        # Fix channel conversion issue:
        input = torch.unsqueeze(input, 0)

        # forward
        outputs = self.model(input)

        return outputs
