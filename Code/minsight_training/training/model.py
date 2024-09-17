from typing import Callable, Optional

import numpy as np
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from sympy import symbols
from sympy.solvers.diophantine import diophantine
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models


def torch_resnet18():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_ftr = model.fc.in_features
    out_ftr = 6
    model.fc = nn.Linear(in_ftr, out_ftr, bias=True)
    return model


def resnet18_map_head_min():
    resnet18 = models.resnet18()
    model = nn.Sequential(
        nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
        resnet18.bn1,
        resnet18.relu,
        resnet18.maxpool,
        resnet18.layer1,
        resnet18.layer2,
        nn.ConvTranspose2d(
            128,
            128,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(2, 1),
            output_padding=(1, 0),
        ),
        nn.Conv2d(128, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0)),
    )
    return model


def resnet18_map_head_resized(input_size=None):
    resnet18 = models.resnet18()

    input_size = np.ceil(np.divide(input_size, 8)).astype(int)
    stride = (3, 2)

    def calc_params(input_size, stride):
        k_x, p_x = symbols("k_x,p_x", integer=True, positive=True)
        k_y, p_y = symbols("k_y,p_y", integer=True, positive=True)
        t_0 = symbols("t_0", integer=True)
        k_x, p_x = diophantine(
            (input_size[0] - 1) * stride[0] + k_x - 2 * p_x - 64
        ).pop()
        k_y, p_y = diophantine(
            (input_size[1] - 1) * stride[1] + k_y - 2 * p_y - 64
        ).pop()
        return k_x.subs(t_0, 1), k_y.subs(t_0, 1), p_x.subs(t_0, 1), p_y.subs(t_0, 1)

    k_x, k_y, p_x, p_y = calc_params(input_size, stride)

    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        resnet18.bn1,
        resnet18.relu,
        resnet18.maxpool,
        resnet18.layer1,
        resnet18.layer2,
        nn.ConvTranspose2d(
            128,
            128,
            kernel_size=(k_x, k_y),
            stride=(stride[0], stride[1]),
            padding=(p_x, p_y),
        ),
        nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    )
    return model


def torch_resnet18_nomap():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_ftr = model.fc.in_features
    out_ftr = 3 * 1350
    model.fc = nn.Linear(in_ftr, out_ftr, bias=True)
    return model


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )
        self.out_channels = out_planes


def torch_mobilenet():
    model = models.mobilenet_v2(num_classes=6)
    model.features[0] = ConvBNActivation(3, 32, stride=2, norm_layer=nn.BatchNorm2d)
    return model


def torch_squeezenet():
    # Parameters from original repo: https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/solver.prototxt
    model = models.squeezenet1_1()
    model.classifier[1] = nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1))
    model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))
    return model


def torch_efficientnet():
    model = EfficientNet.from_name("efficientnet-b0")
    in_ftr = model._fc.in_features
    out_ftr = 6
    model._fc = nn.Linear(in_ftr, out_ftr, bias=True)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 32, kernel_size=(3, 3), stride=(2, 2), image_size=(308, 410), bias=False
    )
    return model


def ConvMixer(h, depth, input_channels=3, kernel_size=9, patch_size=7, n_classes=6):
    Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
    Residual = type("Residual", (Seq,), {"forward": lambda self, x: self[0](x) + x})

    return Seq(
        ActBn(nn.Conv2d(input_channels, h, patch_size, stride=patch_size)),
        *[
            Seq(
                Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))),
                ActBn(nn.Conv2d(h, h, 1)),
            )
            for i in range(depth)
        ],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(h, n_classes)
    )


def ConvMixerForceMap(h, depth, input_channels=3, kernel_size=9, patch_size=7):
    Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
    Residual = type("Residual", (Seq,), {"forward": lambda self, x: self[0](x) + x})

    return Seq(
        ActBn(nn.Conv2d(input_channels, h, patch_size, stride=patch_size)),
        *[
            Seq(
                Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))),
                ActBn(nn.Conv2d(h, h, 1)),
            )
            for i in range(depth)
        ],
        nn.ConvTranspose2d(
            100,
            3,
            kernel_size=(5, 7),
            stride=(1, 1),
            dilation=(5, 1),
            output_padding=(0, 0),
        )
    )


def get_model(params):

    if params.model == "mobilenet":
        model = torch_mobilenet()

        optimizer_ft = optim.RMSprop(
            model.parameters(),
            lr=params.lr,
            alpha=0.9,
            momentum=0.9,
            weight_decay=0.00004,
        )
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=1, gamma=params.gamma
        )

    elif params.model == "squeezenet":
        model = torch_squeezenet()

        optimizer_ft = optim.RMSprop(
            model.parameters(),
            lr=params.lr,
            alpha=0.9,
            momentum=0.9,
            weight_decay=0.0002,
        )
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=1, gamma=params.gamma
        )

    elif params.model == "resnet":
        if params.force_map == True:
            if hasattr(params, "input_size"):
                model = resnet18_map_head_resized(params.input_size)
            elif params.flat_output == True:
                model = torch_resnet18_nomap()
            else:
                model = resnet18_map_head_min()
            print("Doing force map estimation")
        else:
            model = torch_resnet18()
        optimizer_ft = optim.Adam(model.parameters(), lr=params.lr)
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=1, gamma=params.gamma
        )

    elif params.model == "efficientnet":
        model = torch_efficientnet()
        optimizer_ft = optim.Adam(model.parameters(), lr=params.lr)
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=1, gamma=params.gamma
        )

    elif params.model == "convmixer":
        if params.force_map == False:
            model = ConvMixer(100, 6, 3, 9, 7, 6)
        else:
            model = ConvMixerForceMap(100, 6, 3, 9, 7)
        optimizer_ft = optim.AdamW(model.parameters(), lr=params.lr)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer_ft, T_max=params.total_epochs
        )

    return model, optimizer_ft, exp_lr_scheduler
