import torch.nn as nn
from torchvision import models

from torch.optim import lr_scheduler
from torch import optim


def resnet18_map_head():
    resnet18 = models.resnet18()
    model = nn.Sequential(
        nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
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


def get_model(params):

    model = resnet18_map_head()
    optimizer_ft = optim.Adam(model.parameters(), lr=params.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=1, gamma=params.gamma
    )

    return model, optimizer_ft, exp_lr_scheduler
