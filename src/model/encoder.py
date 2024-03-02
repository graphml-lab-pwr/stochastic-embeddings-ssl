from typing import Literal

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50


def get_encoder(
    backbone: Literal["resnet18", "resnet34", "resnet50"],
    dataset: Literal[
        "mnist", "cifar10", "cifar100", "imagenet", "tinyimagenet"
    ],
    compile: bool = False,
) -> tuple[nn.Module, int]:
    if backbone == "resnet18":
        encoder, encoder_out_dim = get_resnet18(dataset)
    elif backbone == "resnet34":
        encoder, encoder_out_dim = get_resnet34(dataset)
    elif backbone == "resnet50":
        encoder, encoder_out_dim = get_resnet50(dataset)
    else:
        raise ValueError("Backbone not recognized")
    if compile:
        encoder = torch.compile(encoder)  # type: ignore
    return encoder, encoder_out_dim


def get_resnet18(dataset: str) -> tuple[nn.Module, int]:
    encoder = resnet18()

    # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
    if dataset in ["cifar10", "cifar100"]:
        encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # for TinyImageNet, replace the first 7x7 conv with smaller 4x4 conv and remove the first maxpool
    elif dataset == "tinyimagenet":
        encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=4, stride=1, padding=1, bias=False
        )
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # for MNIST, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool and reduce the number of channels to 1 (greyscale)
    elif dataset == "mnist":
        encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    else:
        pass

    # replace classification fc layer of Resnet to obtain representations from the backbone
    encoder.fc = nn.Identity()

    encoder_out_dim = 512

    return encoder, encoder_out_dim


def get_resnet34(dataset: str) -> tuple[nn.Module, int]:
    encoder = resnet34()

    # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
    if dataset in ["cifar10", "cifar100"]:
        encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # for MNIST, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool and reduce the number of channels to 1 (greyscale)
    elif dataset == "mnist":
        encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    else:
        pass

    # replace classification fc layer of Resnet to obtain representations from the backbone
    encoder.fc = nn.Identity()

    encoder_out_dim = 512

    return encoder, encoder_out_dim


def get_resnet50(dataset: str) -> tuple[nn.Module, int]:
    encoder = resnet50()

    # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
    if dataset in ["cifar10", "cifar100"]:
        encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # for MNIST, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool and reduce the number of channels to 1 (greyscale)
    elif dataset == "mnist":
        encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    else:
        pass

    # replace classification fc layer of Resnet to obtain representations from the backbone
    encoder.fc = nn.Identity()

    encoder_out_dim = 2048

    return encoder, encoder_out_dim
