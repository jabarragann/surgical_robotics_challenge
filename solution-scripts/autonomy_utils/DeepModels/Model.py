## Standard Library
import os
import json

## External Libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional

from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from zipfile import ZipFile
from autonomy_utils.DeepModels.Dice import DiceScore, DiceLoss, DiceBCELoss

device = torch.device("cuda:0")


#
def get_train_transform():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            ToTensor(),
        ]
    )


# this converts 3 channel into 1 channel
def get_mask_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
        ]
    )


class imageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.folders = os.listdir(path)
        self.transforms = get_train_transform()
        self.mask_transforms = get_mask_transform()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], "images/")

        mask_folder = os.path.join(self.path, self.folders[idx], "masks/")
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
        img = io.imread(image_path)[:, :, :3].astype("float32")
        img = transform.resize(img, (128, 128))

        mask = self.get_mask(mask_folder, 128, 128).astype("float32")

        augmented = self.transforms(image=img, mask=mask)

        img = augmented["image"]
        mask = augmented["mask"]
        ########
        mask = mask[0].permute(2, 0, 1)
        mask = self.mask_transforms(
            mask
        )  # we need it behind the permute call since the transform only support 3 channel
        ########
        return (img, mask)

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
        mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))

        return mask_


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=True):
    """ """
    # Use batch normalization
    if useBN:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            # dim_in = 3, dim_out = 32
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1),
        )
    # No batch normalization
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
        )


## Upsampling


def upsample(ch_coarse, ch_fine):
    """ """
    return nn.Sequential(nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False), nn.ReLU())


# U-Net
class UNet(nn.Module):
    def __init__(self, n_classes, useBN=True):
        """
        Args:
            useBN (bool): Turn Batch Norm on or off
        """
        super(UNet, self).__init__()
        # Downgrade stages
        self.conv1 = add_conv_stage(3, 32, useBN=useBN)
        self.conv2 = add_conv_stage(32, 64, useBN=useBN)
        self.conv3 = add_conv_stage(64, 128, useBN=useBN)
        self.conv4 = add_conv_stage(128, 256, useBN=useBN)

        # Upgrade stages
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128, 64, useBN=useBN)
        self.conv1m = add_conv_stage(64, 32, useBN=useBN)
        # Maxpool
        self.max_pool = nn.MaxPool2d(2)
        # Upsample layers
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)
        # weight initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

        self.CONV = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)
        self.softmax2d = nn.Softmax2d()

    def forward(self, x):
        """
        Forward pass
        """
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))

        conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)
        out = self.CONV(conv1m_out)
        # out = out.squeeze()
        out = torch.sigmoid(out)

        return out
