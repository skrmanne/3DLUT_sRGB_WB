"""
Contains dataset classes for Rendered WB dataset set1 and set2, and Rendered Cube+ dataset.
Augments training data with random resized crop and random horizontal flip.
Supports generating auxiliaru inputs - Shade and Tungsten WB images for each input in training.

Author: Sai Kumar Reddy Manne
Date: 12/03/2021.
Class: CS 7180 Advanced Perception.
"""

import glob
import random
import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision_x_functional as TF_x

class ImageDataset_sRGB_AWB(Dataset):
    def __init__(self, root, mode="train"):
        """
        Modified dataset to retrieve S, T, and AWB ground truths for each input from a dataset.
        Works for Rendered WB dataset set1 and set2.
        """
        self.mode = mode
        
        if mode == "train":
            filename = "/scratch/manne.sa/data/AWB/renderedWB/set1/train.txt"
        else:
            if "set2" in root:
                filename = "/scratch/manne.sa/data/AWB/renderedWB/set2/set2.txt"
            elif "set1" in root:
                filename = "/scratch/manne.sa/data/AWB/renderedWB/set1/val.txt"
            else:
                raise NotImplementedError("Dataset not supported")

        with open(filename) as f:
            self.filenames = f.readlines()
            self.filenames = [x.strip() for x in self.filenames]

    def __getitem__(self, index):

        img_name = self.filenames[index].split("/")[-1]
        awb_name = self.filenames[index].replace("input", "label")
        awb_name = "_".join(awb_name.split("_")[:-2] + ["G_AS.png"])
        sgt_name = "_".join(self.filenames[index].split("_")[:-2] + ["S_AS.png"])
        tgt_name = "_".join(self.filenames[index].split("_")[:-2] + ["T_AS.png"])

        img_input = Image.open(self.filenames[index])
        img_exptC = Image.open(awb_name)
        img_exptS = Image.open(sgt_name)
        img_exptT = Image.open(tgt_name)

        # data augmentations for training batch
        if self.mode == "train":

            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            
            # random resized crop
            img_input = TF.resized_crop(img_input, i, j, h, w, (256, 256))
            img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (256, 256))
            img_exptS = TF.resized_crop(img_exptS, i, j, h, w, (256, 256))
            img_exptT = TF.resized_crop(img_exptT, i, j, h, w, (256, 256))

            # random horizontal flip
            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)
                img_exptS = TF.hflip(img_exptS)
                img_exptT = TF.hflip(img_exptT)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img_exptS = TF.to_tensor(img_exptS)
        img_exptT = TF.to_tensor(img_exptT)

        return {"A_input": img_input, "A_exptS": img_exptS, "A_exptT": img_exptT, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        # train with smaller number of images at max
        return min(5000, len(self.filenames))


class CubeDataset(Dataset):
    """
    Dataset for Rendered Cube+ Dataset.
    Primarily used for testing and only extracts input and ground truth images, no auxiliary inputs.
    """
    def __init__(self, mode="train"):
        self.mode = mode
        
        if mode == "train":
            filename = "/scratch/manne.sa/data/AWB/renderedWB/set1/train.txt"
        else:
            filename = "/scratch/manne.sa/data/AWB/renderedCube/cube.txt"

        with open(filename) as f:
            self.filenames = f.readlines()
            self.filenames = [x.strip() for x in self.filenames]

    def __getitem__(self, index):

        img_name = self.filenames[index].split("/")[-1]
        awb_name = self.filenames[index].replace("input", "label")
        awb_name = awb_name.split("_")[0] + ".JPG"

        img_input = Image.open(self.filenames[index])
        img_exptC = Image.open(awb_name)

        # Training augmentations
        if self.mode == "train":
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.resized_crop(img_input, i, j, h, w, (256, 256))
            img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (256, 256))

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        # train with smaller number of images at max
        return len(self.filenames)
