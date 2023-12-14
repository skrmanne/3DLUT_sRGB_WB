"""
Metric calculation script which takes two folders as input and calculates the following metrics:
1. Mean Square Error
2. Mean Angle Error
3. Delta E 2000.
Along with Q1, Q2, and Q3 values for each metric.

Author: Sai Kumar Reddy Manne
"""
import os, sys
import cv2
from PIL import Image
import numpy as np
import colour
from tqdm import tqdm

def mse(img1, img2):
    """
    Calculate MSE between two images.
    """
    return np.mean((img1 - img2) ** 2)

def mae(img1, img2):
    """
    Calculate Mean Angle Error between two images
    """
    source = np.reshape(img1, [-1, 3]).astype(np.float32)
    target = np.reshape(img2, [-1, 3]).astype(np.float32)

    source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
    target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
    norm = source_norm * target_norm
    
    L = np.shape(norm)[0]
    inds = norm != 0
    
    # calculate cosine similarity
    angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
    angles[angles > 1] = 1

    # generate angle from the cosine similarities
    f = np.arccos(angles)
    f[np.isnan(f)] = 0
    f = f * 180 / np.pi

    return sum(f) / (L)

def delta(img1, img2):
    """
    Calculate Delta E between two images using colour-science library.
    """
    # Convert images to Lab color space
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)

    delta_E = colour.delta_E(lab1, lab2)
    return np.mean(delta_E)

output_folder = sys.argv[1]
label_folder  = sys.argv[2]

mse_list, mae_list, delta_list = [], [], []

for fname in tqdm(os.listdir(output_folder)):
    outname = os.path.join(output_folder, fname)
    labelname = os.path.join(label_folder, fname.split("_")[0] + ".JPG")

    #print(outname, labelname)
    output = cv2.imread(outname)
    label = cv2.imread(labelname)

    mae_list.append(mae(output, label))
    mse_list.append(mse(output, label))
    delta_list.append(delta(output, label))

mae_list, mse_list, delta_list = np.array(mae_list), np.array(mse_list), np.array(delta_list)

# MAE results
print("Mean MAE:{:.3f}, Q1:{:.3f}, Q2:{:.3f}, Q3:{:.3f}".format(np.mean(mae_list), np.quantile(mae_list, 0.25), np.quantile(mae_list, 0.5), np.quantile(mae_list, 0.75)))
# MSE results
print("Mean MSE::{:.3f}, Q1:{:.3f}, Q2:{:.3f}, Q3:{:.3f}".format(np.mean(mse_list), np.quantile(mse_list, 0.25), np.quantile(mse_list, 0.5), np.quantile(mse_list, 0.75)))
# Delta E results
print("Mean Delta E:{:.3f}, Q1:{:.3f}, Q2:{:.3f}, Q3:{:.3f}".format(np.mean(delta_list), np.quantile(delta_list, 0.25), np.quantile(delta_list, 0.5), np.quantile(delta_list, 0.75)))