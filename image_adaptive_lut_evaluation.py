"""
Evaluation script for the 3D LUT WB correction model.
Uses Cube dataset for evaluation, loads pretrained LUTs and classifier model for inference.

Author: Sai Kumar Reddy Manne
"""

import argparse
import time
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_x_contrastive import *
from datasets import *

# arg parser for the script
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=145, help="epoch to load the saved checkpoint")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10", help="directory of saved models")
opt = parser.parse_args()
opt.model_dir = opt.model_dir + '_' + opt.input_color_space

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()

# Define the LUTs, Trilinear Interpolation kernel, and the classifier model 
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
classifier = Classifier()
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])

LUT0.eval()
LUT1.eval()
LUT2.eval()
classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.eval()

if opt.input_color_space == 'sRGB':
    if opt.dataset_name == "cube":
        dataloader = DataLoader(
            CubeDataset(mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )
    else:
        raise NotImplementedError("Dataset not implemented")
else:
    raise NotImplementedError("Dataset not implemented")

def generator(img):
    """Generate a new image from the input image and loaded LUTs.
    Weights are generated from a lower resolution image (downsampled inside the classifier).
    Using the weights for adaptive LUT generation, high resolution output is estimated in trilinear interpolation.
    """
    pred = classifier(img).squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT,img)

    return combine_A


def visualize_result():
    """Saves a generated sample from the validation set.
    Calculates PSNR for the validation set."""
    out_dir = "images/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)

    avg_psnr = 0
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        img_name = batch["input_name"]

        fake_B = generator(real_A)
        save_image(fake_B, os.path.join(out_dir,"%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)

        # calc PSNR
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        avg_psnr += psnr

    print("Average PSNR:", avg_psnr / len(dataloader))

# ----------
#  evaluation
# ----------
visualize_result()
