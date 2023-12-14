"""
Main training script used to train the 3D LUT WB correction model.
Contains the following components:
    1. Generator: 3D LUT
    2. Classifier: Base, MobileNet or ResNet18
    3. Loss functions: MSE, Smoothness, Monotonicity, Contrastive
    4. Optimizer: Adam
    5. Scheduler: MultiStepLR
    6. Dataset: RenderedWB - set1
    7. Visualize: PSNR
    8. Save: Model checkpoints

Author: Sai Kumar Reddy Manne.
Date: 12/03/2021.
Class: CS 7180 Advanced Perception.
"""
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models_x_contrastive import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
torch.autograd.set_detect_anomaly(True)

def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("seed down")
fix_randomness(0)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=400, help="total number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--output_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10", help="path to save model")
parser.add_argument("--model", type=str, choices=["classifier", "mobile"], default="classifier", help="Model architecture")
parser.add_argument("--contrastive_wt", type=float, default=1.0, help="Contrastive loss weight")
parser.add_argument("--mix_random", type=int, default=0, help="Random weights for the contrastive loss")
parser.add_argument("--identity_LUT", action="store_true", help="Use identity LUT as the second LUT")
opt = parser.parse_args()

opt.output_dir = opt.output_dir + '_' + opt.input_color_space
print(opt)

os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_pixelwise = torch.nn.MSELoss()

# Initialize generator and discriminator
LUT0 = Generator3DLUT_identity()
if opt.identity_LUT:
    LUT1 = Generator3DLUT_identity()
else:
    LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()

if opt.model == "classifier":
    classifier = Classifier()
elif opt.model == "mobile":
    classifier = MobileNet()
else:
    raise NotImplementedError("Model [%s] is not implemented." % opt.model)

TV3 = TV_3D()
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

if opt.epoch != 0:
    # Load pretrained models
    LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.output_dir, opt.epoch))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.output_dir, opt.epoch)))
else:
    if opt.model == "classifier":
        # Initialize weights
        classifier.apply(weights_init_normal_classifier)
        if opt.identity_LUT:
            torch.nn.init.constant_(classifier.model[16].bias.data, 0.5)
        else:  
            torch.nn.init.constant_(classifier.model[16].bias.data, 1.0)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(),  LUT2.parameters()),lr=opt.lr, betas=(opt.b1, opt.b2))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[25,50,100,150,200,300], gamma=0.5)

# Dataset loader based on input options
if opt.input_color_space == 'sRGB':
    dataloader = DataLoader(
        ImageDataset_sRGB_AWB("../data/%s" % opt.dataset_name, mode = "train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    psnr_dataloader = DataLoader(
        ImageDataset_sRGB_AWB("../data/%s" % opt.dataset_name,  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
else:
    raise NotImplementedError("Dataset not implemented")

def generator_train(img):
    """
    img: input image
    combine_A: output image
    weights_norm: smoothness regularization

    Generate output image and smoothness regularization using adaptive LUT.
    """
    awb_wt = classifier(img)
    gen_A0 = LUT0(img)
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)

    weights_norm = torch.mean(awb_wt ** 2)

    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        combine_A[b,:,:,:] = awb_wt[b,0] * gen_A0[b,:,:,:] + awb_wt[b,1] * gen_A1[b,:,:,:]  + awb_wt[b,2] * gen_A2[b,:,:,:]

    return combine_A, weights_norm


def generator_eval(img):
    """
    img: input image
    combine_A: output image
    weights_norm: smoothness regularization

    Generate output image and smoothness regularization using adaptive LUT.
    Adaptive LUT is generated using the weights predicted by the classifier.
    Output High resolution image is generated using trilinear interpolation and adaptive LUT.
    Assumes batch size = 1 for validation.
    """
    # assumes batch size = 1 for validation.
    awb_wt = classifier(img).squeeze()
    
    A_LUT = awb_wt[0] * LUT0.LUT + awb_wt[1] * LUT1.LUT + awb_wt[2] * LUT2.LUT
    weights_norm = torch.mean(awb_wt ** 2)

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(A_LUT,img)

    return combine_A, weights_norm

def calculate_psnr():
    """
    PSNR calculator for the validation dataset.
    """
    classifier.eval()
    avg_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        fake_B, _ = generator_eval(real_A)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        avg_psnr += psnr

    return avg_psnr/ len(psnr_dataloader)


def visualize_result(epoch):
    """Saves a generated sample from the validation set"""
    classifier.eval()
    os.makedirs("images/%s/" % opt.output_dir +str(epoch), exist_ok=True)
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        img_name = batch["input_name"]
        fake_B, weights_norm = generator_eval(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        save_image(img_sample, "images/%s/%s/%s.jpg" % (opt.output_dir,epoch, img_name[0]+'_'+str(psnr)[:5]), nrow=3, normalize=False)

# ----------
#  Training
# ----------

prev_time = time.time()
max_psnr = 0
max_epoch = 0
avg_psnr = 0
for epoch in range(opt.epoch, opt.n_epochs):
    mse_avg = 0
    psnr_avg = 0
    classifier.train()
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        real_S = Variable(batch["A_exptS"].type(Tensor))
        real_T = Variable(batch["A_exptT"].type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        fake_B, weights_norm_B = generator_train(real_A)
        fake_S, weights_norm_S = generator_train(real_S)
        fake_T, weights_norm_T = generator_train(real_T)
        weights_norm = (weights_norm_B + weights_norm_T + weights_norm_S)/3.0

        # Pixel-wise loss
        awb_mse = criterion_pixelwise(fake_B, real_B)
        aux_mse = 0.5 * (criterion_pixelwise(fake_S, real_B) + criterion_pixelwise(fake_T, real_B))

        # Additional loss for training stronger color temperature insensitive features.
        if opt.mix_random == 1:
            rand_wt = torch.rand(3)
            fake_BST = (rand_wt[0] * fake_B + rand_wt[1] * fake_S + rand_wt[2] * fake_T)/torch.sum(rand_wt)
        else:
            fake_BST = (fake_B + fake_S + fake_T)/3.0

        contrastive_loss = opt.contrastive_wt * criterion_pixelwise(fake_BST, real_B)
        mse = awb_mse + aux_mse + contrastive_loss

        tv0, mn0 = TV3(LUT0)
        tv1, mn1 = TV3(LUT1)
        tv2, mn2 = TV3(LUT2)
        tv_cons = tv0 + tv1 + tv2
        mn_cons = mn0 + mn1 + mn2

        loss = mse + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons

        # calculate psnr only on the input image's AWB output
        psnr_avg += 10 * math.log10(1 / awb_mse.item())
        mse_avg += mse.item()

        loss.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [psnr: %f, tv: %f, wnorm: %f, mn: %f, contr: %f] ETA: %s"
            % (epoch,opt.n_epochs,i,len(dataloader),psnr_avg / (i+1),tv_cons, weights_norm, mn_cons, contrastive_loss, time_left,
            )
        )

    if epoch % 2 == 0:
        avg_psnr = calculate_psnr()
    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch
    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        LUTs = {"0": LUT0.state_dict(),"1": LUT1.state_dict(),"2": LUT2.state_dict()} #,"3": LUT3.state_dict(),"4": LUT4.state_dict()
        torch.save(LUTs, "saved_models/%s/LUTs_%d.pth" % (opt.output_dir, epoch))
        torch.save(classifier.state_dict(), "saved_models/%s/classifier_%d.pth" % (opt.output_dir, epoch))
        file = open('saved_models/%s/result.txt' % opt.output_dir,'a')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))
        file.close()


