# Adaptive 3D Look Up Tables for Real Time White-Balance Correction
Auto white-balance (AWB) is an important step in the imaging pipeline to account for color cast in the image form the scene illumination. Although white-balance is expected to work without error, in cases where Auto-WB fails or manual WB setting is incorrect, we end up with an image using incorrect colors. While traditional white-balance is performed in the raw domain, correcting WB in sRGB space is an ill-posed problem due to non-linear operations followed by white-balancing in the camera pipeline. Recent approaches propose to pose this problem as an image-to-image translation task and learn correct WB images from incorrect input image. However, these models run deep neural networks to learn the outputs at a smaller resolution and perform different post-processing steps to get a high resolution output. On the other hand, related tasks such as image retouching and photo editing have seen significant improvements in real time processing due to the use of 3D Look Up Tables (LUTs). In this paper, we present an adaptive 3D LUT-based WB correction model that can generate high resolution WB corrected images in real time. Our model contains less than 600K parameters and takes less than 2.5 ms to process an image of 4K resolution using a v100 GPU.

## Installation
This repo requires CUDA, Python 3.8, and GCC 7.2.

The required packages can be easily installed through a conda environment as follows:

```conda create --name awb_lut --file requirements.txt```

Build cuda kernels for the trilinear interpolation using the below commands:

```
cd trilinear_cpp
sh setup.sh
```

## Dataset
We use datasets: Set1, Set2, and Cube+ from the RenderedWB dataset, which can be obtained [here](https://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html)

Modify the path name in ```datasets.py``` file based on the download folder.

## Training and Testing
Please use the sample train and test commands once the data is setup as described above:

### Train:
```python awb_lut_train_contrastive.py --dataset_name set1 --batch_size 64 --n_cpu 8```

### Test:
```python image_adaptive_lut_evaluation.py --epoch 199 --dataset_name cube --model_dir ckpt/```

### Acknowledgement:
This repository is largely based on the [3D_LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) repository.