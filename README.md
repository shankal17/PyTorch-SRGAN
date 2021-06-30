# PyTorch-SRGAN

This is my implementation of the [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) paper to super-resolve faces.

![Sample Result 4](/Results/Results4.JPG)
## Overview

The goal of single-image super resolution is to estimate a high-resolution (HR) image from a corresponding low-resolution (LR) input image. This is an extremely challenging task since high-frequency details are not captured or represented very well in the LR input. The [SRGAN paper](https://arxiv.org/abs/1609.04802) proposes using Generative Adversarial Networks to accomplish this.

# Results
Here are some of the results from my trained model! The image on the left is a low-resolution image that is fed into the trained generator, and the image to the right is the resulting super-resolved image.

![Sample Result 1](/Results/Results_1.JPG)

![Sample Result 2](/Results/Results2.JPG)

![Sample Result 3](/Results/Results3.JPG)

![Sample Result 4](/Results/Results4.JPG)

As you can see, the super-resolved images look rather good. There are however some artifacts that show up at sharp discontinuities, especially in the eyes.

# Usage

## Dataset
Use whatever dataset you want, as long as it encompasses the image domain that you want to run inference on. 

This model was trained on the [CelebA-HQ Dataset](https://github.com/mazzzystar/make-CelebA-HQ) which consists of 30,000 1024x1024 images of celebrity faces as seen below. 

![CelebA-HQ Examples](/README_specific_imgs/CelebA-HQ_Examples.JPG)
I split them into a training and test dataset consisting of 29,000 and 1,000 images, respectively.

## Training
Configure hyperparameters in [configure_training.py](/configure_training.py). **BE SURE TO MATCH [DOWNSAMPLE FACTOR](https://github.com/shankal17/PyTorch-SRGAN/blob/main/configure_training.py#:~:text=config.DATASET.downsample_factor%20%3D%204) WITH THE NUMBER OF [PIXEL SHUFFLE BLOCKS](https://github.com/shankal17/PyTorch-SRGAN/blob/main/models/generator.py#:~:text=self.pixel_shuffle_1%20%3D%20PixelShuffleBlock,pixel_shuffle_2%20%3D%20PixelShuffleBlock(64)%20%23) YOU'RE USING. This is easy to forget and is hard to find the mistake.**

Then just run [train.py](/train.py). Note that pretraining just trains the generator using mean-squared error loss without a discriminator network and adversarial training implements the entire GAN training structure. I had the best luck with small batch sizes (4, 8, or 16) just because of the nature of GAN training. Anyway, expect this to take a month to train.


## Getting Started

Once you have the code, set up a virtual environment if you would like and install the necessary libraries by running the command below.
```bat
pip install -r /path/to/requirements.txt
```
