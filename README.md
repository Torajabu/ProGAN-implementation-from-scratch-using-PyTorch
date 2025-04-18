# ProGAN-implementation-from-scratch-using-PyTorch
uses  CIFAR-10 as our dataset
![progan](https://github.com/Torajabu/ProGAN-implementation-from-scratch-using-PyTorch/blob/main/Screenshot%20from%202025-04-18%2013-35-58.png)

# Progressive Growing GAN PyTorch Implementation

This repository contains a PyTorch implementation of Progressive Growing of GANs (PGGAN) trained on the CIFAR-10 dataset. The implementation follows the progressive training methodology introduced in the paper ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://arxiv.org/abs/1710.10196) by Karras et al.

## Overview

Progressive Growing GAN is a technique for training generative adversarial networks that starts with low-resolution images and incrementally adds layers to increase resolution throughout training. This approach:

- Enables more stable training
- Provides better quality results at higher resolutions
- Reduces training time significantly compared to traditional GAN approaches

This implementation:
- Progressively grows from 4×4 to 16×16 resolution
- Uses WGAN-GP (Wasserstein GAN with gradient penalty) loss function
- Implements fade-in mechanism for smooth transitions between resolutions
- Trains on the CIFAR-10 dataset

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- tqdm

Install dependencies:
```bash
pip install torch torchvision tqdm
```

## Model Architecture

### Generator
- Starts with a dense layer that projects random noise to 4×4 spatial resolution
- Progressively adds upsampling layers to increase resolution
- Uses fade-in mechanism for smooth transitions between resolutions
- Output is normalized with tanh activation

### Discriminator
- Mirror architecture of the generator but in reverse
- Progressively downsamples from higher to lower resolution
- Uses leaky ReLU activation and convolution layers
- Implements WGAN-GP loss for improved stability

## Training Parameters

- Learning Rate: 1e-3
- Batch Size: 64
- Latent Vector Dimension (Z_DIM): 512
- Progressive Training Epochs: [5, 5, 5] for 4×4, 8×8, and 16×16 resolutions respectively
- Gradient Penalty Lambda: 10
- Adam optimizer with betas=(0.0, 0.99)

## Usage

Run the training script:
```bash
python train.py
```

## Training Process

The training follows these steps:
1. Initialize generator and discriminator networks
2. Start training at 4×4 resolution
3. Progressively increase resolution to 8×8 and 16×16
4. At each resolution step:
   - Start with alpha=0 (fade-in phase)
   - Gradually increase alpha to 1 (stabilization phase)
   - Generate sample images after training at each resolution

## Results

Generated images are saved in the `generated_samples` directory, organized by resolution:
- `step_0_4x4/`: 4×4 resolution images
- `step_1_8x8/`: 8×8 resolution images
- `step_2_16x16/`: 16×16 resolution images

## Extending the Implementation

To train on higher resolutions:
1. Modify `MAX_RESOLUTION_STEP` to a higher value
2. Extend `PROGRESSIVE_EPOCHS` list to include epochs for higher resolutions
3. Make sure you have sufficient computational resources for higher resolutions

## References

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

## License

[MIT](LICENSE)
