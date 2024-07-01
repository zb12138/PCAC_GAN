#  PCAC-GAN:ASparse-Tensor-Based Generative Adversarial Network for 3D Point Cloud Attribute Compression

PCAC-GAN/
│
├── data/
│   ├── your_point_cloud_data.ply
│
├── checkpoints/
│   ├── model1.pth
│   ├── model2.pth
│
├── main.py
├── module.py
├── nn.py
├── pc_error.py
├── test.py
├── trainer.py
├── config.yml
├── data_loader.py
├── entropy_model.py


## Requirments
cuda 11.8

pytorch 1.8

MinkowskiEngine 0.5 or higher

torchac 0.9.3

python 3.8

We recommend you to follow https://github.com/NVIDIA/MinkowskiEngine to setup the environment for sparse convolution. 


## Pre-trained models
We trained seven models with seven different parameters.
https://pan.baidu.com/s/1xfSJxzJ1yuPdFro2qfvi-w
code：wiwg 

## Usage

### Training
```shell
 python trainer.py --dataset='training_dataset_rootdir'
```

### Testing

python test.py --filedir data/your_point_cloud_data.ply --ckptdir checkpoints/r0.pth --outdir output --resultdir results --scaling_factor 1.0 --res 1024


