#  PCAC-GAN:ASparse-Tensor-Based Generative Adversarial Network for 3D Point Cloud Attribute Compression

## Requirments
cuda 11.8

pytorch 1.8

MinkowskiEngine 0.5 or higher

torchac 0.9.3

python 3.8

We recommend you to follow https://github.com/NVIDIA/MinkowskiEngine to setup the environment for sparse convolution. 


## Pre-trained models
We trained seven models with seven different parameters.
https://pan.baidu.com/s/1h-xs_orDr6CEdggOlVOAOA 
code：ABCD

## Usage

### Training
```shell
 python trainer.py --dataset='training_dataset_rootdir'
```

### Testing

python main.py compress --input="./testdata/soldier_vox10_0690.ply" --ckpt_dir='./model/1/'
python main.py decompress --input="./testdata/soldier_vox10_0690" --ckpt_dir='./model/1/'

