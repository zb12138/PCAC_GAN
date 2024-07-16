import argparse
import numpy as np
import os
import random
import sys
import torch
import MinkowskiEngine as ME
import yaml
from nn import Encoder, Decoder, Discriminator
from trainer import Trainer
from entropy_model import EntropyBottleneck
from AVRPM import AVRPM  

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")
parser.add_argument("--channel", type=str, choices=['Y', 'U', 'V'], required=True, help="YUV channel to process")

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Setup output directory based on configuration
if not os.path.exists(config['out_dir']):
    os.makedirs(config['out_dir'])

# Log the arguments and settings
with open(os.path.join(config['out_dir'], 'log.txt'), 'w') as f:
    f.write(' '.join(sys.argv) + '\n\n')
    print('Arguments:')
    f.write('Arguments:\n')
    for key, value in sorted(config.items()):
        formatted_string = f'{key}: {value}'
        print(formatted_string)
        f.write(formatted_string + '\n')

# Initialize models based on configuration
encoder = Encoder(dimension=3)
decoder = Decoder(dimension=3)
discriminator = Discriminator(dimension=3)
entropy_bottleneck = EntropyBottleneck(channels=1)
AVRPM = AVRPM(low_res=8, high_res=16)  

# Log model structures
with open(os.path.join(config['out_dir'], 'log.txt'), 'a') as f:
    f.write(str(encoder) + '\n')
    f.write(str(decoder) + '\n')
    f.write(str(discriminator) + '\n')
    f.write(str(entropy_bottleneck) + '\n')
    f.write(str(AVRPM) + '\n')  

# Initialize the trainer with models and configurations
trainer = Trainer(encoder, decoder, discriminator, entropy_bottleneck, avrpm, config, args.channel)

# Start training
trainer.train()
