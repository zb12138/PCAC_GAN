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

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")

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

# Log model structures
with open(os.path.join(config['out_dir'], 'log.txt'), 'a') as f:
    f.write(str(encoder) + '\n')
    f.write(str(decoder) + '\n')
    f.write(str(discriminator) + '\n')

# Initialize the trainer with models and configurations
trainer = Trainer(encoder, decoder, discriminator, config)

# Start training
trainer.train()
