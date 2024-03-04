import argparse
import numpy as np
import os
import random
import sys
import torch
import MinkowskiEngine as ME
import yaml
from model import D, G, skipD, G_inv_Tanh  
from sandwich_trainer import SandwichTrainer  # Your custom training class


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

with open(args.config, 'r') as f:
    parsed = yaml.safe_load(f)  # Using safe_load instead of load

# Setup output directory based on configuration
parsed['out_dir'] = './results/aug_points_{num_points_per_object}_opt_{optimizer}_lr_{d_lr}_{g_lr}_c_{critic_steps}_z1_{z1_dim}_z2_{z2_dim}_d_{d_dim}_clip_{weight_clip}_pool_{pool}_type_{type}_gradpen_{lambda_grad_pen}_Arc_{arc}_invact_{invact}_N_{n_obj}_odim_{o_dim}_obj_{obj}'.format(**parsed)
if not os.path.exists(parsed['out_dir']):
    os.makedirs(parsed['out_dir'])


if isinstance(parsed['obj'], str):
    if parsed['obj'].startswith('multi'):
        num_objects = int(parsed['obj'][5:])
        parsed['obj'] = list(range(num_objects))
    else:
        parsed['obj'] = [int(parsed['obj'])]

# Log the arguments and settings
maxLen = max(len(key) for key in parsed.keys())
fmtString = '\t{:<%d} : {}' % maxLen
with open(os.path.join(parsed['out_dir'], 'log.txt'), 'w') as f:
    f.write(' '.join(sys.argv) + '\n\n')
    print('Arguments:')
    f.write('Arguments:\n')
    for key, value in sorted(parsed.items()):
        formatted_string = fmtString.format(key, value)
        print(formatted_string)
        f.write(formatted_string + '\n')

    # Initialize models based on parsed configuration
    d = skipD(x_dim=parsed['x_dim'], d_dim=parsed['d_dim'], z1_dim=parsed['z1_dim'], o_dim=1)
    g = G(z1_dim=parsed['z1_dim'], z2_dim=parsed['z2_dim'], x_dim=parsed['x_dim'])
    g_inv = G_inv_Tanh(x_dim=parsed['x_dim'], d_dim=parsed['d_dim'], z1_dim=parsed['z1_dim'], pool=parsed['pool'])

    # Log model structures
    f.write(str(d) + '\n')
    f.write(str(g) + '\n')
    f.write(str(g_inv) + '\n')

# Initialize the trainer with models and configurations
trainer = SandwichTrainer(g, d, g_inv, parsed)

# Start training
trainer.train()
