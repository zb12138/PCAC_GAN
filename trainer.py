import os
import h5py
import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from data_loader import DataLoader
import pdb
import torch_model

class Trainer(BaseTrainer):
    def __init__(self, G, D, G_inv, params):
        super(Trainer, self).__init__(G, D, G_inv, params)

    def build(self, inputs):
        # Generator and discriminator outputs
        G_output = self.G(inputs)
        D_real = self.D(inputs)
        D_fake = self.D(G_output)

        # Adversarial loss
        Ladv = -tf.reduce_mean(tf.log(D_real + 1e-8)) - tf.reduce_mean(tf.log(1 - D_fake + 1e-8))

        # Decompression loss
        sigma = 1.0 
        xi = 2.0  
        logits = self.G_inv(G_output)  # Assuming G_inv acts as a decoder
        labels = inputs  # Assuming the original inputs are the ground truth for reconstruction
        Ldec = -sigma * (1 - logits)**xi * tf.log(logits + 1e-8)  # σ-balanced focal loss
        Ldec = tf.reduce_mean(Ldec)


        phi_adv = self.phi_adv
        phi_dec = self.phi_dec
        D = phi_adv * Ladv + phi_dec * Ldec


        lambda_rd = self.lambda_rd
        R = self.calculate_rate(G_output)  # Assuming calculate_rate function exists
        L = lambda_rd * D + R

        return D, L, Ladv

    def calculate_rate(self, compressed_output):
        non_zero_count = tf.cast(tf.count_nonzero(compressed_output), tf.float32)
        total_elements = tf.cast(tf.size(compressed_output), tf.float32)
        R = non_zero_count / total_elements  # Simple ratio of non-zero elements
        return R

    def train(self):
        super(Trainer, self).train()

params = {
    'z1_dim': 100,  # Example parameters
    'z2_dim': 50,
    'x_dim': 3,
    'd_dim': 64,
    'pool': 'max',
    'phi_adv': 1.0,
    'phi_dec': 0.5,
    'lambda_rd': 0.01,
    'batch_size': 32,
    'num_points_per_object': 2048,
    'data_file': 'path/to/your/data.h5',
    'out_dir': 'path/to/output/dir',
    'num_iters': 10000,
    'critic_steps': 5,
    'optimizer': 'adam',
    'd_lr': 0.0001,
    'g_lr': 0.0001,
    'inv_lr': 0.0001,
    'n_obj': 10
}

trainer = Trainer(G, D, G_inv, params)
trainer.train()
