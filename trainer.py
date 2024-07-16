import os
import torch
import numpy as np
from tqdm import tqdm, trange
from data_loader import PointCloudDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Trainer:
    def __init__(self, encoder, decoder, discriminator, entropy_bottleneck, avrpm, params, channel):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.entropy_bottleneck = entropy_bottleneck
        self.AVRPM = AVRPM  
        self.params = params
        self.channel = channel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)
        self.entropy_bottleneck.to(self.device)
        self.AVRPM .to(self.device)  
        
        self.optimizer_G = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.entropy_bottleneck.parameters()), lr=params['g_lr']
        )
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=params['d_lr'])
        
        self.criterion_adv = torch.nn.BCELoss()
        self.criterion_rec = torch.nn.MSELoss()

        self.data_loader = DataLoader(PointCloudDataset(params['data_dir'], channel=self.channel), batch_size=params['batch_size'], shuffle=True)

    def train(self):
        num_epochs = self.params['num_epochs']
        
        for epoch in range(num_epochs):
            for point_cloud in self.data_loader:
                point_cloud = point_cloud.to(self.device)
                coords = point_cloud.view(-1, 3)  # Reshape to N x 3
                feats = torch.ones(coords.shape[0], 1, device=self.device)  # Dummy features
                point_cloud = ME.SparseTensor(feats, coordinates=coords)
                
                high_res_voxels, low_res_voxels = self.avrpm(point_cloud)
                
                combined_voxels = high_res_voxels + low_res_voxels

                # Train discriminator
                self.optimizer_D.zero_grad()
                real_labels = torch.ones(point_cloud.shape[0], 1, device=self.device)
                fake_labels = torch.zeros(point_cloud.shape[0], 1, device=self.device)
                
                real_output = self.discriminator(combined_voxels)
                real_loss = self.criterion_adv(real_output, real_labels)
                
                compressed = self.encoder(combined_voxels)
                quantized, likelihood = self.entropy_bottleneck(compressed)
                reconstructed = self.decoder(quantized)
                fake_output = self.discriminator(reconstructed)
                fake_loss = self.criterion_adv(fake_output, fake_labels)
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_D.step()
                
                # Train generator (encoder-decoder)
                self.optimizer_G.zero_grad()
                
                compressed = self.encoder(combined_voxels)
                quantized, likelihood = self.entropy_bottleneck(compressed)
                reconstructed = self.decoder(quantized)
                rec_loss = self.criterion_rec(reconstructed, point_cloud)
                
                fake_output = self.discriminator(reconstructed)
                g_loss_adv = self.criterion_adv(fake_output, real_labels)
                
                g_loss = rec_loss + g_loss_adv + likelihood.mean()
                g_loss.backward()
                self.optimizer_G.step()
                
            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
