import os
import torch
import numpy as np
from tqdm import tqdm, trange
from data_loader import PointCloudDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Trainer:
    def __init__(self, encoder, decoder, discriminator, params):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)
        
        self.optimizer_G = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=params['g_lr']
        )
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=params['d_lr'])
        
        self.criterion_adv = torch.nn.BCELoss()
        self.criterion_rec = torch.nn.MSELoss()

        self.data_loader = DataLoader(PointCloudDataset(params['data_dir']), batch_size=params['batch_size'], shuffle=True)

    def train(self):
        num_epochs = self.params['num_epochs']
        
        for epoch in range(num_epochs):
            for point_cloud in self.data_loader:
                point_cloud = point_cloud.to(self.device)
                coords = point_cloud.view(-1, 3)  # Reshape to N x 3
                feats = torch.ones(coords.shape[0], 1, device=self.device)  # Dummy features
                point_cloud = ME.SparseTensor(feats, coordinates=coords)
                
                # Train discriminator
                self.optimizer_D.zero_grad()
                real_labels = torch.ones(point_cloud.shape[0], 1, device=self.device)
                fake_labels = torch.zeros(point_cloud.shape[0], 1, device=self.device)
                
                real_output = self.discriminator(point_cloud)
                real_loss = self.criterion_adv(real_output, real_labels)
                
                compressed = self.encoder(point_cloud)
                reconstructed = self.decoder(compressed)
                fake_output = self.discriminator(reconstructed)
                fake_loss = self.criterion_adv(fake_output, fake_labels)
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_D.step()
                
                # Train generator (encoder-decoder)
                self.optimizer_G.zero_grad()
                
                compressed = self.encoder(point_cloud)
                reconstructed = self.decoder(compressed)
                rec_loss = self.criterion_rec(reconstructed, point_cloud)
                
                fake_output = self.discriminator(reconstructed)
                g_loss_adv = self.criterion_adv(fake_output, real_labels)
                
                g_loss = rec_loss + g_loss_adv
                g_loss.backward()
                self.optimizer_G.step()
                
            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
