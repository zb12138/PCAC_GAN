import torch
import MinkowskiEngine as ME
import torch.nn as tnn
from module import Module, ReLU

class SparseConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dimension):
        super(SparseConv, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size, stride=stride, dimension=dimension
        )

    def forward(self, x):
        return self.conv(x)

class SparseConvTranspose(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dimension):
        super(SparseConvTranspose, self).__init__()
        self.conv = ME.MinkowskiConvolutionTranspose(
            in_channels, out_channels, kernel_size, stride=stride, dimension=dimension
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(Module):
    def __init__(self, dimension):
        super(Encoder, self).__init__()
        self.layer1 = Sequential(
            SparseConv(1, 32, 3, 1, dimension),
            ReLU(),
            SparseConv(32, 64, 3, 2, dimension),
            ReLU()
        )
        self.layer2 = Sequential(
            SparseConv(64, 128, 3, 1, dimension),
            ReLU(),
            SparseConv(128, 256, 3, 2, dimension),
            ReLU()
        )
        self.layer3 = Sequential(
            SparseConv(256, 512, 3, 1, dimension),
            ReLU(),
            SparseConv(512, 1024, 3, 2, dimension)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

class Decoder(Module):
    def __init__(self, dimension):
        super(Decoder, self).__init__()
        self.layer1 = Sequential(
            SparseConvTranspose(1024, 512, 3, 2, dimension),
            ReLU(),
            SparseConvTranspose(512, 256, 3, 1, dimension),
            ReLU()
        )
        self.layer2 = Sequential(
            SparseConvTranspose(256, 128, 3, 2, dimension),
            ReLU(),
            SparseConvTranspose(128, 64, 3, 1, dimension),
            ReLU()
        )
        self.layer3 = Sequential(
            SparseConvTranspose(64, 32, 3, 2, dimension),
            ReLU(),
            SparseConvTranspose(32, 1, 3, 1, dimension)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

class Discriminator(Module):
    def __init__(self, dimension):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(dimension)
        self.fc = tnn.Linear(1024, 1)  

    def forward(self, x):
        x = self.encoder(x)
        x = ME.MinkowskiGlobalPooling()(x)
        x = x.flatten(start_dim=1)
        return torch.sigmoid(self.fc(x))
