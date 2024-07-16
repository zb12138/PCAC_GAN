import torch
import torch.nn as nn

class AVRPM(nn.Module):
    def __init__(self, low_res, high_res):
        super(AVRPM, self).__init__()
        self.low_res = low_res
        self.high_res = high_res


        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        density = self.sigmoid(self.conv3(self.conv2(self.conv1(x))))
        

        high_res_mask = (density > 0.5).float()
        low_res_mask = 1 - high_res_mask

        high_res_voxels = x * high_res_mask
        low_res_voxels = x * low_res_mask
        
        return high_res_voxels, low_res_voxels
