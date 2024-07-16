import argparse
import numpy as np
import os
import time
import pandas as pd
import torch
import MinkowskiEngine as ME
from nn import Encoder, Decoder
from data_loader import PointCloudDataset
from torch.utils.data import DataLoader
from pc_error import pc_error
from entropy_model import EntropyBottleneck
from AVRPM import AVRPM  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(encoder, decoder, entropy_bottleneck, avrpm, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    entropy_bottleneck.load_state_dict(checkpoint['entropy_bottleneck'])
    avrpm.load_state_dict(checkpoint['avrpm'])  

def save_ply(filename, coords, features):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {coords.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(coords.shape[0]):
            f.write(f"{coords[i, 0]} {coords[i, 1]} {coords[i, 2]} {int(features[i, 0])} {int(features[i, 1])} {int(features[i, 2])}\n")

def test(filedir, ckptdir, outdir, resultdir, channel, scaling_factor=1.0, res=1024):
    # Prepare directories
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    if not os.path.exists(resultdir): 
        os.makedirs(resultdir)

    # Load data
    dataset = PointCloudDataset(filedir, channel=channel)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Output filename
    filename = os.path.join(outdir, os.path.basename(filedir).split('.')[0])

    encoder = Encoder(dimension=3).to(device)
    decoder = Decoder(dimension=3).to(device)
    entropy_bottleneck = EntropyBottleneck(channels=1).to(device)
    avrpm = AVRPM(low_res=8, high_res=16).to(device)

    # Load trained model
    load_model(encoder, decoder, entropy_bottleneck, avrpm, ckptdir)

    all_results = pd.DataFrame()

    for idx, point_cloud in enumerate(data_loader):
        print('=' * 20, f'Testing point cloud {idx+1}', '=' * 20)
        point_cloud = point_cloud.to(device)
        coords = point_cloud[:, :3]
        feats = point_cloud[:, 3:]
        feats = (feats - feats.min()) / (feats.max() - feats.min())  # Normalize features to [0, 1]
        point_cloud = ME.SparseTensor(feats, coordinates=coords)

        high_res_voxels, low_res_voxels = avrpm(point_cloud)
        combined_voxels = high_res_voxels + low_res_voxels

        compressed = encoder(combined_voxels)
        quantized, _ = entropy_bottleneck(compressed, quantize_mode="symbols")
        reconstructed = decoder(quantized)

        reconstructed_coords = reconstructed.C.cpu().numpy()
        reconstructed_features = reconstructed.F.cpu().numpy()
        output_ply = filename + f'_reconstructed_{idx+1}.ply'
        save_ply(output_ply, reconstructed_coords, reconstructed_features)

        pc_error_metrics = pc_error(filedir, output_ply, res=res, normal=True, show=False)
        print('D1 PSNR:', pc_error_metrics["c[0] PSNR (p2point)"][0])

        results = {
            'point_cloud': idx + 1,
            'D1 PSNR': pc_error_metrics["c[0] PSNR (p2point)"][0]
        }
        all_results = all_results.append(results, ignore_index=True)

    csv_name = os.path.join(resultdir, 'results.csv')
    all_results.to_csv(csv_name, index=False)
    print('Results written to:', csv_name)

    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Attribute Compression Test')
    parser.add_argument('--filedir', type=str, default='data/your_point_cloud_data.ply')
    parser.add_argument('--ckptdir', type=str, default='checkpoints/r0.pth', help='Path to checkpoint directory')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--resultdir', type=str, default='results')
    parser.add_argument('--channel', type=str, choices=['Y', 'U', 'V'], required=True, help="YUV channel to process")
    parser.add_argument('--scaling_factor', type=float, default=1.0)
    parser.add_argument('--res', type=int, default=1024)
    args = parser.parse_args()

    test(args.filedir, args.ckptdir, args.outdir, args.resultdir, args.channel, args.scaling_factor, args.res)
