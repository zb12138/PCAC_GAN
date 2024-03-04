import argparse
import numpy as np
import os
import time
import pandas as pd
import torch
from pcc_model import PCCModel  
from data_utils import load_sparse_tensor, scale_sparse_tensor, write_ply_ascii_geo
from pc_error import pc_error  
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(filedir, ckptdir_list, outdir, resultdir, scaling_factor=1.0, rho=1.0, res=1024):
    # Prepare directories
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    if not os.path.exists(resultdir): 
        os.makedirs(resultdir)

    # Load data
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:', round(time.time() - start_time, 4), 's')

    # Output filename
    filename = os.path.join(outdir, os.path.basename(filedir).split('.')[0])

    model = PCCModel().to(device)


    all_results = pd.DataFrame()

    for idx, ckptdir in enumerate(ckptdir_list):
        print('=' * 20, f'Testing rate {idx+1}', '=' * 20)
        assert os.path.exists(ckptdir), f'Checkpoint does not exist: {ckptdir}'
        ckpt = torch.load(ckptdir, map_location=device)
        model.load_state_dict(ckpt['model'])
        coder = Coder(model=model, device=device)


        x_scaled = scale_sparse_tensor(x, factor=scaling_factor) if scaling_factor != 1 else x

        # Encoding
        start_time = time.time()
        coder.encode(x_scaled, postfix=f'_r{idx+1}')
        print('Encoding Time:', round(time.time() - start_time, 3), 's')

        # Decoding
        start_time = time.time()
        x_dec = coder.decode(postfix=f'_r{idx+1}', rho=rho)
        print('Decoding Time:', round(time.time() - start_time, 3), 's')


        x_dec = scale_sparse_tensor(x_dec, factor=1/scaling_factor) if scaling_factor != 1 else x_dec


        bits = np.sum([os.path.getsize(filename + f'_r{idx+1}' + ext) * 8 for ext in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
        num_points = len(x_dec)
        bpp = bits / num_points
        print('Total bits:', bits, 'BPP:', bpp)


        write_ply_ascii_geo(filename + f'_r{idx+1}_dec.ply', x_dec.cpu().numpy())


        pc_error_metrics = pc_error(filedir, filename + f'_r{idx+1}_dec.ply', res=res, normal=True, show=False)
        print('D1 PSNR:', pc_error_metrics["mseF,PSNR (p2point)"][0])


        results = {
            'num_points(input)': len(x),
            'num_points(output)': num_points,
            'resolution': res,
            'bits': bits,
            'bpp': bpp,
            'D1 PSNR': pc_error_metrics["mseF,PSNR (p2point)"][0]
        }
        all_results = all_results.append(results, ignore_index=True)


    csv_name = os.path.join(resultdir, os.path.basename(filedir).split('.')[0] + '.csv')
    all_results.to_csv(csv_name, index=False)
    print('Results written to:', csv_name)

    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Compression Test')
    parser.add_argument('--filedir', type=str, default='testdata/pointcloud.ply')
    parser.add_argument('--ckptdir_list', nargs='+', default=['ckptdir/model1.pth', 'ckptdir/model2.pth'], help='List of checkpoint directories')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--resultdir', type=str, default='results')
    parser.add_argument('--scaling_factor', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--res', type=int, default=1024)
    args = parser.parse_args()

    test(args.filedir, args.ckptdir_list, args.outdir, args.resultdir, args.scaling_factor, args.rho, args.res)
