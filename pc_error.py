import numpy as np 
import os
import time
import pandas as pd
import subprocess

rootdir = os.path.split(__file__)[0]

def get_points_number(filedir):
    with open(filedir, 'r') as plyfile:
        line = plyfile.readline()
        while line.find("element vertex") == -1:
            line = plyfile.readline()
        number = int(line.split()[-1])
    return number

def number_in_line(line):
    wordlist = line.split(' ')
    for item in wordlist:
        try:
            number = float(item)
            return number
        except ValueError:
            continue
    return None  # If no number found

def pc_error(infile1, infile2, res, normal=False, show=False):
    headers_color = ["c[0] mse  (p2point)", "c[0] PSNR (p2point)",
                     "c[1] mse  (p2point)", "c[1] PSNR (p2point)",
                     "c[2] mse  (p2point)", "c[2] PSNR (p2point)"]

    command = f'{rootdir}/pc_error_d' + \
              f' -a {infile1}' + \
              f' -b {infile2}' + \
              f' --hausdorff=1' + \
              f' --resolution={str(res)}' + \
              f' --color=1'  

    if normal:
        command += f' -n {infile1}'

    results = {}
   
    start = time.time()
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    while True:
        line = subp.stdout.readline().decode('utf-8')  
        if not line:
            break  # Break if no more output
        if show:
            print(line.strip())
        for header in headers_color:
            if line.startswith(header):
                results[header] = number_in_line(line)

    # Print execution time if needed
    # print(f'Execution time for `pc_error`: {round(time.time() - start, 4)} seconds')

    return pd.DataFrame([results])


if __name__ == '__main__':
    infile1 = 'path_to_first_point_cloud.ply'
    infile2 = 'path_to_second_point_cloud.ply'
    resolution = 1024  # Example resolution
    results_df = pc_error(infile1, infile2, resolution, normal=False, show=True)
    print(results_df)
