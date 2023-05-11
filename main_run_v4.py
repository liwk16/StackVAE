import numpy as np
import pandas as pd
from dataloader import *
from GraphVAEv4 import GraphVAEv4
import os
import matplotlib.pyplot as plt
import time
# from plotter import *
import argparse
import torch

parser = argparse.ArgumentParser(description='GraphVAE for time series Anomaly Detection')
parser.add_argument('-n', '--name', type=str, default='NASA')
parser.add_argument('-la', '--lamda', type=float, default=0.5)
parser.add_argument('-k', '--kneighbour', type=int, default=10)
parser.add_argument('-w', '--wsize', type=int, default=100)
parser.add_argument('-a', '--alpha', type=float, default=3.)
parser.add_argument('-i', '--index', type=int)

args = parser.parse_args()
print(args)

window_size = args.wsize
h_dim = 200
z_dim = 20
data_split_rate = 0.5
epochs = 256
lr = 1e-3
lr_decay = 0.75 # decay every 10 epoch, 0.8 for Yahoo!, 0.75 for KPIs
lamb = 10 # 10 for Yahoo!, 1 for KPIs
clip_norm_value = 12.0 # 12.0, 10.0
bs = 64
weight_decay=1e-3

root_path = 'graphvae-resultv4_window_{}_index_{}'.format(window_size, args.index)

dirname = args.name + '_lambda_' + str(args.lamda).replace('.', '_') + '_k_' + str(args.kneighbour) + '_a_' + str(args.alpha)

model_path = os.path.join(root_path, dirname, 'models')
score_path = os.path.join(root_path, dirname, 'scores')
pic_path = os.path.join(root_path, dirname, 'pics')

os.makedirs(model_path, exist_ok=True)
os.makedirs(score_path, exist_ok=True)
os.makedirs(pic_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

start = time.time()
lambda_gcn = args.lamda
assert(lambda_gcn <= 1.)

item = 'A-1.npy' # one instance of NASA dataset, for example
    
train_data, test_data = load_NASA(item)
channels = train_data.shape[1]

vae = GraphVAEv4(window_size, channels, lambda_gcn=lambda_gcn, device=device,
                    neighbour=args.kneighbour,
                    graph_alpha=args.alpha)

vae.fit(train_data)
vae.save(os.path.join(model_path, item.replace('npy', 'pt')))

res = vae.detect(test_data)
np.save(os.path.join(score_path, item), res['recon'])
# plot_heatmap(vae.get_adjmat(), anot=False,
#             save_path=os.path.join(pic_path, item.replace('npy', 'png')))

print('Total cost', time.time() - start)
