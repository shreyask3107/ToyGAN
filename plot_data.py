import torch
import argparse
from utils import plot_toy
from data.factory import get_data

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument('--dataset', type=str, default='two_moons')
parser.add_argument('--num_bits', type=int, default=None)
parser.add_argument('--samples', type=int, default=128*1000)
parser.add_argument('--save', type=eval, default=False)

# Plotting params
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

args = parser.parse_args()
args.train_samples = args.samples
args.test_samples = args.samples
args.batch_size = 128
args.test_batch_size = 128

save_path = 'results' if args.save else None
name = f'{args.dataset}_{args.num_bits}bit' if args.num_bits else args.dataset
# name = 'textgan_exp1'
torch.manual_seed(0)

##################
## Specify data ##
##################

_, test_loader = get_data(args)

##############
## Sampling ##
##############

plot_toy(test_loader.dataset.data, num_bits=args.num_bits, save_path=save_path, name=name)

print('Plotting done')
