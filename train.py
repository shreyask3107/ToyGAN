import os
import time
import json
import torch
import argparse

# Logging
from utils import plot_toy, save_json, MetricMeter
from torch.utils.tensorboard import SummaryWriter

# Data
from data.factory import get_data

# Model
from models.factory import get_model_cls

# Optim
from torch.optim import SGD, Adam, Adamax

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument('--dataset', type=str, default='two_moons')
parser.add_argument('--num_bits', type=int, default=None)
parser.add_argument('--train_samples', type=int, default=128*1000)
parser.add_argument('--test_samples', type=int, default=128*100)

# Model params
parser.add_argument('--model', type=str, default='flow')
model_cls = get_model_cls(parser)
parser = model_cls.add_args(parser)

# Train params
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='adam', choices={'sgd', 'adam', 'adamax'})

# Test params
parser.add_argument('--test_batch_size', type=int, default=1280)
parser.add_argument('--num_samples', type=int, default=128*1000)

# Run params
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--name', type=str, default=time.strftime("%Y-%m-%d_%H-%M-%S"))
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

#####################
## Prepare logging ##
#####################

torch.manual_seed(args.seed)

path = f'results/{args.dataset}/{args.model}_{args.name}'
os.makedirs(path)
save_json(vars(args), path, 'args.json')

writer = SummaryWriter(path)

##################
## Specify data ##
##################

train_loader, test_loader = get_data(args)

###################
## Specify model ##
###################

model = model_cls(args).to(args.device)

#######################
## Specify optimizer ##
#######################

if args.optimizer == 'sgd':
    opt = lambda m: SGD(m.parameters(), lr=args.lr)
elif args.optimizer == 'adam':
    opt = lambda m: Adam(m.parameters(), lr=args.lr)
elif args.optimizer == 'adamax':
    opt = lambda m: Adamax(m.parameters(), lr=args.lr)

if hasattr(model, 'optimizer_groups'):
    optimizer = [opt(m) for m in model.optimizer_groups()]
else:
    optimizer = opt(model)

##############
## Training ##
##############

print('Training...')
model.train()
loss_meter = MetricMeter()
for epoch in range(args.epochs):
    loss_meter.reset()
    for i, x in enumerate(train_loader):
        
        if args.model == 'gan':
            x = x.float()
            
        losses = model.training_step(x, optimizer)
        loss_meter.log(losses)
        print('Epoch: {}/{}, Iter: {}/{}, {}'.format(epoch+1, args.epochs, i+1, len(train_loader), loss_meter), end='\r')
    for k,v in loss_meter: writer.add_scalar(k, v, epoch)
    print('')
save_json(loss_meter.compute(), path, 'loss.json')

#############
## Testing ##
#############

model.eval()
# if hasattr(model, 'test_step'):
print('Testing...')
test_meter = MetricMeter()
with torch.no_grad():
    for i, x in enumerate(test_loader):
        metrics = model.test_step(x)
        test_meter.log(metrics)
        print('Iter: {}/{}'.format(i+1, len(test_loader)), end='\r')
    for k,v in test_meter: writer.add_scalar(k, v, epoch)
    print('')
save_json(test_meter.compute(), path, 'metrics.json')

##############
## Sampling ##
##############

print('Sampling...')
samples = model.sample(args.num_samples)
plot_toy(samples, num_bits=args.num_bits, save_path=path)
