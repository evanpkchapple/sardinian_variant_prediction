import matplotlib.pyplot as plt
from IPython import display

from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from charrnn import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

chars, all_vars, data, _ = load_data('traindata')

print('all vars = ', all_vars)

import random
SEED = 500

random.seed(SEED)
random.shuffle(data)

n_hidden=256
n_layers=3

net = CharRNN(chars, all_vars, n_hidden=n_hidden, n_layers=n_layers)

name1 = 'model_'+str(n_hidden)+'_'+str(n_layers)
train(net, data, epochs=50, n_seqs=4, n_steps=1, lr=0.0001, device=device, val_frac=0.89,
      name = name1, plot=False, early_stop=False)
