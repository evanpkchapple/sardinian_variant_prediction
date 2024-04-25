import matplotlib.pyplot as plt
from IPython import display
import glob
import os
import sys
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def load_data(path):    
    data_files = glob.glob(path +'/*.txt')    
    data = []
    unencoded_data = []
    all_lines = {}
    all_vars = []
    all_text = ''
    i = 0    
    idx2var = []    
    for fn in data_files:
        var = os.path.basename(fn).split('.')[0]
        all_lines[i] = open(fn).readlines()
        idx2var.append((var,i))
        for lines in all_lines[i]:
            all_text += lines
        all_vars.append(var)
        i += 1    
    chars = tuple(set(all_text))    
    list_chars = list(chars)
    list_chars.insert( len(list_chars),'đ')
    list_chars.remove(' ')
    list_chars.insert(0, ' ')
    chars = list_chars
    idx2chars = list_chars
    char2idx = {j: i for i, j in enumerate(idx2chars)} 
    for var in all_lines:
        for line in all_lines[var]:
            new_line = []
            for char in line:
                new_line.append(char2idx[char])
            data.append((new_line, var))
            unencoded_data.append((line, idx2var[var]))
    return chars, all_vars, data, unencoded_data

def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

def get_batches(data, n_seqs, n_steps): 
    batch_size = n_seqs * n_steps
    import random
    random.shuffle(data)
    for line in data:
        arr = np.array(line[0])
        var = line[1]
        n_batches = len(arr) // batch_size
        arr = arr[:n_batches * batch_size]
        arr = arr.reshape((n_seqs, -1))
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]            
            var_arr = np.zeros_like(y, dtype=y.dtype)
            var_arr.fill(var)        
            yield x, [y, var_arr]

class CharRNN(nn.Module):
    def __init__(self, tokens, vars_ls, n_hidden=256, n_layers=2, drop_prob=0.5, num_vars = 4):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vars = vars_ls
        self.chars = tokens
        self.int2var = dict(enumerate(self.vars))
        self.var2int = {var: ii for ii, var in self.int2var.items()}
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, len(self.chars))
        self.fc2 = nn.Linear(n_hidden, num_vars)
             
    def get_n_hidden(self):
        return self.n_hidden
    
    def get_n_layers(self):
        return self.n_layers
    
    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x_char = self.fc1(x)
        x_var = self.fc2(x)        
        return x_char, x_var, hidden
    
    def predict(self, char, hidden=None, device=torch.device('cpu'), top_k=None):
        with torch.no_grad():
            self.to(device)
            try:
                x = np.array([[self.char2int[char]]])
            except KeyError:
                return '', '', hidden, ['', '', '']
            x = one_hot_encode(x, len(self.chars))
            inputs = torch.from_numpy(x).to(device)            
            out_char, out_var, hidden = self.forward(inputs, hidden)
            p = F.softmax(out_char, dim=2).data.to('cpu')            
            p1 = F.softmax(out_var, dim=2).data.to('cpu')
            if top_k is None:
                top_ch = np.arange(len(self.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()                
            p1, top_vars = p1.topk(4)
            top_vars = top_vars.numpy().squeeze()
            p1 = p1.numpy().squeeze()            
            if top_k == 1:
                char = int(top_ch)
            else:
                p = p.numpy().squeeze()
                char = np.random.choice(top_ch, p=p / p.sum())
            top_k_chars = []
            for i in top_ch:
                top_k_chars.append(self.int2char[i])            
            out_vars = []
            for i in top_vars:
                out_vars.append(self.int2var[i])
            return self.int2char[char], out_vars, hidden, top_k_chars

def save_checkpoint(net, opt, filename, train_history={}):
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'tokens': net.chars,
                  'train_history': train_history,
                  'var_ls': net.vars
                  }
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])
    return net, checkpoint

plt.ion()

def train(net, data, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, device=torch.device('cpu'),
          name='checkpoint', early_stop=True, plot=True):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion_char = nn.CrossEntropyLoss()
    criterion_var = nn.CrossEntropyLoss()
    net.to(device)
    min_val_loss = 10.**10
    train_history = {'epoch': [], 'step': [], 'loss': [], 'val_loss': []}
    n_chars = len(net.chars)
    n_vars = len(net.vars)
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    for e in range(epochs):
        hidden = None
        for x, y in get_batches(data, n_seqs, n_steps):
            x = one_hot_encode(x, n_chars)
            inputs, targets_char, targets_var = torch.from_numpy(x).to(device), torch.tensor(y[0], dtype = torch.long).to(device), torch.tensor(y[1], dtype = torch.long).to(device)
            net.zero_grad()
            output_char, output_var, hidden = net.forward(inputs, hidden)
            loss_char = criterion_char(output_char.view(n_seqs * n_steps, n_chars), targets_char.view(n_seqs * n_steps))
            loss_var = criterion_var(output_var.view(n_seqs * n_steps, n_vars), targets_var.view(n_seqs * n_steps))
            loss = loss_char + loss_var
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            hidden = (hidden[0].detach(), hidden[1].detach())
        with torch.no_grad():
            val_h = None
            val_losses = []
            for x, y in get_batches(val_data, n_seqs, n_steps):
                x = one_hot_encode(x, n_chars)
                inputs, targets_char, targets_var = torch.from_numpy(x).to(device), torch.tensor(y[0], dtype = torch.long).to(device), torch.tensor(y[1], dtype = torch.long).to(device)
                output_char, output_var, val_h = net.forward(inputs, val_h)
                val_loss_char = criterion_char(output_char.view(n_seqs * n_steps, n_chars), targets_char.view(n_seqs * n_steps))
                val_loss_var = criterion_var(output_var.view(n_seqs * n_steps, n_vars), targets_var.view(n_seqs * n_steps))
                val_loss = val_loss_char.item() + val_loss_var.item()
                val_losses.append(val_loss)
            mean_val_loss = np.mean(val_losses)
            train_history['epoch'].append(e+1)
            train_history['loss'].append(loss.item())
            train_history['val_loss'].append(mean_val_loss)
        print('e=' + str(e))
        if (e+1) % 5 == 0:
            save_checkpoint(net, opt, 'final_model_'+str(e+1)+'.net', train_history={})
        if plot:
            plt.clf()
            plt.plot(train_history['loss'], lw=2, c='C0')
            plt.plot(train_history['val_loss'], lw=2, c='C1')
            plt.xlabel('epoch')
            plt.title("{}   Epoch: {:.0f}/{:.0f}   Loss: {:.4f}   Val Loss: {:.4f}".format(
                datetime.now().strftime('%H:%M:%S'),
                e+1, epochs,
                loss.item(),
                mean_val_loss), color='k')
            display.clear_output(wait=True)
            display.display(plt.gcf())
        else:
            print("{}   Epoch: {:.0f}/{:.0f}   Loss: {:.4f}   Val Loss: {:.4f}".format(
                datetime.now().strftime('%H:%M:%S'),
                e+1, epochs,
                loss.item(),
                mean_val_loss))
        if mean_val_loss < min_val_loss:
            save_checkpoint(net, opt, name+'.net', train_history=train_history)
            min_val_loss = mean_val_loss
        if early_stop:
            if e - np.argmin(train_history['val_loss']) > 10:
                display.clear_output()
                print('Validation loss does not decrease further, stopping training.')
                break
    plt.plot(train_history['epoch'], train_history['loss'], label='loss')
    plt.plot(train_history['epoch'], train_history['val_loss'], label='val_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss_graph_hidden' + str(net.get_n_hidden()) + '_layers' + str(net.get_n_layers()) + '.png')
