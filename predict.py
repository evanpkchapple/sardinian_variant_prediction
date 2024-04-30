import sys 
import matplotlib.pyplot as plt
from IPython import display

from charrnn import *

from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from colorama import Fore, Back, Style

import re

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    net = CharRNN(checkpoint['tokens'], checkpoint['var_ls'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])

    train_history = checkpoint['train_history']

    return net, train_history
          
def test(net, n_lines=1, prime='import', top_k=None, device='cpu', display=False):
    max_len = 20
    net.to(device)
    net.eval()

    chars1 = []
    chars2 = []
    chars3 = []
    
    h = None
    for ch in prime:
        char, out_var, h, top_k_chars = net.predict(ch, h, device=device, top_k=top_k)
    
    chars1.append(top_k_chars[0])
    chars2.append(top_k_chars[1])
    chars3.append(top_k_chars[2])

    h1 = h
    h2 = h
    h3 = h
    
    for ii in range(1, max_len):
        res = net.predict(chars1[-1], h1, device=device, top_k=top_k)
        char = res[0]
        h1 = res[2]


        if char == ' ':
            chars1.append(' ')
            break
        if char == '.' or char == ',':
            break
        else:
            chars1.append(char)
                      
    for ii in range(1, max_len):
        res = net.predict(chars2[-1], h2, device=device, top_k=top_k)
        char = res[0]
        h2 = res[2]


        if char == ' ':
            chars2.append(' ')
            break
        if char == '.' or char == ',':
            break
        else:
            chars2.append(char)
                      
    for ii in range(1, max_len):
        res = net.predict(chars3[-1], h3, device=device, top_k=top_k)
        char = res[0]
        h3 = res[2]
        

        if char == ' ':
            chars3.append(' ')
            break
        if char == '.' or char == ',':
            break
        else:
            chars3.append(char)            
            
    print(prime)
    print(chars1)
    print(chars2)
    print(chars3)
    print(out_var)
    return (prime + ''.join(chars1), prime + ''.join(chars2), prime + ''.join(chars3)), out_var[0]
    
    
    

def clean_string(x):
    return x


cp = sys.argv[1]


net, _ = load_checkpoint(cp)

saved_clicks = 0
saved_chars = 0
correct_var_pred = 0
total_var_pred = 0

_, _, _, data = load_data('testdata')


import random
SEED = 500

random.seed(SEED)
random.shuffle(data)

print(len(data))

data = list(set(data))

print(len(data))

numchar = 0

for l in data:
    for c in l[0]:
        numchar = numchar + 1
        
print('total char', numchar)


for lines in data:
    print('line: \t\t ', lines)
    line = lines[0]
    var_corr = lines[1][0]
    i = 0
    clean_prefix = clean_string(line)
    while i < len(clean_prefix):
        temp_prefix=clean_prefix[0:i+1]
        print(Style.RESET_ALL + 'Prime: \t\t' + temp_prefix + Style.RESET_ALL)
        temp_output, var = test(net, 1, prime=temp_prefix, top_k=3)
        print('Actual Var: \t', var_corr)
        print('Guessed Var: \t', var)
        
        continuations = [temp_output[0][len(temp_prefix):],
            temp_output[1][len(temp_prefix):],
            temp_output[2][len(temp_prefix):]]
        #print(temp_prefix, '|||', temp_output, continuations)
        if temp_output[0] == clean_prefix[0:len(temp_output[0])]:
            print('Correct!: \t', temp_prefix + Fore.GREEN + continuations[0] + Style.RESET_ALL, ' ['+str(len(continuations[0])-1)+']')
            saved_chars = saved_chars + len(temp_output[0]) - (i + 1)
            saved_clicks = saved_clicks + len(temp_output[0]) - (i + 1) - 1
            i = len(temp_output[0])
        elif temp_output[1] == clean_prefix[0:len(temp_output[1])]:
            print('Correct!: \t', temp_prefix + Fore.GREEN + continuations[1] + Style.RESET_ALL, ' ['+str(len(continuations[1])-1)+']')
            saved_chars = saved_chars + len(temp_output[1]) - (i + 1)
            saved_clicks = saved_clicks + len(temp_output[1]) - (i + 1) - 1
            i = len(temp_output[1])
        elif temp_output[2] == clean_prefix[0:len(temp_output[2])]:
            print('Correct!: \t', temp_prefix + Fore.GREEN + continuations[2] + Style.RESET_ALL, ' ['+str(len(continuations[2])-1)+']')
            saved_chars = saved_chars + len(temp_output[2]) - (i + 1)
            saved_clicks = saved_clicks + len(temp_output[2]) - (i + 1) - 1
            i = len(temp_output[2])
        else:
            i = i + 1
    _, var = test(net, 1, prime=temp_prefix, top_k=3)
    if var_corr == var:
        print('Correct VAR: \t', var)
        correct_var_pred = correct_var_pred + 1
        total_var_pred = total_var_pred + 1
    else:
        print('Wrong VAR: \t', var, ' (Actual: ', var_corr, ')')
        total_var_pred = total_var_pred + 1
            
print('Saved clicks: \t',str(saved_clicks))
print('Chars saved: ', str(saved_chars))
print('Correct Var: ', str(correct_var_pred))
print('Total Var: ', str(total_var_pred))