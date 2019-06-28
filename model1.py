#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:36:55 2019

@author: s1888641
"""


import numpy as np 
import pandas as pd 
import os, pdb, argparse
import tqdm as tqdm
from collections import defaultdict, OrderedDict


import torch, torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Any results you write to the current directory are saved as output.

def get_args():
    """
    Args define program usage.
    """
    info = 'Script preprocesses data and saves, given i/o directories.'
    parser = argparse.ArgumentParser(info)
    parser.add_argument('--lsf-dir', default=None, help='path to lsf directory')
    parser.add_argument('--ema-dir', default=None, help='path to ema directory')
    
    #parser.add_argument('--file-encoding', default='utf-8', help='encoding to read in text file corpus')
    return parser.parse_args()

def save_model(model, model_path):
    """Save model."""
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

class BuildModel(nn.ModuleList):
    def __init__(self, args):
        super(BuildModel, self).__init__()
        print('loading dict {}'.format('lsf_dict'))
        self.lsf_dict = self.load_data(args.lsf_dir)
        print('loading dict {}'.format('ema_dict'))
        self.ema_dict = self.load_data(args.ema_dir)
        self.hidden_dim = 256
        self.batch_size = 1
        #self.batch_size = args.batch_size
        
        self.input_dim = set([v.shape[1] for v in self.lsf_dict['channel_lsfd'].values()]).pop()
        self.output_dim = set([v.shape[1] for v in self.ema_dict['channel_ema'].values()]).pop()
        
        print('input dimensionality of lsfs:\t\t{} '.format(self.input_dim))
        print('output dimensionality of emas:\t\t{}'.format(self.output_dim))        
        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=self.input_dim, hidden_size=self.hidden_dim)
        # second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim) 
        # dropout layer for the output of the second layer cell
        self.dropout = nn.Dropout(p=0.5)
        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        
    def load_data(self, lsf_dir):
        data_dict = defaultdict(dict)
        for d1 in os.listdir("./{}".format(lsf_dir)):
            print('loading data into dict from {} : {}'.format(lsf_dir, d1))
            if not "channel" in d1: continue
            for d2 in os.listdir("./{}/{}".format(lsf_dir,d1)):
                np_array = np.load(os.path.join(lsf_dir,d1,d2), allow_pickle=True)
                if d1=='channel_name': data_dict.update({d2.rsplit('.')[0]:np_array})
                else:  data_dict[d1].update({d2.rsplit('.')[0]:np_array})
        return data_dict

    def forward(self, x, hc):
        """
            x: input to the model
                *  x[t] - input of shape (batch, input_size) at time t
                
            hc: hidden and cell states
                *  tuple of hidden and cell state
        """ 
        
        # empty tensor for the output of the lstm
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.output_dim))
        
        # pass the hidden and the cell state from one lstm cell to the next one
        # we also feed the output of the first layer lstm cell at time step t to the second layer cell
        # init the both layer cells with the zero hidden and zero cell states
        hc_1, hc_2 = hc, hc
        # for every step in the sequence
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1( x.transpose(1,0).unsqueeze(dim=0)[: , : , t], hc_1)
            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1
            # pass the hidden state from the first layer to the cell in the second layer
            hc_2 = self.lstm_2(h_1, hc_2)
            # unpack the hidden and cell states from the second layer cell
            h_2, c_2 = hc_2
            
            ###### IS ARCHITECTURE CORRECT!? DOUBLE CHECK THIS!... ######
            
            # form the output of the fc
            output_seq[t] = self.fc(self.dropout(h_2))
        # return the output sequence

        return output_seq.view((self.sequence_len * self.batch_size, -1))
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(self.batch_size, self.hidden_dim),
                torch.zeros(self.batch_size, self.hidden_dim))
        
    def get_len(self, length):
        self.sequence_len = length

# batching function
def get_batches(arr, n_seqs_in_a_batch, n_characters):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    
    batch_size = n_seqs_in_a_batch * n_characters
    n_batches = len(arr)//batch_size
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs_in_a_batch, -1))
    
    for n in range(0, arr.shape[1], n_characters):
        # The features
        x = arr[:, n:n+n_characters]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_characters]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


def run():
    """
    
    """
    net = BuildModel(args)
    # define the loss and the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # get the validation and the training data (10%)
    samples = len(net.lsf_dict['channel_lsfd'].values())
    val_idx = int(samples * (1 - 0.1))
    sample_keys = sorted([k for k in net.lsf_dict['channel_lsfd'].keys()])
    data_keys, val_data_keys = sample_keys[:val_idx], sample_keys[val_idx:]
    # empty list for the validation losses
    #### IF BATCH TRAINING CREATE A DIFFERENT DICTIONARY, USE PADDING ####
    val_losses, samples =  [], []
    count=int(0)
    for epoch in range(10):
        
        # reinit the hidden and cell steates
        hc = net.init_hidden()
        
        for k,v in dict(net.lsf_dict['channel_lsfd']).items():
            count+=1
            x = net.lsf_dict['channel_lsfd'][k]
            net.get_len(net.lsf_dict['channel_lsfd'][k].shape[0])
            
            # get the torch tensors training data
            # also transpose the axis for the training set and the targets
            
            x_train = torch.from_numpy(x) #.transpose([1, 0, 2])
            
            
            targets = torch.from_numpy(net.ema_dict['channel_ema'][k])  # tensor of the target
            
            # zero out the gradients
            optimizer.zero_grad()
            
            # get the output sequence from the input and the initial hidden and cell states
            output = net(x_train, hc)
            
            # calculate the loss
            # we need to calculate the loss across all batches, so we have to flat the targets tensor
            loss = criterion(output, targets)
            #print('count: {}, loss: {}, '.format(count, loss.item()))
            # calculate the gradients
            loss.backward()
            
            # update the parameters of the model
            optimizer.step()
        
            # feedback every 10 batches
            if count % 10 == 0: print('count: {}, loss: {}, '.format(count, loss.item()))

def main(args):
    run()
if __name__ == '__main__':
    args = get_args()

    main(args)
