#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:36:55 2019

@author: s1888641
"""


import numpy as np 
import pandas as pd 
import os, pdb, argparse, re, random
from tqdm import tqdm
from collections import defaultdict, OrderedDict

import torch, torch.nn as nn, torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
# Any results you write to the current directory are saved as output.
get_seq_lens = lambda l : [x[1] for x in l]
return_first = lambda l : [x[0] for x in l]
def get_args():
    """
    Args define program usage.
    """
    info = 'Script preprocesses data and saves, given i/o directories.'
    parser = argparse.ArgumentParser(info)
    parser.add_argument('--lsf-dir', help='path to lsf directory')
    parser.add_argument('--ema-dir', help='path to ema directory')
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
        self.lsf_dict = self.load_data(args.lsf_dir)
        self.ema_dict = self.load_data(args.ema_dir)
        self.hidden_dim    = 256
        self.fc_hidden_dim = 512
        self.batch_size    = 6
        self.num_layers    = 1
        #self.batch_size = args.batch_size
        self.input_dim = set([v.shape[1] for v in self.lsf_dict['channel_lsfd'].values()]).pop()
        self.output_dim = set([v.shape[1] for v in self.ema_dict['channel_ema'].values()]).pop()
        
        print('input dimensionality of lsfs:\t\t{} '.format(self.input_dim))
        print('output dimensionality of emas:\t\t{}'.format(self.output_dim))        
        # L1 lstm cell: should use sigmoid as R activation
        self.bidirlstm = nn.LSTM(input_size    = self.input_dim, 
                                 hidden_size   = self.hidden_dim,
                                 bidirectional = True,
                                 batch_first   = True)
        # self.dropout = nn.Dropout(p=0.5) # possible dropout lyr for lstm out
        # fully connected layer connecting output of LSTM cell to fc output
        self.fc = nn.Linear(in_features  = self.fc_hidden_dim,
                            out_features = self.output_dim)
    
    def load_means(self):
        return np.array([float(i[0])*10**int(i[1]) for i in self.norm])
    
    def load_4xSTDs(self):
        return np.array([4*(float(i[0])*10**int(i[1])) for i in self.norm])
    
    def load_data(self, directory):
        ema_norm_str, data_dict = '(\d+\.\d+)e([\-\+]\d+)', defaultdict(dict)
        for d1 in os.listdir("./{}".format(directory)):
            print('loading data to dict from {} : {}'.format(directory, d1))
            if not "channel" in d1: continue
            for d2 in os.listdir("./{}/{}".format(directory,d1)):
                key = d2.split('.')[0]
                if re.match('ema_.*.txt',d2): 
                    with open(os.path.join(directory, d1, d2), 'r') as f:
                        self.norm = re.findall(ema_norm_str,f.readlines()[0])
                    if 'means' in d2: data_dict.update({key:self.load_means()})
                    elif 'stds' in d2:data_dict.update({key:self.load_4xSTDs()})
                    continue
                np_array = np.load(os.path.join(directory, d1, d2), 
                                   allow_pickle = True)
                if d1=='channel_name': data_dict.update({key:np_array})
                else:  data_dict[d1].update({key:np_array})
        return data_dict

    def forward(self, x, hc):
        """
            x: PackedSequence
            hc: hidden and cell states
            tuple of hidden and cell state
        """
        # empty tensor for the output of the lstm
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.output_dim))
        packed_outputs, (h, c) = self.bidirlstm(x.float())
        # repad (unpack) padded outputs
        lstm_output, _ = pad_packed_sequence(packed_outputs, 
                                             batch_first=True, 
                                             total_length=self.sequence_len)
        output_seq = self.fc(lstm_output)
        return output_seq
    
    def init_hidden(self): # initialize hidden state and cell state to zeros
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        
    def get_len(self, length):
        self.sequence_len = length
    
def get_batches(in_dict, batch_size=None, shuffle=False, net_in=False):
    batches, batch, frames_to_names, mxfr2nmfr =  [], [], defaultdict(set), defaultdict(int)
    for k,s in [(k,v.shape[0]) for k,v in in_dict.items()]: frames_to_names[s].add(k)
    print('batch target size={}\nsegmenting data into batches...'.format(batch_size))
    # Run through k:frames --> v:names dict in order
    for frames, names in sorted(frames_to_names.items(), key=lambda x:x[0]):
        batch.extend([(n,frames) for n in names])
        if len(batch) >= batch_size:
            mxfr2nmfr.update({frames:sorted(batch,reverse=True, key=lambda x:x[1])})
            batch = []
    if len(batch) > 0: mxfr2nmfr.update({frames:sorted(batch,reverse=True, key=lambda x:x[1])})
    if shuffle: random.shuffle(mxfr2nmfr)
    for max_len, batch in mxfr2nmfr.items():
        t = [torch.FloatTensor(in_dict[k]) for k, f in batch]
        padded_seq_batch = pad_sequence(t, batch_first=True)
        if net_in: 
            packed_seq_batch =  pack_padded_sequence(padded_seq_batch, 
                                                     lengths=get_seq_lens(batch), 
                                                     batch_first=True)
            batches.append((packed_seq_batch, batch))
        else: # packed batch sequences are not needed for net input, only ouput
            batches.append((padded_seq_batch, batch))
    return batches

def unnormalize_output(batches, norm_data):
    pdb.set_trace()
    unnormalized_data = []
    for b in batches:
        norm = torch.FloatTensor(norm_data).repeat([*b.shape][0],[*b.shape][1],1)
        assert norm.shape == b.shape
        unnormalized_data.append(torch.mul(norm, b))
    
    return unnormalized_data

def run():
    """
    
    """
    net = BuildModel(args)
    # define the loss and the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion, criterion_mm = nn.MSELoss(), nn.MSELoss()
    
    # get batches for train/ validation sets
    sample_number = len(net.lsf_dict['channel_lsfd'].values())
    val_idx = int(sample_number * (1 - 0.1)) # validation 10%, training 90%
    samples = sorted([k for k in net.lsf_dict['channel_lsfd'].keys()])
    ts, vs, tt, vt = defaultdict(str), defaultdict(str), defaultdict(str), defaultdict(str)
    ts = {k:net.lsf_dict['channel_lsfd'][k] for k in samples[:val_idx]}
    vs = {k:net.lsf_dict['channel_lsfd'][k] for k in samples[val_idx:]}    
    tt = {k:net.ema_dict['channel_ema'][k]  for k in samples[:val_idx]}
    vt = {k:net.ema_dict['channel_ema'][k]  for k in samples[val_idx:]}
    print('getting batches...')
    # define batch training (t) and validation (v) sets for source (s) and target (t)
    batch_ts  = get_batches(ts, batch_size=6, shuffle=False, net_in=True)
    batch_vs  = get_batches(vs, batch_size=6, shuffle=False, net_in=True)
    
    batch_tt  = get_batches(tt, batch_size=6, shuffle=False, net_in=False)
    batch_vt  = get_batches(vt, batch_size=6, shuffle=False, net_in=False)
    batch_ttm = unnormalize_output(return_first(batch_tt), net.ema_dict['ema_stds'])
    batch_vtm = unnormalize_output(return_first(batch_vt), net.ema_dict['ema_stds'])
    #[FloatTensor(i) for k,i in net.lsf_dict['channel_lsfd'].items()]
    count=int(0)
    for epoch in range(10):
        # reinit the hidden and cell states
        hc = net.init_hidden()
        # Get batches: 1) lsf ready for input & 2a) ema norm & 2b) ema mm for target output
        for s, t, tm in tqdm(zip(batch_ts, batch_tt, batch_ttm)):
            src_pkd_seq, batch = s
            tgt_seq, _ = t
            pdb.set_trace()
            count += 1
            net.get_len(max(get_seq_lens(batch)))
            # get the torch tensors training data
            #batch_targets = pad_sequence(tensortargets, batch_first=True)
            # zero out the gradients
            optimizer.zero_grad()
            # get the output sequence from the input and the initial hidden and cell states
            output = net(src_pkd_seq.float(), hc)

            #normalized_stds = torch.stack([torch.FloatTensor(net.ema_dict['ema_stds']) for i in range(output.size(0))])
            
#            for o, n, index in zip(output, normalized_stds, range(output.shape[0])):
#                print(o.shape, n.shape)
#                output_mm.append(torch.FloatTensor([i*j for i,j in zip(o.tolist(),n.tolist())]))
#                target_mm.append(torch.FloatTensor([i*j for i,j in zip(targets[index].tolist(),n.tolist())]))
#            
#            output_mm=torch.stack(output_mm)
#            target_mm=torch.stack(target_mm)
            # calculate the loss
            # we need to calculate the loss across all batches, so we have to flat the targets tensor
            output_mm = unnormalize_output(output, net.ema_dict['ema_stds'])
            
            loss       =    criterion(output, tgt_seq)
            loss_mm    =    criterion_mm(output_mm ,tgt_seq)
            print('count: {}, loss:    {}, '.format(count, loss.item(tgt_seq)))
            print('count: {}, loss_mm: {}, '.format(count, loss_mm.item()*1000))
            # calculate the gradients
            
            # update the parameters of the model
            optimizer.step()
        
            # feedback every 10 batches:     val_data_batches   = get_batches(val_data, 6, False)
            #if count % 10 == 0: print('count: {}, loss: {}, '.format(count, loss.item()))

def main(args):
    run()

if __name__ == '__main__':
    args = get_args()
    main(args)
