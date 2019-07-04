#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Kleber Noel for disseration: 
    inversion mapping using a Local Linear embedding
June 2019
"""
import numpy as np 
#import pandas as pd 
import os, pdb, argparse, re, random
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from time import time
import torch, torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
 
get_seq_lens = lambda l : [x[1] for x in l]
return_first = lambda l : [x[0] for x in l]
dim1_list    = lambda l : [x.shape[1] for x in l]

def get_args():
    """
    Args define program usage.
    """
    info = 'Script preprocesses data and saves, given i/o directories.'
    parser = argparse.ArgumentParser(info)
    parser.add_argument('--lsf-dir', help='path to lsf directory')
    parser.add_argument('--ema-dir', help='path to ema directory')
    parser.add_argument('--shuffle', action='store_true', help='shuffle batches')
    parser.add_argument('--batch-size', type=int, default=6, help='shuffle batches')
    parser.add_argument('--num-layers', type=int, default=2, help='layers')

    return parser.parse_args()

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path, use_cuda=False):
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

def unnormalize_output(in_dict, batches, batches_set=True):
    unnormalized_data = []
    stds_tensor = torch.FloatTensor(in_dict)
    if batches_set==True: # A WHOLE BATCH
        for b in batches:
            norm = stds_tensor.repeat(b.shape[0],b.shape[1],1)
            assert norm.shape == b.shape
            unnormalized_data.append(torch.mul(norm, b))
        return unnormalized_data
    else: 
        b = batches
        norm = stds_tensor.repeat(b.shape[0],b.shape[1],1)
        return torch.mul(norm, b)


class LoadInputOutput():
    def __init__(self, args):
        self.lsf_dict = self.load_data(args.lsf_dir)
        self.ema_dict = self.load_data(args.ema_dir)
        self.shuffle  = args.shuffle
        
        n_input_dims  = set(dim1_list(self.lsf_dict['channel_lsfd'].values()))
        n_output_dims = set(dim1_list(self.ema_dict['channel_ema'].values()))
        assert (len(n_output_dims)==1 and len(n_input_dims)==1)
        self.input_dim  = n_input_dims.pop()
        self.output_dim = n_output_dims.pop()
        print('input dimensionality of lsfs:\t\t{} '.format(self.input_dim))
        print('output dimensionality of emas:\t\t{}'.format(self.output_dim))

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
                    if 'means' in d2: data_dict.update({key:torch.FloatTensor(self.load_means())})
                    elif 'stds' in d2:data_dict.update({key:torch.FloatTensor(self.load_4xSTDs())})
                    continue
                np_array = np.load(os.path.join(directory, d1, d2), 
                                   allow_pickle = True)
                if d1=='channel_name': data_dict.update({key:np_array})
                else:  data_dict[d1].update({key:torch.FloatTensor(np_array)})
        return data_dict
    
    def get_batches(self):
        sample_number = len(self.lsf_dict['channel_lsfd'].values())
        self.val_idx = int(sample_number * (1 - 0.1)) # validation 10%, training 90%
        self.sample_names = sorted([k for k in self.lsf_dict['channel_lsfd'].keys()])
        print('getting batches...')
        # define batch training (t) and validation (v) for...
        #   source (s)
        self.batch_ts, self.batch_vs = self.make_batch(self.lsf_dict['channel_lsfd'])
        # & target (t)
        self.batch_tt, self.batch_vt = self.make_batch(self.ema_dict['channel_ema'])
        # Normalize outputs to millimeters (for calculating RMSE in mm)
        ### DO FOR WHOLE BATCH
        self.batch_ttm = unnormalize_output(self.ema_dict['ema_stds'], [i for i in self.batch_tt[0]], True)
        self.batch_vtm = unnormalize_output(self.ema_dict['ema_stds'], [i for i in self.batch_vt[0]], True)
        
    def make_batch(self, in_dict, split=True):
        """
        Makes and pads a batch using data dictionary
        """
        report = 'batch target size = {}\nsegmenting data into batches...'
        print(report.format(args.batch_size))

        if split: #split into train, validation sets
            t, v = self.return_train_valid(in_dict) #INSERT NEW make_batch var to give random sequence for train/val
            return self.make_batch(t, False), self.make_batch(v, False)
        batches, batch, =  [], []
        frames_to_names, mxfr2nmfr = defaultdict(set), defaultdict(int)
        for k,s in [(k,v.shape[0]) for k,v in in_dict.items()]: frames_to_names[s].add(k)
        # Run through k:frames --> v:names dict in order (lowest to highest)
        for frames, names in sorted(frames_to_names.items(), key=lambda x:x[0]):
            batch.extend([(n,frames) for n in names])
            if len(batch) >= args.batch_size and batch!=0: ##BUG IN CODE... FIX
                mxfr2nmfr.update({frames:sorted(batch,reverse=True, key=lambda x:x[1])})
                batch = []
        if len(batch) > 0: 
            mxfr2nmfr.update({frames:sorted(batch,reverse=True, key=lambda x:x[1])})
        n2f=[]
        for batch in mxfr2nmfr.values():
            if batch==0: continue
            t = [torch.FloatTensor(in_dict[k]) for k, f in batch]
            padded_seq_batch = pad_sequence(t, batch_first=True)
            batches.append(padded_seq_batch)
            n2f.append([(k,in_dict[k].shape[0]) for k, f in batch])
        return batches, n2f
    
    def return_train_valid(self, in_dict):
        t, v, = defaultdict(str), defaultdict(str)
        t = {k:in_dict[k] for k in self.sample_names[:self.val_idx]}
        v = {k:in_dict[k] for k in self.sample_names[self.val_idx:]}   
        return t,v
    
class BuildModel(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(BuildModel, self).__init__()
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.hidden_dim    = 256
        self.fc_hidden_dim = 512
        self.batch_size    = 6 ##### CHANGE THIS!
        self.num_layers    = 2
        #self.batch_size = args.batch_size
        # L1 lstm cell: should use sigmoid as R activation and tanh as final
        self.bidirlstm = nn.LSTM(input_size    = self.input_dim, 
                                 hidden_size   = self.hidden_dim,
                                 bidirectional = True,
                                 num_layers    = args.num_layers)
        
        self.fc = nn.Linear(in_features  = self.fc_hidden_dim,
                            out_features = self.output_dim)
        self.hc = None

    def forward(self, padded_seq_batch, n2f):
        """
            x: PackedSequence
            hc: hidden and cell states
            tuple of hidden and cell state
        """
        self.get_len(padded_seq_batch.shape[0])
        # empty tensor for the output of the lstm
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.output_dim))
        
        packed_seq_batch =  pack_padded_sequence(padded_seq_batch, 
                                                 lengths=get_seq_lens(n2f), 
                                                 batch_first=True)
        #pdb.set_trace()
        self.hc = self.init_hidden(4, max(packed_seq_batch[1].tolist()), self.hc)

        t8 = time()
        packed_outputs, self.hc = self.bidirlstm(packed_seq_batch, self.hc)
        # repad (unpack) padded outputs
        t9 = time()
        lstm_output, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        output_seq = self.fc(lstm_output)
        self.forward_time=t9-t8

        return output_seq
    
    def init_hidden(self, a, b, x = None):
        if x==None:
            return (Variable(torch.randn(a, b, self.hidden_dim)),
                    Variable(torch.randn(a, b, self.hidden_dim)))
        else:
            return (Variable(torch.randn(a, b, self.hidden_dim)),
                    Variable(torch.randn(a, b, self.hidden_dim)))

    def get_len(self, length):
        self.sequence_len = length

def run():
    """
    
    """
    data  = LoadInputOutput(args)

    model = BuildModel(args, data.input_dim, data.output_dim)
    # get batches for train/ validation sets
    data.get_batches()
    count, t0 = int(0), time()
    # define the network, loss and the optimizer
    print('model architecture:\n\n{}'.format(model))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion, criterion_mm = nn.MSELoss(), nn.MSELoss()
    model_history={}
    for epoch in range(100):
        # reinit the hidden and cell states
        # Get batches: 1) lsf ready for input & 2a) ema norm & 2b) ema mm for target output
        #hc = model.init_hidden()
        zipped_batches = zip(data.batch_ts[0], 
                             data.batch_tt[0], 
                             data.batch_ttm,
                             data.batch_tt[1])
        progress_bar = tqdm(zipped_batches, desc='| Epoch {:03d}'.format(epoch), 
                            leave=False, disable=False)
        for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, n2f in progress_bar:
            #model.hc = model.init_hidden()
            
            count += 1
            optimizer.zero_grad() # zero out the gradients
            out_seq = model(Variable(pad_src_seq), n2f)
            out_seq_mm = unnormalize_output(data.ema_dict['ema_stds'], out_seq, False)
            loss       =    criterion(out_seq, Variable(pad_tgt_seq))
            loss_mm    =    criterion_mm(out_seq_mm, pad_tgt_seqm)
            model_history.update({count:(epoch, len(n2f), count, loss.item(), loss_mm.item()*1000, model.forward_time)})
            report_template = 'epoch:{}, batch:{}, count:{}, loss:{:.4}, loss_mm:{:.6}, forward_time:{:.4}'
            report = report_template.format(epoch, len(n2f), count, loss.item(), loss_mm.item()*1000, model.forward_time)
            progress_bar.set_description(report)
            loss.backward(retain_graph=False) # update model parameters
            optimizer.step()

            model.hc = model.hc[0].data, model.hc[1].data
            if int(loss_mm.item())==1: break
            #feedback every 10 batches:     val_data_batches   = get_batches(val_data, 6, False)
#            if count % 10 == 0:             
#                print('count: {}, loss:    {:.5}, '.format(count, loss.item()))
#                print('count: {}, loss_mm: {:.7}, '.format(count, loss_mm.item()*1000))
            #pdb.set_trace()
      
    tend=time()
    print(tend-t0)
    
    pdb.set_trace()
    None
    None
def main(args):
    run()

if __name__ == '__main__':
    args = get_args()
    main(args)
