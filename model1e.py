#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Kleber Noel for disseration: 
    inversion mapping using a Local Linear embedding
June 2019
"""
import numpy as np 
from numpy import mean

#import pandas as pd 
import os, pdb, argparse, re
from random import shuffle
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from time import time
import torch, torch.nn as nn
import torch.optim as optim
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

get_seq_lens = lambda l : [x[-1] for x in l]
get_names    = lambda l : [x[0] for x in l]
dim1_list    = lambda l : [x.shape[1] for x in l]
report = 'count:{}, batch_size:{}, loss:{:.4}; mm:{:.4}, frames:{}, cor:{:.4}'

# GET CUDA STATUS:

def get_args():
    """
    Args define program usage.
    """
    info = 'Script preprocesses data and saves, given i/o directories.'
    parser = argparse.ArgumentParser(info)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test',  action='store_true', default=False)
    
    parser.add_argument('--lsf-dir',  help='path to lsf directory')
    parser.add_argument('--mfcc-dir', help='path to mfcc directory')
    parser.add_argument('--ema-dir',  help='path to ema directory')
    parser.add_argument('--bilstm-layers', type=int, default=2, help='number of bi-directional LSTM layers')

    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle batches')
    parser.add_argument('--batch-size', type=int, default=6, help='shuffle batches')
    
    parser.add_argument('--cuda', action='store_true', default=False, help='GPU support')

    parser.add_argument('--load-dir', help='directory to load a saved model from')
    parser.add_argument('--save-dir', help='directory to save a final model in')
    
    parser.add_argument('--checkpoint-dir', default = '', help='checkpoint directory to load/save model from/in')
    parser.add_argument('--patience', type=int, default=10, help='save model after n epochs without improvement')
    parser.add_argument('--save-internal', default=5, help='epoch arg to save the model parameters every *n* times')

    parser.add_argument('--graph-data', action='store_true', help='graph data')
    
    return parser.parse_args()

class LoadInputOutput():
    def __init__(self, args, d):
        if args.lsf_dir: self.lsf_dict = self.load_data(args.lsf_dir)
        if args.mfcc_dir: self.mfcc_dict= self.load_data(args.mfcc_dir)
        self.ema_dict = self.load_data(args.ema_dir)
        self.device   = d
        self.shuffle  = args.shuffle
        self.input_dim, self.output_dim = self.get_io_dims()
        print('input dimensionality of lsfs:\t\t{} '.format(self.input_dim))
        print('output dimensionality of emas:\t\t{}'.format(self.output_dim))
    
    def get_io_dims(self):
        n_input_dims  = set(dim1_list(self.lsf_dict['channel_lsfd'].values()))
        n_output_dims = set(dim1_list(self.ema_dict['channel_ema'].values()))
        assert {len(n_output_dims),len(n_input_dims)}.pop()==1 # sanity check
        return n_input_dims.pop(), n_output_dims.pop()

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
                    if 'means' in d2: data_dict.update({key:FloatTensor(self.load_means())})
                    elif 'stds' in d2:data_dict.update({key:FloatTensor(self.load_4xSTDs())})
                    continue
                np_array = np.load(os.path.join(directory, d1, d2), 
                                   allow_pickle = True)
                if d1=='channel_name': data_dict.update({key:np_array})
                else:  data_dict[d1].update({key:FloatTensor(np_array)})
        return data_dict
    
    def get_batches(self):
        total_samples = len(self.lsf_dict['channel_lsfd'].values())
        # currently just choose a validation index: 10%, training 90%
        self.val_idx = int(total_samples * (1 - 0.1))
        self.sample_names = sorted([k for k in self.lsf_dict['channel_lsfd'].keys()])
        report = 'minimum batch target size = {}\nsegmenting data into batches...'
        print(report.format(args.batch_size))
        # define batch TRAINING (t) and VALIDATION (v) for...
        #   SOURCE set (s) & TARGET set (t)
        self.bts, self.bvs = self.make_batches(self.lsf_dict['channel_lsfd'])
        self.btt, self.bvt = self.make_batches(self.ema_dict['channel_ema'])
        # Normalize outputs to millimeters (for calculating RMSE in mm)
        ### DO FOR WHOLE BATCH
        batch_list_tt = [i for i in self.btt[0]]
        batch_list_vt = [i for i in self.bvt[0]]
        self.bttm = self.unnormalize_output(batch_list_tt, True)
        self.bvtm = self.unnormalize_output(batch_list_vt, True)
    
    def reorder(self, t, i, tup=False):
        if tup: return ([t[0][n] for n in i],[t[1][n] for n in i])
        if not tup: return [t[n] for n in i]
    
    def shuffle_batches(self):
        """
        Shuffles batches based on numpy:
            get len of training/validation batches -> 0,1, ... len(batch).
        """
        t, v = np.arange(len(self.bts[0])), np.arange(len(self.bvs[0]))
        np.random.shuffle(t), np.random.shuffle(v) # new order for t & v.
        t, v = t.tolist(), v.tolist()
        self.bts  = self.reorder(self.bts, t, tup=True)
        self.bvs  = self.reorder(self.bvs, v, tup=True)
        self.btt  = self.reorder(self.btt, t, tup=True)
        self.bvt  = self.reorder(self.bvt, v, tup=True)
        self.bttm = self.reorder(self.bttm, t, tup=False)
        self.bvtm = self.reorder(self.bvtm, v, tup=False)
        
    def make_batches(self, in_dict, split=True):
        """
        Makes and pads a batch using data dictionary
        """
        if split: #split into train, validation sets
            t, v = self.return_train_valid(in_dict) #INSERT NEW make_batch var to give random sequence for train/val
            return self.make_batches(t, False), self.make_batches(v, False)
        batches, batch, =  [], []
        frames_to_names, mxfr2nmfr = defaultdict(set), defaultdict(int)
        for k,s in [(k,v.shape[0]) for k,v in in_dict.items()]: frames_to_names[s].add(k)
        # Run through k:frames --> v:names dict in order (lowest to highest)
        for frames, names in sorted(frames_to_names.items(), key=lambda x:x[0]):
            batch.extend([(n,frames) for n in names])
            ### Aim to get up to or above batch size... issue: dif batch sizes.
            if len(batch) >= args.batch_size and batch!=0: ## why batch=0? FIX!
                mxfr2nmfr.update({frames:sorted(batch,reverse=True, key=lambda x:x[1])})
                batch = []
        if len(batch) > 0: # remainder that didn't fit into target batch sizes
            mxfr2nmfr.update({frames:sorted(batch,reverse=True, key=lambda x:x[1])})
        n2f=[]
        for batch in mxfr2nmfr.values():
            if batch==0: continue
            t = [FloatTensor(in_dict[k]).to(self.device) for k, f in batch]
            batches.append(pad_sequence(t, batch_first=True))
            n2f.append([(k,in_dict[k].shape[0]) for k, f in batch])
        return batches, n2f
    
    def return_train_valid(self, in_dict):
        t, v, = defaultdict(str), defaultdict(str)
        t = {k:in_dict[k] for k in self.sample_names[:self.val_idx]}
        v = {k:in_dict[k] for k in self.sample_names[self.val_idx:]}   
        return t,v
    
    def unnormalize_output(self, batches, batches_set=True):
        unnormalized_data = []
        stds_tensor = FloatTensor(self.ema_dict['ema_stds']).to(self.device)
        if batches_set==True: # A WHOLE BATCH
            for b in batches:
                norm = stds_tensor.repeat(b.shape[0],b.shape[1],1)
                unnormalized_data.append(torch.mul(norm, b))
            return unnormalized_data
        else:
            b = batches
            norm = stds_tensor.repeat(b.shape[0],b.shape[1],1)
            return torch.mul(norm, b)

    def zip_validation_batches(self):
        """
        Val args in zip: 1. src; 2. tgt; 3. tgt (mm), 4. names & sizes
        """
        return zip(self.bvs[0], self.bvt[0], self.bvtm, self.bvt[1])

    def zip_training_batches(self):
        """
        Train args in zip: 1. src; 2. tgt; 3. tgt (mm), 4. names & sizes
        """
        return zip(self.bts[0], self.btt[0], self.bttm, self.btt[1])

    def get_prog_bar(self, zipped_batches, batches_type, epoch):
        t = len(self.bts[0]) if batches_type=='training' else len(self.bvs[0])
        bar = tqdm(iterable = zipped_batches, 
                   total = t,
                   desc = '| {} Epoch {:03d}'.format(batches_type, epoch),
                   leave = False,
                   disable = False)
        return bar

class BuildModel(nn.Module):
    def __init__(self, args, d, input_dim, output_dim):
        super(BuildModel, self).__init__()
        self.args          = args
        self.device        = d
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.hidden_dim    = 256
        self.fc_hidden_dim = 512
        self.bilstm_layers = args.bilstm_layers
        self.batch_size    = args.batch_size # default: 6
        # L1 lstm cell: should use sigmoid as R activation and tanh as final
        self.bidirlstm = nn.LSTM(input_size    = self.input_dim, 
                                 hidden_size   = self.hidden_dim,
                                 bidirectional = True,
                                 num_layers    = self.bilstm_layers)
        
        self.fc = nn.Linear(in_features  = self.fc_hidden_dim,
                            out_features = self.output_dim)
        self.hc = None
        self.criterion, self.criterion_mm = nn.MSELoss(), nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.is_new = False # assume model loaded from chkpnt til proven otherwise

    def forward(self, padded_seq_batch):
        """
            padded_seq_batch: padded sequence
            hc: hidden and cell states
            tuple of hidden and cell state
        """
        self.get_len(padded_seq_batch.shape[0])
        self.out_seq = torch.empty((self.sequence_len,
                                       self.batch_size,
                                       self.output_dim))
        packed_seq_batch =  pack_padded_sequence(padded_seq_batch, 
                                                 lengths=self.seq_lens, 
                                                 batch_first=True)
        
        ## Reinitialize hidden state, WRT to the dimensions of the LSTM
        ## as LSTM input dim1 changes due to packed sequence dim1 changes in 
        ## batch. Some batches are dif sizes, hidden state size changes 
        ## Before, I was only passing the packed_seq_batch to bidirlstm.

        self.hc = self.init_hidden(2*self.bilstm_layers, 
                                   max(packed_seq_batch[1].tolist()), 
                                   self.hc)
        packed_outputs, self.hc = self.bidirlstm(packed_seq_batch, self.hc)
        lstm_out, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        self.out_seq = self.fc(lstm_out)
    
    def init_hidden(self, a, b, x = None):
        return (Variable(torch.randn(a, b, self.hidden_dim,device=self.device)),
                Variable(torch.randn(a, b, self.hidden_dim,device=self.device)))

    def init_history(self):
        if self.args.checkpoint_dir:
            print('checkpoint "{}" not found'.format(args.checkpoint_dir))
            print('initialising history and creating new model...')
        self.history, self.is_new = defaultdict(dict), True 
        self.history['training'], self.history['validation'] = OrderedDict(), OrderedDict()

    def get_len(self, length):
        self.sequence_len = length
    
    def get_cor(self, output, target):
        """
        function for correlations between target and output in a batch
        normalized over the number of frames samples
        """
        self.av_cor = float(0)
        for frames, o, t in zip(self.seq_lens, output, target):
            vo = o - torch.mean(o[:frames,:])
            vt = t - torch.mean(o[:frames,:])
            numerator = torch.sum(vo * vt)
            denominator = (torch.sqrt(torch.sum(vo ** 2)) * torch.sqrt(torch.sum(vt ** 2)))
            cor = numerator/denominator
            self.av_cor += cor.item()*frames/sum(self.seq_lens)
        
    def get_lens_names_frames(self, names_to_frames):
        """
        computed batch-wise
        """
        self.seq_lens = get_seq_lens(names_to_frames)
        self.names = get_names(names_to_frames)
        self.frames = sum(self.seq_lens)

    def get_losses(self, out_seqm, pad_tgt_seq, pad_tgt_seqm):
        self.loss    = self.criterion(self.out_seq, pad_tgt_seq)
        self.loss_mm = self.criterion_mm(out_seqm, pad_tgt_seqm).item()*1000

    def propagate_loss(self):
        self.loss.backward()
        self.optimizer.step()
    
    def update_info(self, epoch, phase):
        """
        phase is either training or testing.
        info during epoch is updated as a set of loss values (mm and norm), 
        frame number, correlation, sample number for each minibatch
        at end, average loss is computed.
        """
        self.epoch = epoch
        sample_number = len(self.names)
        s = tuple([sample_number, self.frames, self.loss.item(), self.loss_mm, self.av_cor])
        if self.history[phase].get(epoch): self.history[phase][epoch].append(s)
        else: self.history[phase].update({epoch:[s]})

    def set_prog_bar_desc(self, prog_bar, c):
        batch_size = len(self.names)
        prog_bar.set_description(report.format(c, batch_size, self.loss.item(), 
                                               self.loss_mm, self.frames, 
                                               self.av_cor))
        
    def get_epoch(self):
        """
        since checkpoints/models are only saved after validation,
        we only look into epoch of validation set here.
        If process was cancelled in 1st train, start new history
        """
        if dict(self.history.get('validation'))=={}: 
            self.init_history()
            self.__init__(self.args, self.input_dim, self.output_dim)
            return 0
        else:
            last_epoch = max(dict(self.history.get('validation')).keys())
            return last_epoch
        
    def compute_batch_averages(self, phase):
        """
        average loss (norm & mm), and cor calculated over batch for phase
        normalize over entire batch instead of retaining single batch samples
        """
        b_lm, b_l, b_c = float(0), float(0), float(0)
        total_in_batch = sum([i[0] for i in dict(self.history[phase])[self.epoch]])
        for n, f, l, lm, c in dict(self.history[phase])[self.epoch]:
            b_c  += c * n  / total_in_batch
            b_l  += l * n  / total_in_batch 
            b_lm += lm * n / total_in_batch 
        del self.history[phase][self.epoch]
        self.history[phase].update({self.epoch:{b_l, b_c, b_lm}})
        print('\nOverall {0}: lossmm = {1:.3f} cor = {2:.3f}'.format(phase, b_lm, b_c))
        if phase=='validation': return b_l

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, device.type))

def save_checkpoint(model, history, filepath):
    state = {'epoch': model.epoch,
             'state_dict': model.state_dict(),
             'optimizer': model.optimizer.state_dict(),
             'history': model.history}
    torch.save(state, filepath)

def load_checkpoint(model, args, device):
    checkpoint = torch.load(args.checkpoint_dir, map_location = device.type)
    model.epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    model.history = checkpoint['history']

def get_knn(batch_out_seq, seq_lens, kneighbors):
    from sklearn.manifold import locally_linear_embedding as lle
    from sklearn.neighbors import NearestNeighbors
    import numpy.matlib as matlib
    from numpy.linalg import inv
    
    # By default the knn function below will return the same vector.
    # e.g. k = 4 vector: 0, neighbors [0, 1, 2, 3]. We want [1, 2, 3, 4]
    # therefore add one to get distinct from the reference.
    knn = NearestNeighbors(n_neighbors = kneighbors + 1)
    pdb.set_trace()
    for sample, l in zip(batch_out_seq, seq_lens):
        W = np.zeros((kneighbors,sample.shape[0]))
        knn.fit(sample[:l,:])
        k_nn, k_ni = knn.kneighbors(sample[:l,:], return_distance=True)
        k_nn, k_ni = k_nn[:,1:], k_ni[:,1:]
        for i in range(l):
            Z = sample[:l,][k_ni[i]] - matlib.repmat(sample[:l,][i], kneighbors, 1)
            C = Z @ Z.T
            W[:,i] = inv(C)*np.ones((kneighbors, 1))
            W[:,i] = W[:,i]/W[:,i].sum(axis=1,keepdims=1)
            

def get_device(cuda):
    if torch.cuda.is_available() and cuda: return torch.device('cuda')
    else: return torch.device('cpu')

def validate(model, data, epoch, count):
    val_prog_bar = data.get_prog_bar(data.zip_validation_batches(), 
                                     'validation', epoch)
    torch.no_grad(), model.eval()
    for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, n2f in val_prog_bar:
        count += 1

        #if count > 4:  break # CODE TO STOP AND TEST 
            #k_nearest_neighbors = get_knn(model.out_seq.detach().numpy(), model.seq_lens)
        model.get_lens_names_frames(n2f)
        model.forward(Variable(pad_src_seq)) # pass thru model
        model.get_cor(model.out_seq, pad_tgt_seq) # get correlation
        out_seq_mm = data.unnormalize_output(model.out_seq, False) #MSE(mm)
        model.get_losses(out_seq_mm, pad_tgt_seq, pad_tgt_seqm)
        
        model.update_info(epoch, 'validation')
        model.set_prog_bar_desc(val_prog_bar, count)

def train(model, data, epoch, count):
    train_prog_bar = data.get_prog_bar(data.zip_training_batches(), 
                                       'training', epoch)
    model.train()
    for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, n2f in train_prog_bar:
        count += 1
        #if count > 2: break # CODE TO STOP AND TEST
        model.get_lens_names_frames(n2f)
        model.forward(Variable(pad_src_seq)) # pass thru model
        model.get_cor(model.out_seq, pad_tgt_seq) # get correlation
        out_seq_mm = data.unnormalize_output(model.out_seq, False) #MSE(mm)
        model.get_losses(out_seq_mm, pad_tgt_seq, pad_tgt_seqm)
        model.propagate_loss()
        model.optimizer.zero_grad()
        model.update_info(epoch, 'training')
        model.set_prog_bar_desc(train_prog_bar, count)

def train_val_loop(model, data):
    bad_epochs = 0
    best_valid_loss = float('inf')
    print('Model Overview:\n\n{}'.format(model))
    while bad_epochs < args.patience:
        count = 0
        epoch += 1
        if args.shuffle: data.shuffle_batches()
        train(model, data, epoch, count)
        validate(model, data, epoch, count)
        model.compute_batch_averages('training')
        bv_l = model.compute_batch_averages('validation')
        if bv_l < best_valid_loss: # Decide whether to terminate training while loop
            best_valid_loss, bad_epochs = bv_l, 0
            if epoch % args.save_internal ==0: save_checkpoint(model, model.history, args.checkpoint_dir)
        else: bad_epochs += 1
    print('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
    print('Saving Model...')
    try: save_model(model, args.save_model)
    except: save_checkpoint(model, model.history, args.checkpoint_dir)


def test(model, data, count, epoch):
    val_prog_bar = data.get_prog_bar(data.zip_validation_batches(), 
                                     'validation', epoch)
    torch.no_grad(), model.eval()
    for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, n2f in val_prog_bar:
        count += 1
        model.get_lens_names_frames(n2f)
        model.forward(Variable(pad_src_seq)) # pass thru model
        model.get_cor(model.out_seq, pad_tgt_seq) # get correlation
        out_seq_mm = data.unnormalize_output(model.out_seq, False) #MSE(mm)
        model.get_losses(out_seq_mm, pad_tgt_seq, pad_tgt_seqm)
        if count > 0: # break # CODE TO STOP AND TEST 
            k_nearest_neighbors = get_knn(model.out_seq.detach().numpy(), 
                                          model.seq_lens,
                                          kneighbors = 5)
            pdb.set_trace()
            model.out_seq.detach()
            k_nearest_neighbors
            
        
        model.set_prog_bar_desc(val_prog_bar, count)

def run():
    """
    Run Training and Validation over a number of epochs
    """
    d = get_device(args.cuda)
    data  = LoadInputOutput(args, d)
    model = BuildModel(args, d, data.input_dim, data.output_dim)
    data.get_batches()
    if args.cuda: model = model.cuda()
    if args.load_dir: load_model(model, args.load_dir, d)
    elif os.path.isfile(args.checkpoint_dir): load_checkpoint(model, args, d) 
    else: model.init_history()
    
    epoch = 0 if model.is_new else model.get_epoch()
    if args.train: train_val_loop(model, data, epoch)
    if args.test: test(model, data, 0, epoch)

def main(args):
    run()

if __name__ == '__main__':
    args = get_args()
    main(args)
