
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
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import NearestNeighbors
import os, pdb, argparse, re, sys
from tqdm import tqdm
from collections import defaultdict, OrderedDict, Counter
import torch, torch.nn as nn
import torch.optim as optim
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from numpy.linalg import inv

import itertools
import more_itertools as mit

get_seq_lens = lambda l : [x[-1] for x in l] # seq lens usually at end of list
get_names    = lambda l : [x[0] for x in l] # names usually at beginning of list
dim1_list    = lambda l : [x.shape[1] for x in l]
get_tensors  = lambda t : [i for i in t[0]]

report = 'count:{}, batch_size:{}, loss:{:.4}; mm:{:.4}, frames:{}, cor:{:.4}'

#### hand selected templates from mngu0 data transformed into tract variables ####
tvs = {'velar': np.array([25.831184 , -2.681592 , 16.368872 ,  5.429029 , -6.4213333, 1.349422 , -7.69129  ,  6.394051 , -7.2242336]), 
       'postalveolar': np.array([25.93068   , -2.5815926 , 15.283978  ,  3.6875253 , -7.121334, 0.56355697, -8.791288  ,  0.63655245, -7.6242332 ]), 
       'alveolar': np.array([25.632206 , -2.4815922, 14.716658 ,  4.48851  , -5.621334 ,   1.772168 , -7.391287 ,  3.2768078, -5.1242332]), 
       'dental': np.array([25.603516 , -2.3815928, 15.938946 ,  4.221515 , -6.121334 ,   2.7013173, -6.991289 ,  5.897674 , -2.6242332]), 
       'labiodental': np.array([26.172504 ,  2.2184076, 14.534442 ,  7.4435406, -7.121334 ,  43.794888 , 69.20871  , 11.95761  , -6.6242332]), 
       'bilabial': np.array([17.613914 , -2.7815924, 15.041609 ,  4.657608 , -5.621334 , 5.1692524, -7.2912884,  7.6192145, -7.6242332]), 
       'ii': np.array([25.378927  , -2.7815924 , 16.311037  ,  2.276886  , -6.121334, 0.70774424, -6.991289  ,  5.760368  , -6.7242336 ]), 
       'e': np.array([25.532724 , -2.4815922, 14.205633 ,  5.503099 , -5.621334, 2.6772861, -7.2912884,  6.052591 , -6.6242332]), 
       'a': np.array([25.622255 , -2.4815922, 15.871043 ,  5.61481  , -7.121334 , 3.9926832, -8.291288 ,  6.621691 , -7.6242332]), 
       '@@r': np.array([25.622255 , -2.7815924, 16.015305 ,  4.742157 , -5.621334 , 3.6340206, -7.7912884,  6.0947943, -7.2242336]), 
       'uu': np.array([24.44954  , -3.4815922, 16.62077  ,  0.3961326, -5.621334 , 1.3739431, -8.091288 ,  5.097271 , -7.2242336]), 
       'oo': np.array([23.518715 , -3.4815922, 16.297546 ,  3.2291753, -5.9213333, 4.0727873, -7.2912884,  6.6357594, -7.824234 ]), 
       'uh': np.array([24.290945 , -2.681592 , 15.190786 ,  4.7154617, -6.621334 , 2.5055137, -8.19129  ,  6.536007 , -7.824234 ]), 
       'o': np.array([24.219208 , -3.181592 , 15.692355 ,  3.2781274, -6.621334 , 4.889791 , -8.291288 ,  6.5219393, -7.6242332]), 
       'aa': np.array([25.079872 , -2.681592 , 16.4195   ,  4.0034723, -7.321335 , 4.9458637, -7.5912876,  7.2694426, -8.324234 ]), 
       'ai_ow': np.array([26.172504 , -2.2815924, 15.033297 ,  5.236104 , -6.121334 , 4.7500606, -8.791288 ,  6.8352637, -7.824234 ]), 
       'r': np.array([21.612265 , -3.4815922, 16.082289 ,  4.0034723, -7.321335 , 4.614786 , -7.991289 ,  7.2694426, -8.324234 ])}

def get_args():
    """
    Args define program usage.
    """
    info = 'Script preprocesses data and saves, given i/o directories.'
    parser = argparse.ArgumentParser(info)
    parser.add_argument('--train', action = 'store_true', default = False)
    parser.add_argument('--test',  action = 'store_true', default = False)
    parser.add_argument('--test-lle',  action = 'store_true', default = False)
    
    # LLE
    parser.add_argument('--maxneighbors', type=int, default=12, help='number of bi-directional LSTM layers')
    parser.add_argument('--templates-type', help = 'type of templates to be used in model')
    parser.add_argument('--scale', action = 'store_true', help = 'use normalized EMA in the model')
    
    # DIR
    parser.add_argument('--lsf-dir',  help = 'path to lsf directory')
    parser.add_argument('--mfcc-dir', help = 'path to mfcc directory')
    parser.add_argument('--ema-dir',  help = 'path to ema directory')
    parser.add_argument('--partition-dir', help = 'path to txt files dictating data partitions')
    
    parser.add_argument('--mm', action = 'store_true', help = 'change SI units to mm for distances (only when using raw ema)')
    # KEEP LTJ
    parser.add_argument('--tjl', action = 'store_true', default = False, help = 'keep only tongue, jaw and lip features (TJL)')
    
    # TRACT VARIABLES
    parser.add_argument('--tv', action = 'store_true', default = False, help = 'transform ema vectors tract variables (TV)')
    parser.add_argument('--plot', action = 'store_true', help = 'graph data')
    
    # HATSWAP
    parser.add_argument('--hatswap', action = 'store_true', default = False)
    
    # ARCHITECTURE
    parser.add_argument('--bilstm-layers', type=int, default=2, help='number of bi-directional LSTM layers')

    # BATCH DETAILS
    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle batches')
    parser.add_argument('--batch-size', type=int, default=6, help='shuffle batches')

    # GPU    
    parser.add_argument('--cuda', action = 'store_true', default=False, help='GPU support')

    # MODEL
    parser.add_argument('--load-dir', help='load a saved model from directory')
    parser.add_argument('--save-dir', help='save a final model in directory')

    # TRAINING DETAILS
    parser.add_argument('--checkpoint-dir', default = 'checkpoints', help = 'checkpoint directory to load/save model from/in')
    parser.add_argument('--patience', type = int, default = 10, help='save model after n epochs without improvement')
    parser.add_argument('--save-internal', default = 5, help = 'epoch arg to save the model parameters every *n* times')

    
    return parser.parse_args()

def lineMagnitude(x1, y1, x2, y2):
    #https://maprantala.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
    lineMagnitude = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return lineMagnitude

def distancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    #https://maprantala.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
    lineMag = lineMagnitude(x1, y1, x2, y2)
    
    if lineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
    
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (lineMag * lineMag)
    
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
    
    return DistancePointLine

def get_cl(arr, index, ema_t_l):
    """
    calculates constriction length (CL):
        T?CL[n] = median{T?x for all T?x} - T?x[n]
    """
    return torch.median(torch.cat(ema_t_l)[:,index]) - arr[:,index]

def get_cd(arr, index_x, index_y, palate_line_segments):
    """
    calculates tongue constriction degree of td, tb, tt
        T?CD[n] = Min for all x {sqrt((T?x[n]-x)**2-(T?y[n]-pal(x))**2)}
    step 1: half of the convex hull (upper half) to get 3 nearest neighbors (line segments)
    knn is used to speed up computation
    """
    knn = NearestNeighbors(3)
    knn.fit(X = np.concatenate(palate_line_segments[0::2]))
    knn_sample = knn.kneighbors(arr[:,index_x:index_y], return_distance = False)
    constriction_list = []
    for index, knn in zip(range(arr.shape[0]), knn_sample):
        pxy = arr[index,index_x:index_y].numpy()
        constriction_candidates = []
        try: 
            for segments in [palate_line_segments[i-1] for i in knn]:
                yx1, yx2 = segments[0], segments[1]
                constriction_candidates.append(distancePointLine(pxy[0], pxy[1], yx1[0], yx1[1], yx2[0], yx2[1]))
        except: pdb.set_trace()
        constriction_list.append(min(constriction_candidates))
    return FloatTensor(constriction_list)
    
def get_la(arr):
    """
    calculates lip aperture (euclidean distance between UL / LL): 
        pertinent indexes:
        LLx : 8, ULx: 10, LLy : 9, ULy : 11
        LA[n] = sqrt((LLx[n]-ULx[n])**2+(LLy[n]-ULy[n])**2)
    returns lip aperture
    """
    return torch.sqrt((arr[:,8]-arr[:,10])**2+(arr[:,9]-arr[:,11])**2)

def get_lp(arr, ema_t_l):
    """
    calculates (lower) lip protrusion (displacement from median position):
        LP[n] : LLx[n] - median{LLx for all n}
    """
    return arr[:,8] - torch.median(torch.cat(ema_t_l)[:,8]).item()
    
def get_ja(arr):
    """
    calculates jaw aperture (euclidean distance between UL / J (LI)):
        pertinent indexes:
        Jx : 6, ULx: 10, Jy : 7, ULy : 11
    """
    return torch.sqrt((arr[:,6]-arr[:,10])**2+(arr[:,7]-arr[:,11])**2)
def get_tvs(v, palate_line_segments, ema_t_l):
    """
    v is a matrix of EMA and dimensionality (nsamples * 12)
    returns FloatTensor tvs
    """
    tvs_mat = torch.stack([get_la(v), 
                           get_lp(v, ema_t_l),
                           get_ja(v), 
                           get_cd(v,0,2, palate_line_segments), 
                           get_cl(v,0, ema_t_l), 
                           get_cd(v,2,4, palate_line_segments), 
                           get_cl(v,2, ema_t_l), 
                           get_cd(v,4,6, palate_line_segments), 
                           get_cl(v,4, ema_t_l)]).transpose(dim0=1,dim1=0)
    return tvs_mat
    
def get_palate_line_segments(speaker_tensor, name, mt_hull, plot_bool):
    """
    data: unordered x,y coordinates from tongue coils T1/TT, T2/TB, T3/TD
    and data from T1/TT in the case of 'faet0', 'mjjn0', 'fsew0' are used 
    to create a convex hull estimate as to where the palate is
    in order to calculate Tract Variables.
    """
    t_data = speaker_tensor[:,:6].reshape(-1,2)
    hull_y = t_data[mt_hull.vertices, 1] # convex hull y coords
    hull_x = t_data[mt_hull.vertices, 0] # convex hull x coords
    # min/max x coordinates
    iminx = np.where(hull_x==min(hull_x))[0].item()
    imaxx = np.where(hull_x==max(hull_x))[0].item()
    # geometric logic: high hull values (mean(y)) define palate/larynx
    if mean(hull_y[iminx:imaxx+1]) > mean(np.concatenate((hull_y[imaxx:],hull_y[:iminx+1]))): 
        upper_hull_y, upper_hull_x = hull_y[iminx:imaxx+1], hull_x[iminx:imaxx+1]
    elif mean(hull_y[iminx:imaxx+1]) < mean(np.concatenate((hull_y[imaxx:],hull_y[:iminx+1]))): 
        upper_hull_y, upper_hull_x = np.concatenate((hull_y[imaxx:],hull_y[:iminx+1])), np.concatenate((hull_x[imaxx:], hull_x[:iminx+1]))
    elif mean(hull_y[imaxx:iminx+1]) > mean(np.concatenate((hull_y[iminx:],hull_y[:imaxx+1]))): 
        upper_hull_y, upper_hull_x = hull_y[imaxx:iminx+1], hull_x[imaxx:iminx+1]
    elif mean(hull_y[imaxx:iminx+1]) < mean(np.concatenate((hull_y[imaxx:],hull_y[:iminx+1]))): 
        upper_hull_y, upper_hull_x = np.concatenate((hull_y[iminx:],hull_y[:imaxx+1])), np.concatenate((hull_x[iminx:], hull_x[:imaxx+1]))
    upper_hull = np.array([upper_hull_y,upper_hull_x]).transpose()
    
    print('selected {}'.format(name))
    ## Take max y of points 10th-70th x percentile of tongue tip.
    tt_quantized = np.round(speaker_tensor[:,4:6].reshape(-1,2),2)
    
    if name in {'faet0', 'fsew0'}: pxy = np.round(np.percentile(tt_quantized[:,:], q=[10,20,30,40,50,60,70], axis=0),2)
    if name in {'msak0'}: pxy = np.round(np.percentile(tt_quantized[:,:], q=[2,10], axis=0),2)
    if name in {'mngu0'}: pxy = np.round(np.percentile(tt_quantized[:,:], q=[0.2, 0.5, 4.4, 9,20], axis=0),2)
    if name in {'maps0'}: pxy = np.round(np.percentile(tt_quantized[:,:], q=[10,20,30,40,50,60,70], axis=0),2)
    if name in {'maps0', 'mngu0', 'msak0','faet0', 'fsew0'}:
        px, _ = pxy[:,0], pxy[:,1]
        xp_indices = [np.where(x==tt_quantized[:,0]) for x in px]
        candidates_list = [tt_quantized[indices] for indices in xp_indices]
        maxpxy = [np.argmax(l[:,1]) for l in candidates_list]
        alveolar_ridge = np.concatenate([c[i] for c,i in zip(candidates_list, maxpxy)]).reshape(-1,2) + 0.5
        keep = [i[0] or i[1] for i in zip(np.array(upper_hull[:,1]>max(alveolar_ridge[:,0])),np.array(upper_hull[:,1] < min(alveolar_ridge[:,0])))]
        upper_hull = upper_hull[keep]
        alveolar_and_hull = np.array((upper_hull[:,1],upper_hull[:,0])).T.tolist()
        # remove indicies between xmin and xmax of alveolar indices
        alveolar_and_hull.extend(np.array((alveolar_ridge[:,0],alveolar_ridge[:,1])).T.tolist())
        alveolar_and_hull = sorted(alveolar_and_hull, key=lambda x:x[0])
        alveolar_and_hull = np.array(alveolar_and_hull)
    if plot_bool:
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.weight"] = "regular"
        plt.rcParams["font.size"] = "14"
        heatmap, xedges, yedges = np.histogram2d(t_data[:,0], t_data[:,1], bins=(512,512))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.title('{}'.format(name))
        plt.ylabel('y')
        plt.xlabel('x')
        plt.imshow(heatmap.T, origin='lower',
                   extent=extent, interpolation='nearest',
                   cmap=plt.get_cmap('viridis'))
        if plot_bool and name in {'maps0', 'mngu0', 'msak0','faet0', 'fsew0'}:
            plt.plot(alveolar_ridge[:,0], alveolar_ridge[:,1], 'x')
            plt.plot(alveolar_and_hull[:,0], alveolar_and_hull[:,1], '--c', lw=2)
        plt.plot(upper_hull_x, upper_hull_y, '--b', lw=2)
        plt.show()
        pdb.set_trace()
    duplicated_points = np.array([list(mit.flatten(i)) for i in mit.windowed(upper_hull, n=2)])
    # create line segments to calculate minimum distance from...
    palate_line_segments = [(np.array([i[1],i[0]]), np.array([i[3],i[2]])) for i in duplicated_points]
    return palate_line_segments


class LoadInputOutput():
    def __init__(self, args, device):
        self.get_data_partition(args.partition_dir)
        self.plot_bool = args.plot
        if args.lsf_dir: self.in_dict = self.load_data(args.lsf_dir, 'in_dict')
        if args.mfcc_dir: self.in_dict = self.load_data(args.mfcc_dir, 'in_dict')
        self.ema_dict = self.load_data(args.ema_dir, 'ema_dict')
        if args.tjl: self.keep_tjl() # keep [T3xy, T2xy, T1xy, Jxy, ULxy, LLxy] (12 feat. vector)
        self.get_name_set()
        self.transform_data(args.scale, args.mm)  # transform (scaling and/or Tract Variables)
        self.device   = device
        self.shuffle  = args.shuffle
        self.input_dim, self.output_dims = self.get_io_dims(args)
        print('input dimensionality of feats:\t\t{} '.format(self.input_dim))
        print('output dimensionality of emas:\t\t{}'.format(self.output_dims))

    def get_name_set(self):
        self.name_set = set()
        for k,v in self.ema_dict['channel_ema'].items(): self.name_set.add(re.sub('_\d+\w?','',k))
    
    def get_data_partition(self, partition_dir):
        """
        Get data partition from file.
        """
        self.partitions = defaultdict(set)
        for file in os.listdir(partition_dir):
            if not file.endswith("files.txt") or file.endswith("files"): 
                continue
            with open(os.path.join(partition_dir, file), 'r') as f:
                dataset_name = file.split('.')[0]
                sample_names = set([i.strip('\n') for i in f.readlines()])
                self.partitions[dataset_name] = sample_names
    
    def keep_tjl(self):
        articulator_order = \
            ['T3x', 'T3y', 'T2x', 'T2y', 'T1x', 'T1y', 'Jx', 'Jy', 'ULx', 'ULy', 'LLx', 'LLy']
        name_to_channel_name = \
            {re.sub('_\d+\w?','',k):[re.sub('LI','J',c[1].decode('utf-8')) for c in v] 
             for k,v in self.ema_dict['channel_names'].items()}
        name_to_new_order = {k:[v.index(p) for p in articulator_order] for k,v in name_to_channel_name.items()}
        self.ema_dict['channel_ema'].update({n:t[:,name_to_new_order[k]] for n,t in self.ema_dict['channel_ema'].items()
                                                for k,v in name_to_new_order.items() if k in n})
        self.ema_dict['channel_names'].update({n:[p[1].decode('utf-8') for p in t[name_to_new_order[k]]]
                                                 for n,t in self.ema_dict['channel_names'].items() 
                                                 for k,v in name_to_new_order.items() if k in n})

    def get_io_dims(self, args):
        i = bool(args.lsf_dir or args.mfcc_dir)
        o = bool(args.ema_dir)
        if i: n_input_dims  = set(dim1_list(self.in_dict['channel_feats'].values()))
        # extract each spkr and corresponding dimension for each speaker
        if o: n_output_dims = {re.sub('_\d+\w?','',k):v.shape[1]
                               for k,v in self.ema_dict['channel_ema'].items()}
        spkrnames = set([re.sub('_\d+\w?','',n) for k,v in self.partitions.items() for n in v])
        n_output_dims = {k:v for k,v in n_output_dims.items() if {k}.intersection(spkrnames)}
        if i and o:
            if len({len(n_output_dims),len(n_input_dims)})==1: return n_input_dims.pop(), n_output_dims
            elif len(n_output_dims)>1: return n_input_dims.pop(), n_output_dims
        elif i and len(n_input_dims)!=1: return n_input_dims, None
        elif o and len(n_output_dims)!=1: return None, n_output_dims

    def load_means(self):
        return np.array([float(i[0])*10**int(i[1]) for i in self.norm])

    def load_4xSTDs(self):
        return np.array([4*(float(i[0])*10**int(i[1])) for i in self.norm])

    def load_ema_norm(self, data_dict, key, file_path):
        # take name from directory to create norm files in future
        ema_norm_str = '(\d+\.\d+)e([\-\+]\d+)'
        with open(file_path , 'r') as f:
            self.norm = re.findall(ema_norm_str, f.readlines()[0])# ORDER: X,Y; T3, T2, T1, J, LL, UL
            if file_path.endswith('means.txt'):
                data_dict.update({key:FloatTensor(self.load_means())})
            elif file_path.endswith('stds.txt'):
                data_dict.update({key:FloatTensor(self.load_4xSTDs())})
        return data_dict
    
    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]
    
    def transform_data(self, scale_bool, mm_bool):
        """
        Creates scaled data self.scaler_dict and self.name_set from data input.
        MinMaxScaler methods: .fit_transform, .tranform or .inverse_transform
        also .data_max_ and .data_min_
        """
        # reorganise tensors to get norm parameters
        #if tract_variable_bool: from scipy.spatial import ConvexHull
        if scale_bool: from sklearn.preprocessing import MinMaxScaler
        self.scaler_dict = defaultdict(int)
        print('sanity check: are s.d\'s roughly the same?')
        for name in self.name_set:
            if name == 'fjmw0': continue # DELETE ALL fjmw0
            if mm_bool and name in {'maps0', 'msak0','ss2404','fsew0','fjmw0','falh0', 'faet0', 'mjjn0'}:
                #### MOCHA DATA (10^-5) is a factor of 10^3 larger than mngu0 (10^-2)
                #### change to standard (mm).
                self.ema_dict['channel_ema'].update({k:v/100 for k,v in self.ema_dict['channel_ema'].items() 
                                                     if name in k})
            elif mm_bool and name in {'mngu0'}: # If data is mngu0 change to standard (mm).
                self.ema_dict['channel_ema'].update({k:v*10 for k,v in self.ema_dict['channel_ema'].items() 
                                                     if name in k})
            # all data should now be in mm.
            # delete all samples with nan.
            self.ema_t_l = self.delete_nan_ema(name)
            ema_channel_names = self.ema_dict['channel_names'][name+'_1']
            namestds = [np.sqrt(np.var(i)) for i in torch.cat(self.ema_t_l,0).numpy().transpose()]
            assert len(namestds) == len(ema_channel_names)
            print('{} stds (mm):'.format(name),', '.join(['{}: {:.4f}'.format(channel_name,std) for channel_name, std in zip(ema_channel_names, namestds)])+'.')
            #if np.mean(namestds)>5: continue  # ONLY CONTINUE FOR THE MOMENT
#            if tract_variable_bool:
#                # convert speaker samples to TVs.
#                t_data = speaker_tensor[:,:6].reshape(-1,2)
#                self.mt_hull = ConvexHull(t_data)

#                # Take only tongue indicies for convex hull
#                # self.convex_hull_points = tongue_data[self.mt_hull.simplices,:]
#                if name not in {'mngu0','maps0','msak0','fsew0','faet0'}: continue
#                try:
#                    pbar = tqdm({k:v for k,v in self.ema_dict['channel_ema'].items() if name in k}.items())
#                    for k,v in pbar:
#                        pbar.set_description("creating Tract Variables for {}. {} frames".format(k, v.shape[0]))
#                        tv_ema = self.get_la(v),
#                                              self.get_lp(v), 
#                                              self.get_ja(v),
#                                              self.get_cd(v,0,2), #tdcd
#                                              self.get_cl(v,0),   #tdcl
#                                              self.get_cd(v,2,4), #tbcd
#                                              self.get_cl(v,2),   #tbcl
#                                              self.get_cd(v,4,6), #ttcd
#                                              self.get_cl(v,4)]).transpose(dim0=1,dim1=0).numpy()  #ttcl
#                        np.min(tv_ema)
#                        channel_names = np.array(['LA',  'LP',  'JA',  'TDCD', 'TDCL',
#                                                  'TBCD','TBCL','TTCD','TTCL'])
#                        np.save(os.path.join('data/ema_tv', 'channel_time', k),
#                            self.ema_dict['channel_time'][k], allow_pickle = True)
#                        np.save(os.path.join('data/ema_tv', 'channel_ema', k),
#                            tv_ema, allow_pickle = True)
#                        np.save(os.path.join('data/ema_tv', 'channel_names', k),
#                            channel_names, allow_pickle = True)
#                except:
#                    pbar.postfix('{} failed'.format(k))

            if scale_bool:
                self.scaler = MinMaxScaler(feature_range = (-1, 1))
                self.scaler.fit_transform(torch.cat(self.ema_t_l, 0)) # info preserved
                self.ema_dict["scaler"].update({name:self.scaler})
                scaled_ema = {key:self.scaler.transform(x) for key, x in self.ema_dict['channel_ema'].items()
                              if re.sub('_\d+\w?','',key)==name}
                self.ema_dict['channel_ema'].update(scaled_ema)
    

    def delete_nan_ema(self, name):
        """
        EMA should be nan free. However, this function is used to delete EMA 
        files with nans before transforming.
        """
        self.ema_t_l = []
        d_nan = {k:v for k,v in self.ema_dict['channel_ema'].items() if np.where(np.isnan(v))[0].shape == np.array([]).shape}
        nan_keys = set(self.ema_dict['channel_ema'].keys()).difference(set(d_nan.keys()))
        for k in nan_keys:
            del self.ema_dict['channel_ema'][k], self.ema_dict['channel_names'][k]
            del self.ema_dict['channel_time'][k]
        del nan_keys, d_nan
        return [v for k,v in self.ema_dict['channel_ema'].items() if name in k]
    
    def load_data(self, directory, name):
        """
        This function loads data from directories with different structures
        key: d = dimensions, f = frames, lsfd OR mfcc = <features>,
             n = names
        
        d1         |d2                 |data (vector/string)
        ___________|___________________|___________________________
        <features>:|channel_<features> |f
                   |channel_names      |d
                   |channel_time       |f
        ___________|___________________|___________________________                   
        ema:       |channel_ema        |f
                   |channel_names      |d
                   |channel_time       |f
                   |scale_data *       |n
        
        This function produces the following dictionaries:
        
        in_dict  == {'channel_<feature>':{<sample>: tensor(<features1>,...)},
                     'channel_names'    :[(0, <featurename>), ...],
                     'channel_time'     :{<sample>: tensor(0.005, ...)}
                     }

        ema_dict == {'channel_ema'      :{<sample>: tensor(<ema_features1>, ...},
                     'channel_names'    :[(0, <place_of_articulation>), ...],
                     'channel_time'     :{<sample>: tensor(0.005, ...)},
                     'scale_data'       :MinMaxScaler(),
                     }
        """
        data_dict = defaultdict(dict)
        for d1 in os.listdir(directory):
            if not ("channel" in d1 or "norm" in d1): continue
            print('loading data from {}/{} to {}'.format(directory, d1, name))
            for d2 in os.listdir(os.path.join(directory, d1)):
                key = d2.split('.')[0]
                file_path = os.path.join(directory, d1, d2)
                if re.match('ema_(means|stds)\.txt',d2): 
                    data_dict = self.load_ema_norm(data_dict, key, file_path)
                    continue
                try: array = np.load(file_path, allow_pickle = True)
                except: pdb.set_trace()
                if d1.endswith('channel_names'): data_dict['channel_names'].update({key:array})
                elif re.match('channel_(lsfd|mfcc)',d1): 
                    data_dict['channel_feats'].update({key:FloatTensor(array)})
                else: data_dict[d1].update({key:FloatTensor(array.astype('float32'))})
        return data_dict

    def batch_source_data(self):
        self.btrs = self.batch(self.in_dict['channel_feats'], 'trainfiles')
        self.bvas = self.batch(self.in_dict['channel_feats'], 'validationfiles')
        self.btes = self.batch(self.in_dict['channel_feats'], 'testfiles')
    
    def batch_target_data(self):
        self.btrt = self.batch(self.ema_dict['channel_ema'], 'trainfiles')
        self.bvat = self.batch(self.ema_dict['channel_ema'], 'validationfiles')
        self.btet = self.batch(self.ema_dict['channel_ema'], 'testfiles')
    
    def get_target_data_mm(self, name):
        self.btrtm = self.unnormalize_output(name, get_tensors(self.btrt), True)
        self.bvatm = self.unnormalize_output(name, get_tensors(self.bvat), True)
        self.btetm = self.unnormalize_output(name, get_tensors(self.btet), True)

    def get_batches(self, batch_size, test_lle):
        """
        batch_size is int(), test_lle is boolean
        define TRAINING (tr) & TESTING (te) & VALIDATION (va) sets for...
        SOURCE set (s) & TARGET set (t)
        Normalize outputs to millimeters (for calculating RMSE in mm)
        
        b(tr|va|te)(s|t) : a tuple containing [0] tensors for batch processing
                                              [1] names of samples in each batch
        
        """
        self.sample_names = sorted([k for k in self.ema_dict['channel_ema'].keys()])
        report = 'Min batch target size = {}\nBatching data...'
        print(report.format(batch_size))
        self.get_l2n()
        self.batch_source_data()
        self.batch_target_data()
        r_names = set(j.split('_')[0] for i in self.partitions.values() for j in i)
        for name in self.name_set.intersection(r_names): 
            self.get_target_data_mm(name)          
        
    def reorder(self, t, i, tup=False): 
        if tup : return ([t[0][n] for n in i],[t[1][n] for n in i])
        else: return [t[n] for n in i]
    
    def shuffle_batches(self):
        """
        Shuffles batches using random.shuffle:
            get len of training/validation batches -> 0,1, ... len(batch).
            new order for tr, te & va by shuffling
            reorder
        """
        tr = np.arange(len(self.btrs[0]))
        va = np.arange(len(self.bvas[0]))
        te = np.arange(len(self.btes[0]))
        np.random.shuffle(tr), np.random.shuffle(va), np.random.shuffle(te)
        tr, va, te = tr.tolist(), va.tolist(), te.tolist()
        self.btrs  = self.reorder(self.btrs,  tr, tup = True)
        self.bvas  = self.reorder(self.bvas,  va, tup = True)
        self.btes  = self.reorder(self.btes,  te, tup = True)
        self.btrt  = self.reorder(self.btrt,  tr, tup = True)
        self.bvat  = self.reorder(self.bvat,  va, tup = True)
        self.btet  = self.reorder(self.btet,  te, tup = True)
        self.btrtm = self.reorder(self.btrtm, tr, tup = False)
        self.bvatm = self.reorder(self.bvatm, va, tup = False)
        self.btetm = self.reorder(self.btetm, te, tup = False)
    
    def get_l2n(self):
        """
        Sequences should not change length between neural net input and output.
        Use ema signal to define sequence lengths.
        This method creates and updates (1) an ordered set dict and 
        (2) an ordered int ddict
        1. self.l2n: set ddict: {length : {name1, ...} ...}
        2: self.n2l: int ddict: {name   : length, ...}
        """
        self.n2l, self.l2n = defaultdict(int), defaultdict(set)
        self.n2l.update({k:v.shape[0] for k,v in self.ema_dict['channel_ema'].items()})
        for k, s in self.n2l.items(): self.l2n[s].add(k)
        self.n2l = dict(OrderedDict(sorted(self.n2l.items(), key=lambda x:x[1])))
        self.l2n = dict(OrderedDict(sorted(self.l2n.items(), key=lambda x:x[0])))

    def batch(self, d, partition):
        """
        Make and pad a batch using data dictionary
            Aim: get samples in a batch >= args.batch_size (ideal batch size)
            returns (padded_tensors for cpu or gpu, samplenames_to_lengths)
        """
        #### SEPARATE NAMES HERE IN FUTURE
        batches, minibatch, =  [], defaultdict(list)
        batch_n2id2l = defaultdict(dict)
        n2l = []
        cut_partition = set([re.sub("^(\w+)_\d+\w*$",r"\1",name) for name in self.partitions[partition]])
        for length, names in self.l2n.items():
            relevant = names.intersection(self.partitions[partition])
            if relevant: # filter only ids appearing in partition file
                trunc_names = set([re.sub("^(\w+)_\d+\w*$",r"\1",name) for name in names]).intersection(cut_partition)
                for name in trunc_names:
                    minibatch[name].extend([(n,length) for n in relevant if re.sub("^(\w+)_\d+\w*$",r"\1",n)==name])
                    if len(minibatch[name]) >= args.batch_size:
                        batch_n2id2l[name].update({length:sorted(minibatch[name], 
                                                             reverse = True, 
                                                             key = lambda x:x[1])})
                        minibatch[name] = [] 
        for name in trunc_names:
            if len(minibatch) > 0:
                minibatch[name].extend([(n,length) for n in relevant if re.sub("^(\w+)_\d+\w*$",r"\1",n)==name])
                batch_n2id2l[name].update({length:sorted(minibatch[name],reverse = True, 
                                                         key = lambda x:x[1])})
        # Here, batches should be in spkrname groups.
        # All spkrname groups should be of the same dimensionality
        # local n2l is created per batch after partition
        for name, dictionary in batch_n2id2l.items():
            for max_l, n_l in dictionary.items():
                if n_l==[]: continue # supress error
                try:
                    t = [FloatTensor(d[n]).to(self.device) for n, _ in n_l]
                    batches.append(pad_sequence(t, batch_first=True))
                except: 
                    pdb.set_trace()
                n2l.append([(n, d[n].shape[0]) for n, l in n_l])
        return batches, n2l

    def return_train_valid(self, in_dict):
        t, v, = defaultdict(str), defaultdict(str)
        t = {k:in_dict[k] for k in self.sample_names[:self.val_idx]}
        v = {k:in_dict[k] for k in self.sample_names[self.val_idx:]}   
        return t, v

    def unnormalize_output(self, name, batches, batches_set = True):
        unnormalized_data = []
        scaler = self.ema_dict['scaler'][name]
        if batches_set==True: # A WHOLE BATCH
            for b in batches:
                batch = torch.stack([FloatTensor(scaler.inverse_transform(s.detach())) for s in b])
                unnormalized_data.append(batch)
            return unnormalized_data
        else: # minibatch (len == batchsize)
            return torch.stack([FloatTensor(scaler.inverse_transform(s.detach())) for s in batches])
        
    def zip_training_batches(self):
        """
        Train args in zip: 1. src; 2. tgt; 3. tgt (mm), 4. names & sizes
        """
        if self.shuffle: self.shuffle_batches()
        return zip(self.btrs[0], self.btrt[0], self.btrtm, self.btrt[1])

    def zip_validation_batches(self):
        """
        Val args in zip: 1. src; 2. tgt; 3. tgt (mm), 4. names & sizes
        """
        if self.shuffle: self.shuffle_batches()
        return zip(self.bvas[0], self.bvat[0], self.bvatm, self.bvat[1])

    def zip_testing_batches(self):
        """
        Test args in zip: 1. src; 2. tgt; 3. tgt (mm), 4. names & sizes
        """
        if self.shuffle: self.shuffle_batches()
        return zip(self.btes[0], self.btet[0], self.btetm, self.btet[1])

    def get_prog_bar(self, zipped_batches, batches_type, epoch):
        if batches_type=='training': t = len(self.btrs[0])
        if batches_type=='validation': t = len(self.bvas[0])
        if batches_type=='testing': t = len(self.btes[0])
        bar = tqdm(iterable = zipped_batches, 
                   total = t,
                   desc = '| {} epoch {:03d}'.format(batches_type, epoch),
                   leave = False,
                   disable = False)
        return bar

class BuildModel(nn.Module):
    def __init__(self, args, device, input_dim, output_dims):
        super(BuildModel, self).__init__()
        self.args          = args
        self.device        = device
        self.output_dims   = output_dims
        self.criterion     = nn.MSELoss()
        self.criterion_mm  = nn.MSELoss()
        if args.test_lle: return None
        self.input_dim     = input_dim
        self.hidden_dim    = 256
        self.fc_hidden_dim = 512
        self.bilstm_layers = args.bilstm_layers
        self.batch_size    = args.batch_size
        # L1 lstm cell: should use sigmoid as R activation and tanh as final
        self.bidirlstm = nn.LSTM(input_size    = self.input_dim, 
                                 hidden_size   = self.hidden_dim,
                                 bidirectional = True,
                                 num_layers    = self.bilstm_layers)
        self.fc_dict = dict() # Utilize dict for hatswap architecture
        if args.hatswap:
            for name, dim in self.output_dims.items():
                fc = nn.Linear(self.fc_hidden_dim, dim)
                self.fc_dict.update({name:fc})
        elif len(set([v for v in self.output_dims.values()])) > 1: 
            print('sets of different length. Either define different fcs (hatswap), or use same dimensionality for normalized vector')
        else: self.fc_dict.update({'normalized':nn.Linear(self.fc_hidden_dim)})
        self.hc = None
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.is_new = False 
        ##TODO CHANGE assumption tht model loaded from chkpnt til proven wrong

    def forward(self, padded_seq_batch):
        """
            padded_seq_batch: padded sequence
            hc: hidden and cell states
            tuple of hidden and cell state
        """
        #hatswapping code:
        if len(self.get_spkr_names()) > 1:
            print('Error. Multiple names in batch, single name expected.')
            if self.args.hatswap:
                print('Single names ONLY (per batch) for hatswap training.')
                sys.exit()
        elif self.get_spkr_names().pop() not in self.fc_dict.keys():
            print('Error. Name is not present in fc_dict')
        else: self.current_name = self.get_spkr_names().pop()
        
        self.get_len(padded_seq_batch.shape[0])
        self.out_seq = torch.empty((self.sequence_len, self.batch_size, self.output_dims[self.current_name]))
        
        packed_seq_batch =  pack_padded_sequence(padded_seq_batch, lengths = self.seq_lens, batch_first=True)
        self.hc = self.init_hidden(2*self.bilstm_layers, max(packed_seq_batch[1].tolist()))
        packed_outputs, self.hc = self.bidirlstm(packed_seq_batch, self.hc)
        lstm_out, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        self.out_seq = self.fc_dict[self.current_name](lstm_out)            
                
    
    def get_spkr_names(self):
        return {re.sub('_\d+\w?','',name) for name in self.names}
    
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
    
    def get_cor(self, compute_over, output, target):
        """
        function for correlations between target and output in a batch
        """        
        frames = output.shape[0] if not type(output) == torch.Tensor else None
        self.av_cor = float(0)
        if compute_over == 'single':
            vo = [torch.mean(a)-a for a in output[:frames,:].transpose(dim0=1,dim1=0)]
            vt = [torch.mean(a)-a for a in target[:frames,:].transpose(dim0=1,dim1=0)]
            numerator = torch.sum(torch.stack(vo).transpose(dim0=1,dim1=0) * torch.stack(vt).transpose(dim0=1,dim1=0))
            denominator = (torch.sqrt(torch.sum(torch.stack(vo).transpose(dim0=1,dim1=0) ** 2)) * torch.sqrt(torch.sum(torch.stack(vt).transpose(dim0=1,dim1=0) ** 2)))
            if denominator.item() == 0.0: cor = 0
            else: cor = numerator/denominator
            self.cor = cor.item()
        else: 
            for frames, o, t in zip(self.seq_lens, output, target):
                vo = [torch.mean(a)-a for a in o[:frames,:].transpose(dim0=1,dim1=0)]
                vt = [torch.mean(a)-a for a in t[:frames,:].transpose(dim0=1,dim1=0)]
                numerator = torch.sum(torch.stack(vo).transpose(dim0=1,dim1=0) * torch.stack(vt).transpose(dim0=1,dim1=0))
                denominator = (torch.sqrt(torch.sum(torch.stack(vo).transpose(dim0=1,dim1=0) ** 2)) * torch.sqrt(torch.sum(torch.stack(vt).transpose(dim0=1,dim1=0) ** 2)))
                if denominator.item() == 0.0: cor = 0
                else: cor = numerator/denominator
                self.cor = cor.item()
        
    def get_lens_names_frames(self, names_to_lengths):
        """
        computed batch-wise
        """
        self.seq_lens = get_seq_lens(names_to_lengths)
        self.names = get_names(names_to_lengths)
        self.frames = sum(self.seq_lens)

    def get_losses(self, out_seqm, pad_tgt_seqm, pad_tgt_seq = torch.tensor([np.nan])):
        if pad_tgt_seq.shape == pad_tgt_seqm.shape:
            self.loss       = self.criterion(self.out_seq, pad_tgt_seq)
            self.rmseloss   = torch.sqrt(self.loss)
        else: self.loss     = None
        self.rmseloss_mm = np.sqrt(self.criterion_mm(FloatTensor(out_seqm), pad_tgt_seqm).item()) #cm -> mm
        

    def propagate_loss(self):
        self.rmseloss.backward()
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
        self.loss = self.loss.item() if self.loss else None
        s = tuple([sample_number, self.frames, self.loss, 
                   self.rmseloss_mm, self.cor, self.current_name])
        if self.history[phase].get(epoch): self.history[phase][epoch].append(s)
        else: self.history[phase].update({epoch:[s]})

    def set_prog_bar_desc(self, prog_bar, c):
        batch_size = len(self.names)
        prog_bar.set_postfix(count = c, batchsz = batch_size, 
                             ls = self.loss, 
                             ls_mm = self.rmseloss_mm, 
                             frames = self.frames, 
                             cor = self.cor,
                             name = self.current_name)
        
    def get_epoch(self):
        """
        since checkpoints/models are only saved after validation,
        we only look into validation epoch set here.
        If process was cancelled in 1st train, start new history
        """
        if dict(self.history.get('validation'))=={}: 
            self.init_history()
            self.__init__(self.args, self.input_dim, self.output_dim)
            return 0
        else: return max(dict(self.history.get('validation')).keys())
        
    def compute_batch_averages(self, phase, t_len = None, kneighbors = None):
        """
        average loss (norm & mm), and cor calculated over batch for phase
        """
        report = '{0}, {1}, {2:.5f}, {3:.5f}'
        #samples_per_speaker = Counter([t[-1] for t in dict(self.history[phase])[self.epoch]])
        frames_per_speaker = defaultdict(int)
        for t in [(t[-1],t[-5]) for t in dict(self.history[phase])[self.epoch]]: 
            frames_per_speaker[t[0]]+=t[1]
        averaged_history = {}        
        for name_g in set([t[-1] for t in dict(self.history[phase])[self.epoch]]):
            frames_in_batch    = frames_per_speaker[name_g]
            b_lm, b_l, b_c = float(0), float(0), float(0)
            for n, f, l, lm, c, name in dict(self.history[phase])[self.epoch]:
                if name != name_g: continue
                b_c  += (c * f)  / (frames_in_batch)
                b_l  += (l * f)  / (frames_in_batch)  if l != None else 0
                b_lm += (lm * f) / (frames_in_batch)
            averaged_history.update({name: (b_l, b_c, b_lm)})
            if phase == 'test-lle': 
                print('{}, {}, '.format(t_len, kneighbors), 
                      report.format(name_g, phase, b_lm, b_c))
            else : 
                print(report.format(name_g, phase, b_lm, b_c))
        
        del self.history[phase][self.epoch]
        for name, data in averaged_history.items():
            self.history[phase].update({name:data})
        if phase=='validation': return b_l
        
    def suppress_nans(self, arr):
        valid_frames   = np.delete(np.arange(0,arr.shape[0]), np.argwhere(np.isnan(arr)))
        return arr[valid_frames,:]
        
    def lle_reconstruction(self, arr, templates, k):
        """
        Get LLE encoding of EMA in terms of k-nearest neighbors.
        See paper:
            https://cs.nyu.edu/~roweis/lle/publications.html
        Or algorithm page:
            https://cs.nyu.edu/~roweis/lle/algorithm.html
        In this code neighbors are articulator configurations known as templates.
        """
        template_array = np.array(list(templates.values()))
        template_keys = [key for key in templates.keys()]
        arr = self.suppress_nans(arr) # suppress nans: delete frames with nans
        knn = NearestNeighbors(n_neighbors = k, algorithm = 'kd_tree')
        knn.fit(X = template_array)
        k_nn, k_ni = knn.kneighbors(arr, return_distance=True)
        data = []
        for i in range(arr.shape[0]):
            top_keys = [template_keys[j] for j in k_ni[i]]
            new_order = sorted(range(len(top_keys)), key=lambda k: top_keys[k])
            top_keys = np.array(top_keys)[new_order] # order labels
            Z = template_array[k_ni[i,:],] - matlib.repmat(arr[i,:], k, 1)
            C = Z @ Z.T
            if np.linalg.det(C) == 0: 
                # regularize Cov Mat "C" if det(C) == 0
                # det(C) == 0 is usually the case for k > d
                invC = inv(C+np.ones(C.shape)*0.001*np.matrix.trace(C))
                Wi = invC.dot(np.ones((k, 1)))
            else: Wi = inv(C).dot(np.ones((k, 1)))
            Wi_norm = Wi/sum(Wi)
            scaled_weights = Wi/sum(abs(Wi))
            if len(new_order) > 1: # order weights only for k > 1
                scaled_weights = scaled_weights.squeeze()[new_order]
            reconstruction = np.dot(template_array[k_ni[i,:],].transpose(), Wi_norm).squeeze()
            data.append((reconstruction, top_keys, scaled_weights))
        reconstruction = np.stack([i[0] for i in data])
        labels = np.stack([i[1] for i in data])
        weights = np.stack([i[2] for i in data])
        return reconstruction, labels, weights
    
    def unpad_seq(self, p_seqs, ns_2_ls):
        return [pseq[:l,:] for pseq, l in zip(p_seqs, get_seq_lens(ns_2_ls))]
        
    def test_lle(self, train_prog_bar, test_prog_bar, kneighbors):
        """
        To run tests on templates, templates_dicts is created
        templates_dicts is a dictionary of dicts.
        
        t = ["T3x", "T3y", "T2x", "T2y", "T1x", "T1y", "Jx", "Jy", "ULx", "ULy", "LLx", "LLy"]
        or 
        t = ['LA',  'LP',  'JA',  'TDCD', 'TDCL', 'TBCD', 'TBCL', 'TTCD', 'TTCL']
        
        Vowel test:
        remove = [{'ai_ow', 'r'},
                  {'@@r', 'ai_ow', 'r'}, 
                  {'@@r', 'uh', 'ai_ow', 'r'}, 
                  {'@@r', 'uh', 'o', 'oo', 'ai_ow', 'r', 'e'}]
        Glides test:
        remove = [{'@@r', 'ai_ow', 'r'},
                  {'@@r', 'ai_ow'}, 
                  {'@@r', 'r'}]
        """
        p = os.path.join(os.getcwd(),'data/Rplot_ema/experiment1/vowel')
        #### hand selected templates from mngu0 data ####
        templates = {'velar':        np.array([5.33,  0.59, 3.79,  0.13, 2.16, -0.38, 0.9,  -2.77, -0.99, -0.05, -0.73, -2.62])*10,
                     'postalveolar': np.array([5.4,  -0.48, 3.9,   0.2,  2.2,   0.2,  0.8,  -2.77, -0.98, -0.03, -0.72, -2.61])*10,
                     'alveolar':     np.array([5.25, -0.48, 3.76,  0.09, 1.95, -0.1,  0.76, -2.67, -0.97, -0.05, -0.71, -2.60])*10,
                     'dental':       np.array([5.3,  -0.48, 3.72,  0,    1.7,  -0.4,  0.85, -2.8,  -0.96, -0.04, -0.73, -2.59])*10,
                     'labiodental':  np.array([5.4,  -1,   -3.9,  -0.8,  2.1,  -1.2,  1.25, -2.7,  -0.5,  -0.0,  -0.2,  -2.6])*10,
                     'bilabial':     np.array([5.25, -0.5,  3.75, -0.25, 2.2,  -0.5,  0.6,  -2.6,  -1,    -0.3,  -0.8,  -2.05])*10,
                     'ii':           np.array([5.3,  -0.25,  3.72, 0.2,  2.11, -0.32, 0.83, -2.66, -1.00, -0.07, -0.80, -2.60])*10,
                     'e':            np.array([5.25, -0.6,  3.75,  0,    2.1,  -0.35, 0.71, -2.63, -0.97, -0.05, -0.71, -2.59])*10,
                     'a':            np.array([5.4,  -0.7,  3.85, -0.14, 2.2,  -0.4,  0.86, -2.75, -0.97, -0.05, -0.72, -2.60])*10,
                     '@@r':          np.array([5.25, -0.51, 3.8,  -0.1,  2.16, -0.35, 0.85, -2.69, -1.0,  -0.07, -0.75, -2.62])*10,
                     'uu':           np.array([5.25,  0,    3.83,  0.4,  2.16, -0.25, 0.85, -2.7,  -1.07, -0.07, -0.8,  -2.5])*10,
                     'oo':           np.array([5.28, -0.35, 3.75, -0.14, 2.22, -0.40, 0.85, -2.71, -1.07, -0.07, -0.75, -2.4])*10,
                     'uh':           np.array([5.35, -0.57, 3.84,  0.01, 2.22, -0.39, 0.72, -2.69, -0.99, -0.03, -0.78, -2.45])*10,
                     'o':            np.array([5.35, -0.4,  3.85, -0.23, 2.20, -0.39, 0.76, -2.63, -1.04, -0.05, -0.80, -2.46])*10,
                     'aa':           np.array([5.42, -0.53, 3.78, -0.23, 2.27, -0.46, 0.85, -2.63, -0.99, -0.05, -0.79, -2.55])*10,
                     'ai_ow':        np.array([5.3,  -0.6,  3.9,  -0.22, 2.22, -0.42, 0.85, -2.75, -0.95, -0.05, -0.65, -2.65])*10,
                     'r':            np.array([5.42, -0.53, 3.82, -0.2,  2.27, -0.46, 0.73, -2.6,  -1.07, -0.15, -0.85, -2.3])*10
                     }

        from scipy.spatial import ConvexHull
        train_data = defaultdict(dict)
        for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, ns_2_ls in train_prog_bar:
            name = {re.sub('_\d+\w?', '', spkr[0]) for spkr in ns_2_ls}.pop()
            train_data[name].update({n[0]:bseq for n,bseq in zip(ns_2_ls, self.unpad_seq(pad_tgt_seqm, ns_2_ls))})
        templates_dicts = {}
        if self.args.templates_type == 'hand_selected': # (cm *10 -> mm)
#            remove = [{'@@r', 'uh', 'ai_ow', 'r'}]
            remove = [{'@@r', 'uh', 'ai_ow', 'postalveolar', 'o', 'e', 'dental'}]
            count = 0
            # normal ema:
            # python model1r.py --test-lle --ema-dir data/ema --scale --partition-dir datafcc --batch-size 10 --shuffle --templates-type hand_selected --mm --tjl
            # tvs:
            # python model1r.py --test-lle --ema-dir data/ema_tv --scale --partition-dir data/ data/mfcc --batch-size 10 --shuffle --templates-type hand_selected --tv
            # random:
            # python model1r.py --test-lle --ema-dir data/ema_tv --scale --partitio data/mfcc --batch-size 10 --shuffle --templates-type random --tv
            if self.args.tv: 
                templates = tvs
            for d, keys in zip([templates]*len(remove), remove):
                count += 1
                new_templates = {}
                for k,v in d.items():
                    if k not in keys: new_templates.update({k:v})
                templates_dicts.update({'set{}'.format(count):new_templates})
        elif self.args.templates_type == 'pca': 
            pdb.set_trace()
            train_data = [v for v in {k:v for k,v in train_data['mngu0'].items() if 'mngu0' in k}.values()]
            templates_dicts = self.get_pc_templates(train_data, [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        elif self.args.templates_type == 'random':

            t_templates = range(3,5)
            mngu0_data = torch.cat([v for v in train_data['mngu0'].values()]).numpy()
            templates_lists = [np.random.choice(mngu0_data.shape[0], 12 + 2**i, replace = False) for i in t_templates]
            count = 0
            for indexes in templates_lists:
                count += 1
                new_templates = {}
                template_count = 0
                for t in mngu0_data[indexes,:]:
                    template_count += 1
                    new_templates.update({'t{}'.format(template_count):t})
                templates_dicts.update({'set{}'.format(count):new_templates})
                
        test_data = defaultdict(dict)
        for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, ns_2_ls in test_prog_bar:
            name = {re.sub('_\d+\w?', '', spkr[0]) for spkr in ns_2_ls}.pop()
            test_data['seq'].update({n[0]:bseq  for n,bseq in zip(ns_2_ls, self.unpad_seq(pad_tgt_seq, ns_2_ls))})
            test_data['seqm'].update({n[0]:bseq for n,bseq in zip(ns_2_ls, self.unpad_seq(pad_tgt_seqm, ns_2_ls))})
            test_data['names'].update({n[0]:name for n in ns_2_ls})
        
        #### LLE TEST BLOCK ####
        for coef, templates_dict in templates_dicts.items():
            print(coef)
            for k in range(1,13):
                try:
                    #for i in range(1,4):
                    self.seq_lens = []
                    reconstructions = []
                    for name, sample_mat in test_data['seqm'].items():
                        reconstruction, labels, weights  = self.lle_reconstruction(sample_mat.numpy(), templates_dict, k)
                        reconstructions.append(FloatTensor(reconstruction))
                        self.get_lens_names_frames([(name, sample_mat.shape[0])])
                        self.current_name = self.get_spkr_names().pop()
                        self.seq_lens = [reconstruction.shape[0] for reconstruction in reconstructions]
                        self.get_cor('single', FloatTensor(reconstruction), test_data['seqm'][name])
                        self.get_losses(reconstruction, sample_mat)
                        self.update_info(coef, "test-lle")
                    self.compute_batch_averages("test-lle", len(templates_dict), k)
                except Exception as e: print(e)

                
                #print(reconstruction, sample_mat)
                #print('jaw & lip & tongue', coef, kneighbors, mean(rmse_list_jlt), mean(cor_list_jlt))
    def get_pc_templates(self, speaker_tensors, coefs):
        """
        in: speaker tensors (list of samples)
        Experiment 2a: 
            Using principal component analysis, we templates that diverge
            from means and variances of pc articulatory configurations.
            e.g. we select the most extreme configurations of points.
        """
        pdb.set_trace()
        #model.set_prog_bar_desc(test_prog_bar, "loading speaker(s) for pca")
        # TODO:
        #unpadded_seqs  = self.unpad_seq(pad_tgt_seq)
        #unpadded_seqsm = self.unpad_seq(pad_tgt_seqm)
        # TODO:
        import sklearn
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        all_spkr_s = torch.cat([i[:,:12] for i in speaker_tensors], 0)
        if np.where(np.isnan(all_spkr_s))[0].size!=0: #suppress nans
            all_spkr_s = np.delete(all_spkr_s, np.where(np.isnan(all_spkr_s.numpy()))[0],0)
        pca = PCA(n_components=all_spkr_s.shape[1])
        pca.fit(all_spkr_s)
        real = False
        if real:
            knn = NearestNeighbors(1)
            knn.fit(X = all_spkr_s)
        i = 0
        coefs_templates_dict = {}
        for coef in tqdm(coefs):
            templates = {}
            for length, vector in zip(pca.explained_variance_, pca.components_):
                pc_vec = pca.mean_ + (np.sqrt(length) * coef) * vector
                print(pc_vec)
                if real:
                    k_ni = knn.kneighbors(np.expand_dims(pc_vec,axis=0), return_distance = False)
                    templates.update({'pc_{}'.format(i+1):all_spkr_s[k_ni].numpy()})
                else: templates.update({'pc_{}'.format(i+1):pc_vec})
                i += 1
            coefs_templates_dict.update({'z_{}'.format(coef):templates})
        return coefs_templates_dict

def save_model(model, model_path):
    try: torch.save(model.state_dict(), model_path)
    except: 
        filename = 'untitled_model'
        print('no name entered for model calling model {}'.format(filename))
        torch.save(model.state_dict(), os.path.join(model_path,filename))

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

def get_device(cuda):
    if torch.cuda.is_available() and cuda: return torch.device('cuda')
    elif cuda:
        print('No cuda devices found, defaulting to cpu')
        return torch.device('cpu')
    else: return torch.device('cpu')

def validate(model, data, epoch, count):
    val_prog_bar = data.get_prog_bar(data.zip_validation_batches(), 'validation', epoch)
    torch.no_grad(), model.eval()
    for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, n2f in val_prog_bar:
        count += 1
        model.get_lens_names_frames(n2f)
        model.forward(Variable(pad_src_seq)) # pass thru model
        model.get_cor('batch', model.out_seq, pad_tgt_seq) # get correlation
        out_seq_mm = data.unnormalize_output(model.current_name, model.out_seq, False) #MSE(mm)
        model.get_losses(out_seq_mm, pad_tgt_seqm, pad_tgt_seq)
        model.update_info(epoch, 'validation')
        model.set_prog_bar_desc(val_prog_bar, count)

def train(model, data, epoch, count):
    train_prog_bar = data.get_prog_bar(data.zip_training_batches(), 'training', epoch)
    model.train()
    for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, n2f in train_prog_bar:
        count += 1
        model.get_lens_names_frames(n2f)
        model.forward(Variable(pad_src_seq)) # pass thru model
        model.get_cor('batch',model.out_seq, pad_tgt_seq) # get correlation
        out_seq_mm = data.unnormalize_output(model.current_name, model.out_seq, False) #MSE(mm)
        model.get_losses(out_seq_mm, pad_tgt_seqm, pad_tgt_seq)
        model.propagate_loss()
        model.optimizer.zero_grad()
        model.update_info(epoch, 'training')
        model.set_prog_bar_desc(train_prog_bar, count)


def train_val_loop(model, data, epoch):
    bad_epochs = 0
    best_valid_loss = float('inf')
    print('Model Overview:\n\n{}'.format(model))
    while bad_epochs < args.patience:
        count = 0
        epoch += 1
        train(model, data, epoch, count)
        validate(model, data, epoch, count)
        model.compute_batch_averages('training')
        bv_l = model.compute_batch_averages('validation')
        if bv_l < best_valid_loss: # Decide whether to terminate training while loop
            best_valid_loss, bad_epochs = bv_l, 0
            #if epoch % args.save_internal == 0: save_checkpoint(model, model.history, args.checkpoint_dir)
        else: bad_epochs += 1
    print('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
    print('Saving Model...')
    try: save_model(model, args.save_model)
    except: save_checkpoint(model, model.history, args.checkpoint_dir)


def test(model, data, count, epoch):
    ## TODO: SWITCH TO TESTING BATCH.
    test_prog_bar  = data.get_prog_bar(data.zip_testing_batches(), 'testing', epoch)
    torch.no_grad(), model.eval()
    if args.test_lle:
        train_prog_bar = data.get_prog_bar(data.zip_training_batches(), 'training', epoch)
        #model.set_prog_bar_desc(test_prog_bar, "loading speakers for lle")
        model.test_lle(train_prog_bar, test_prog_bar, range(1,12))
        print('finished testing lle.\nExiting')
        sys.exit()
    for pad_src_seq, pad_tgt_seq, pad_tgt_seqm, names_to_lengths in test_prog_bar:
        count += 1
        model.get_lens_names_frames(names_to_lengths)
        model.forward(Variable(pad_src_seq)) # pass thru model
        model.get_cor('batch',model.out_seq, pad_tgt_seq) # get correlation
        out_seq_mm = data.unnormalize_output(model.current_name, model.out_seq, False) #MSE(mm)
        model.get_losses(out_seq_mm, pad_tgt_seqm, pad_tgt_seq)
        model.set_prog_bar_desc(test_prog_bar, count)

def run():
    """
    Run Training and Validation over a number of epochs
    """
    device = get_device(args.cuda)
    data  = LoadInputOutput(args, device)
    model = BuildModel(args, device, data.input_dim, data.output_dims)
    data.get_batches(args.batch_size, args.test_lle)
    if args.cuda: model = model.cuda()
    # we either want to load a model or initialize history.
    if args.load_dir: load_model(model, args.load_dir, device)
    elif os.path.isfile(args.checkpoint_dir): load_checkpoint(model, args, device) 
    else: model.init_history()
    epoch = 0 if model.is_new else model.get_epoch()
    if args.train: train_val_loop(model, data, epoch)
    if args.test or args.test_lle: test(model, data, 0, epoch)
    print('Terminating program.')

def main(args):
    run()

if __name__ == '__main__':
    args = get_args()
    main(args)
