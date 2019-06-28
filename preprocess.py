#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 19:18:57 2019
Aim: Preprocess files to give .wav -> MFCC, fbank features
@author: s1888641
"""

import re, argparse, pdb, os
from tqdm import tqdm
import numpy as np
#import python_speech_features
#import simpleaudio
import scipy.io.wavfile as wav
from collections import defaultdict #, OrderedDict

def get_args():
    """
    Args define program usage.
    """
    info = 'Script preprocesses data and saves, given i/o directories.'
    parser = argparse.ArgumentParser(info)
    parser.add_argument('--lab-dir', default=None, help='path to label directory')
    parser.add_argument('--wav-dir', default=None, help='path to wav directory')
    parser.add_argument('--ema-dir', default=None, help='path to ema directory')

    parser.add_argument('--feats', default=None, help='features to extract from waveform')
    
    #parser.add_argument('--file-encoding', default='utf-8', help='encoding to read in text file corpus')
    return parser.parse_args()

class Preprocess():
    '''Class that loads, stores etc an EST_Track format file.'''
    def __init__(self, args):
        #self.lab_dict = self.load_labs(args.lab_dir) if args.lab_dir else None
        #self.feats_dict = self.load_feats(args.wav_dir, args.feats) if args.wav_dir else None
        self.ema_dict = self.load_ema(args.ema_dir) if args.ema_dir else None
        
#        self.T = None # time channel
#        self.D = None # the channel data
#        self.N = None # channel names
#        self._data = None # raw data read from the file 
#        self.name = None

    def load_labs(self, path):
        '''
        Read in labels
        (and utterances?)
        '''
        file_count, lab_dict = int(0), defaultdict(tuple)
        directory = os.getcwd()+'/'+path
        time, phone =  '([\d\.]+)', '([\w\d!@#]+)'
        line_to_match = '^\t {0} \d+ \t{1}$'.format(time, phone)
        for file in tqdm(os.listdir(path), desc = 'loading labels'):
            #print(file)
            if not file.endswith(".lab"): continue
            with open(os.path.join(directory, file)) as f:
                labs = []
                file_count+=1
                for line in f:
                    m = re.findall(line_to_match, line)
                    if m: labs.append(m[0])
                lab_dict[file] = tuple(labs)
        # REPORT DETAILS
        number_of_phones = len([p for v in lab_dict.values() for p in v])
        report_1 = '{0:,} (phone, time) tuples recovered from {1:,} lab files'
        report_2 = 'average phones per lab file {0:.1f}'
        print(report_1.format(number_of_phones, file_count))
        print('lab_dict is a default_dict.')
        print('structure ',set((type(k), type(v)) for k,v in lab_dict.items()))
        print(report_2.format(number_of_phones/file_count))
        return lab_dict

    def load_feats(self, path, feats):
        '''
        Load wav data from a specified directory
        '''
        file_count, feat_dict = int(0), defaultdict(tuple)
        directory = os.getcwd() + '/' + path
        if feats=='mfcc': from python_speech_features import mfcc, delta
        elif feats=='fbank': from python_speech_features import logfbank #, fbank
        if feats==None: return None
        for file in tqdm(os.listdir(path), desc = 'loading wavs and creating {} features'.format(feats)):
            if not file.endswith(".wav"): continue
            features = []
            file_count+=1
            (rate,sig) = wav.read(directory + '/' + file)
            if feats=='mfcc':
                mfcc_feat_0 = mfcc(sig, rate, numcep = 13, appendEnergy = True)
                features = np.concatenate((mfcc_feat_0, delta(mfcc_feat_0,1), delta(mfcc_feat_0, 2)), axis=1)
                if file_count==1: print('Creating mfccs 12 cepstrum + e for 0th, 1st and 2nd order delta features ({}d)'.format(features.shape[1]))
            elif feats=='fbank':
                fbank_feat = logfbank(sig,rate)
                features = fbank_feat 
                if file_count==1: print('Creating logfbank features ({}d)'.format(features.shape[1]))
            #pdb.set_trace()
            feat_dict[file] = features
        #filename_to_frames = {k:v.shape[0] for k,v in feat_dict.items()}
        report_1 = '{0:,} {1} vectors extracted from {2:,} wav files'
        report_2 = 'average vectors per wav file {0:.1f}'
        total_vectors = sum({k:v.shape[0] for k,v in feat_dict.items()}.values())
        print(report_1.format(total_vectors, feats, file_count))
        print(report_2.format(total_vectors / file_count))
        print('feat_dict is a defaultdict.')
        print('structure ',set((type(k), type(v)) for k,v in feat_dict.items()))
        return feat_dict

    def load_ema(self, path):
        """
        Load ema data from directory
        """
        file_count, ema_dict = int(0), defaultdict(tuple)
        flag = bool
        directory = os.getcwd() + '/' + path
        for file in tqdm(os.listdir(path), desc = 'loading ema data'):
            if not file.endswith(".ema"): continue
            
            with open(os.path.join(directory, file), 'rb') as f:
                ema = []
                file_count += 1
                for line in f:
                    print(line)
                    if re.findall(b'EST_Header_End', line): flag = True 
                    #re.findall(line_to_match, line)
                    if flag:
                        #len(tuple(int(h, 16) for h in " ".join(["{:02x}".format(x) for x in line]).split()))
                        
                        ema_ints = [int(h, 16) for h in " ".join(["{:02x}".format(x) for x in line]).split()]
                        ema.append(ema_ints)
                pdb.set_trace()
                flag=False
                print(len(ema)/500)
                ema_dict[file] = tuple(ema)
        
def main(args):
    #parser.add_argument( '-a', '--ascii', dest='dump_ascii', default=False, action='store_true', help='dump the file to stdout as ascii' )
    #parser.add_argument( '-i', '--info', dest='info', default=False, action='store_true', help='dump info about the file to stdout as ascii' )
    preprocess = Preprocess(args)
    ### TODO: OUTPUT PICKLED DEFAULTDICT ###

if __name__ == '__main__':
    args = get_args()
    main(args)
