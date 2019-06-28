#!/usr/bin/env python
#
##################################################################
##    Author: Korin Richmond                                    ##
##   Purpose: Deal with EST Track files in python               ##
##            (this is isn't really complete or robust code,    ##
##             but it does the job in simple cases)             ##
##                                                              ##
##   Created: Tue 13 September 2011                             ##
##                                                              ##
##  Copyright (C) Korin Richmond 2011                           ##
##  All rights reserved.                                        ##
##################################################################
##                                                              ##
## Licence conditions apply to the download and use of this     ##
## software - please see the file "LICENCE" accompanying this   ##
## file in the same directory for details                       ##
##                                                              ##
## DISCLAIMER:                                                  ##
##                                                              ##
## EDINBURGH UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK       ##
## DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,        ##
## INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND      ##
## FITNESS. IN NO EVENT SHALL EDINBURGH UNIVERSITY OR THE       ##
## CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR          ##
## CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING    ##
## FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF   ##
## CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT   ##
## OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS      ## 
## SOFTWARE.                                                    ##
##                                                              ##
##################################################################

import re, sys, pdb, os
import numpy as np
import cPickle as pickle
from collections import defaultdict

class estfile:
    '''Class that loads, stores etc an EST_Track format file.'''

    def __init__( self ):
        self.T = None # time channel
        self.D = None # the channel data
        self.N = None # channel names
        self._data = None # raw data read from the file 
        self.name = None

    def parseheader( self, infile ):
        '''Read and parse the header from file object "infile"'''

        if infile.readline().strip() != 'EST_File Track':
            raise ValueError( 'not an EST Track file: %s ' % infile.name )

        # defaults
        dtype = 'binary';
        bo = 'l';
        nrows = 0;
        ncols = 0;
        N = {};
        
        line = infile.readline().strip()
        while line != 'EST_Header_End':
            if line == 'DataType ascii':
                dtype = 'ascii'
            elif line == 'DataType binary':
                dtype = 'binary'
            elif line == 'ByteOrder 01':
                bo = '<'
            elif line == 'ByteOrder 10':
                bo = '>'
            else:
                mo = re.search( 'name\s+(?P<name>\S+)|NumFrames\s+(?P<nf>\d+)|NumChannels\s+(?P<nc>\d+)|Channel_(?P<ch>\d+)\s+(?P<n>\S+)', line )
                if mo is None:
                    pass # ignore other fields
                elif mo.group('nf'):
                    nrows = int(mo.group('nf'))
                elif mo.group('nc'):
                    ncols = int(mo.group('nc'))
                elif mo.group('ch'):
                    N[int(mo.group('ch'))] = mo.group('n')
                elif mo.group('name'):
                    self.name = mo.group('name')
                else:
                    raise ValueError( 'Error parsing EST Track header: %s' % infile.name )
            line = infile.readline().strip()

        return (dtype, nrows, ncols, bo, N)

    def load( self, parser ):
        '''Load data from EST Track format file "filename" into
        this estfile instance.'''
        p = os.getcwd() + '/' + parser.in_directory
        self.channel_time_dict = defaultdict(dict)
        self.channel_lsfd_dict = defaultdict(dict)
        self.channel_name_dict = defaultdict(dict)
        print('loading data from {}*'.format(p))
        for filename in os.listdir(p):
            if not parser.file_suffix in filename: continue
            s = p + filename if p[-1]=='/' else p +'/'+filename
            filein = file( s , 'r' )
            (dtype, nrows, ncols, bo, N) = self.parseheader(filein)
            self.N = N
            if dtype == 'binary':
                format = np.dtype( bo + 'f' )
                self._data = np.fromfile( filein, format ).reshape(nrows, ncols+2)            
            else:
                self._data = np.genfromtxt( filein )
            self.T = self._data[:,0]
            self.D = self._data[:,2:ncols+2]
            #print 'rows x cols = ', self.D.shape
            #print 'name = ', self.name
            
            
            self.channel_time_dict.update({self.name:self.T})
            self.channel_lsfd_dict.update({self.name:self.D})
            self.channel_name_dict.update({self.name:self.N})
            filein.close()
        

    def save( self, out_directory ):
        '''Save data using numpy.'''
        
def main():
    from optparse import OptionParser

    usage = 'usage: estfile [options] file'
    parser = OptionParser( usage, version='estfile 1.0.0' )
    parser.add_option( '-a', '--ascii', dest='dump_ascii', default=False,
                       action='store_true', help='dump the file to stdout as ascii' )
    parser.add_option( '-o', '--out-directory', dest='out_directory',
                       help='path to output (save) directory')
    parser.add_option( '-i', '--in-directory', dest='in_directory',
                       help='path to input (lsf/ema) directory')
    parser.add_option( '-s', '--suffix', dest='file_suffix',
                       help='scan files in dir only with a particular suffix')
    
    (options, args) = parser.parse_args()

    o = options.out_directory
    pdb.set_trace()
    # Directory output names:
    ctd = "channel_time"
    cld = "channel_{}".format(options.file_suffix[1:])
    cnd = "channel_name"
    
    o = o if o[-1]=='/' else o + '/'
    
    if not options.in_directory and options.out_directory: parser.error( 'Must supply a directory to process' )
    
    trackfile = estfile()
    trackfile.load(options)
    
    print('saving channel time data')
    for k,v in trackfile.channel_time_dict.items():
        np.save(o+ctd+'/'+k, v, allow_pickle=True)
    print('saving channel {} data'.format(options.file_suffix))
    for k,v in trackfile.channel_lsfd_dict.items():
        np.save(o+cld+'/'+k, v, allow_pickle=True)
    print('saving channel name data (in single set)')

    cnd_pickle_out = open(o + cnd + '/channel_names',"wb")
    pickle.dump(trackfile.channel_name_dict.values()[0].items(), cnd_pickle_out)
    cnd_pickle_out.close()

if __name__ == '__main__':
    main()
