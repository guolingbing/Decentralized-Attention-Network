import os, random, datetime, pickle, time, argparse
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from code.decentRL.layers import KGEP
from code.utils.wrappers import EPWrapper
from code.utils.wrappers import reset_graph




if __name__ == '__main__':
    # overall
    parser = argparse.ArgumentParser(description='Entity Prediction Settings')
    parser.add_argument('--reindex', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default='data/FB15k-237/')
    parser.add_argument('--openEP', type=bool, default=False, help='training on the open dataset')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--hidden_size', type=int, default=512) 
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3*3)
    parser.add_argument('--gpu', type=int, default=0)

    # customize parameters for decentRL
    parser.add_argument('--decentRL', type=bool, default=True, help='decentralized or centralized')
    parser.add_argument('--decentRL_dp_rate', type=float, default=0.3, help='drop rate of decentRL')
    parser.add_argument('--layernorm', type=bool, default=False)
    parser.add_argument('--operator', type=str, default='*', choices=['+', '*'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--rel_attn', type=bool, default=True) 
    parser.add_argument('--loss_type', type=str, default='decentRL', choices=['decentRL', 'InfoNCE', 'L2', 'None'])

    # customize parameters for the decoder
    parser.add_argument('--decoder_dp_rate', type=float, default=0.5, help='drop rate of the decoder')                                                                  
    parser.add_argument('--decoder', type=str, default='DistMult', choices=['TransE', 'DistMult', 'ComplEx'])
    parser.add_argument('--num_sampled', type=int, default=2048*4, help='number of sampled negative examples in a batch')

    args = parser.parse_args()

    # gpu settings
    physical_devices = tf.config.list_physical_devices('GPU') 
    selected = physical_devices[args.gpu]
    tf.config.experimental.set_memory_growth(selected, True)
    tf.config.experimental.set_visible_devices(selected, 'GPU')



    
    # initialize a wrapper and process the dataset
    wrapper = EPWrapper(args=args)
    wrapper.read_data()

    # fit a model
    wrapper.fit(Model=KGEP, args=args)

    wrapper.train()
    wrapper.test(wrapper.test_data, wrapper.reader._tail_test_filter_mat, wrapper.model, wrapper.args, wrapper.test_adj_mats)