import random
import time
from datetime import datetime
import argparse, gc

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy


from code.decentRL.layers import KGEA
from code.utils.wrappers import EAWrapper
from code.utils.wrappers import reset_graph




if __name__ == '__main__':
    #param settings

    #param settings

    # overall setting
    parser = argparse.ArgumentParser(description='Entity Alignment Settings')
    parser.add_argument('--input', type=str, default="./data/DBP15K/fr_en/mtranse/0_3/")
    parser.add_argument('--output', type=str, default='./output/results/')
    parser.add_argument('--two_hop', type=bool, default=True)
    parser.add_argument('--openEA', type=bool, default=False, help='training on the open dataset')
    parser.add_argument('--layer_dims', type=list, default=[512,]*5)  
    parser.add_argument('--batch_size', type=int, default=4500)
    parser.add_argument('--max_epoch', type=int, default=80)
    parser.add_argument('--start_valid', type=int, default=50)
    parser.add_argument('--eval_metric', type=str, default='inner')
    parser.add_argument('--hits_k', type=list, default=[1, 10])
    parser.add_argument('--eval_threads_num', type=int, default=10)
    parser.add_argument('--eval_normalize', type=bool, default=True)
    parser.add_argument('--eval_csls', type=int, default=2)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--adj_number', type=int, default=1)
    parser.add_argument('--sim_th', type=float, default=0.5)
    parser.add_argument('--eval_on_each_layer', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)

    # constrastive loss setting
    parser.add_argument('--neg_multi', type=int, default=10)  # for negative sampling
    parser.add_argument('--neg_margin', type=float, default=1.5)  # margin value for negative loss
    parser.add_argument('--neg_param', type=float, default=0.1)  # weight for negative loss




    # decentRL setting
    parser.add_argument('--decentRL', type=bool, default=True, help='decentralized or centralized')
    parser.add_argument('--loss_type', type=str, default='decentRL', choices=['decentRL', 'InfoNCE', 'L2', 'None'])
    parser.add_argument('--rel_attn', type=bool, default=True)
    parser.add_argument('--layernorm', type=bool, default=True)
    parser.add_argument('--operator', type=str, default='+', choices=['+', '*'])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='drop rate of decentRL')

    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU') 
    selected = physical_devices[args.gpu]
    tf.config.experimental.set_memory_growth(selected, True)
    tf.config.experimental.set_visible_devices(selected, 'GPU')


    # initialize a wrapper and process the dataset
    wrapper = EAWrapper(args=args)
    wrapper.read_data()
    
    # fit a model
    wrapper.fit(Model=KGEA, args=args)

    wrapper.train()
    
    wrapper.test()
    # wrapper.save()