import tensorflow as tf
import pandas as pd
import numpy as np
import gc, os, logging
from datetime import datetime

from .data_loaders import *
from ..align.util import no_weighted_adj, gcn_load_data
from .gcnload4ep import gcn_load_data as ep_load_adjmat

def reset_graph(model):
    tf.keras.backend.clear_session()
    del model.model
    tf.compat.v1.reset_default_graph()
    gc.collect()

def cal_ranks(probs, method, label):
    if method == 'min':
        probs = probs - probs[range(len(label)), label].reshape(len(probs), 1)
        ranks = (probs > 0).sum(axis=1) + 1
    else:
        ranks = pd.DataFrame(probs).rank(axis=1, ascending=False, method=method)
        ranks = ranks.values[range(len(label)), label]
    return ranks

def cal_performance(ranks, tops=[1, 3, 10]):
    hits = [sum(ranks <= top) * 1.0 / len(ranks) for top in tops]
    # h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    mr = sum(ranks) * 1.0 / len(ranks)
    return hits, mrr, mr

def sample_mini_batches(data, range_max, batch_size=2048, permutation=True):
    if data.shape[0] % batch_size == 0:
        batch_num = data.shape[0] // batch_size
    else:
        batch_num = data.shape[0] // batch_size + 1
    
    if permutation:
        indices = np.random.choice(data.shape[0], data.shape[0], replace=False)
    else:
        indices = np.arange(data.shape[0])
        
    
    for i in range(batch_num):
        one_batch_indices = indices[i*batch_size: (i+1)*batch_size]
        pos_batch_data = data[one_batch_indices]
        
        yield pos_batch_data



class KGWrapper:
    def __init__(self, args):
        self.args = args
        self.save_name = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        self.init_logger()

    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        

        LOG_FORMAT = '%(asctime)s -- %(levelname)s#: %(message)s '
        DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a'
        LOG_PATH = 'logs/%s.log' % self.save_name


        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        fh = logging.FileHandler(LOG_PATH)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)



    def convert_to_sparse_mat(self, adj, radj):
        radj_mats = [[tf.cast(tf.sparse.SparseTensor(indices=a[0], values=a[1], dense_shape=a[2]), tf.float32)
                  for a in aa]
                  for aa in radj]

        adj_mats = [tf.cast(tf.sparse.SparseTensor(
                    indices=a[0], values=a[1], dense_shape=a[2]), tf.float32) for a in adj]

        adj_mats = [[a, *b] for a, b in zip(adj_mats, radj_mats)]

        return adj_mats

    def read_data(self):
        pass

    def fit(self, Model):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def save(self):
        ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
        ckpt_path = 'ckpts/' + self.save_name
        
        ckpt.save(file_prefix=ckpt_path)
    
    def restore(self, save_name=None):
        if save_name is None:
            save_name= self.save_name
        ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
        ckpt_path = 'ckpts/' + save_name + '-1'
        
        ckpt.restore(ckpt_path)#.assert_consumed()


class EAWrapper(KGWrapper):
    def __init__(self, args):
        self.args = args
        super(EAWrapper, self).__init__(args)
        
    def read_data(self):
        args = self.args
        
        if args.openEA:
            complete_adj, _, _, _, _, test_ref_ent1, test_ref_ent2, _, ent_num, rel_num, _, complete_radj= \
                gcn_load_data(args.input, is_two=args.two_hop, is_decentRL=args.decentRL, return_radj=True)    

            adj, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, tri_num, ent_num, rel_num, rel_id_mapping, radj= \
                gcn_load_data(args.input.replace('data','opendata'), is_two=args.two_hop, e_num=ent_num, r_num=rel_num, is_decentRL=args.decentRL, return_radj=True)
            
            adj_mats = self.convert_to_sparse_mat(adj, radj)
            complete_adj_mats = self.convert_to_sparse_mat(complete_adj, complete_radj)
            
            
            test_open_ref_mask = ~np.in1d(test_ref_ent1, ref_ent1)
            self.test_open_ref_ent1 = list(np.array(test_ref_ent1)[test_open_ref_mask])
            self.test_open_ref_ent2 = list(np.array(test_ref_ent2)[test_open_ref_mask])
            
            
            self.test_adj_mats, self.test_ref_ent1, self.test_ref_ent2 = complete_adj_mats, test_ref_ent1, test_ref_ent2
        else:
            adj, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, tri_num, ent_num, rel_num, rel_id_mapping, radj = \
                gcn_load_data(args.input, is_two=args.two_hop, is_decentRL=args.decentRL, return_radj=True)
            
            adj_mats = self.convert_to_sparse_mat(adj, radj)
            self.test_adj_mats, self.test_ref_ent1, self.test_ref_ent2 = adj_mats, ref_ent1, ref_ent2
            

            
            
        
        
        self.model_params = [adj_mats, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2,
                                                 tri_num, ent_num, rel_num, args]
        return
        

    def fit(self, Model, args=None):
        if args is not None:
            # update args
            self.args = args
            self.model_params[-1] = args

        self.logger.info(vars(self.args))
        self.model = Model(*self.model_params)
        self.model.logger = self.logger
    
                
    def train(self):
        args = self.args
        model = self.model
        
        model.train(args.batch_size, max_epochs=args.max_epoch, start_valid=args.start_valid, eval_freq=args.eval_freq)
        
    def test(self):
        args = self.args
        model = self.model
        
        # results = (hits@1, hits@10, MRR)
        results = model.test(self.test_adj_mats, self.test_ref_ent1, self.test_ref_ent2)    
        return results
        
class EPWrapper(KGWrapper):
    def __init__(self, args):
        self.args = args
        super(EPWrapper, self).__init__(args)

    def read_data(self):
        args = self.args

        data_path = args.data_path
        if '/WN18/' in data_path:
            Reader = WordNetReader
        else:
            Reader = FreeBaseReader

        def load_from_dataset(data_path, args):
            reader = Reader()
            reader._options = args
            reader.args = args
            reader.read(data_path)

            test_data = reader._test_data[['h_id','r_id','t_id']].values
            train_data = reader._train_data[['h_id','r_id','t_id']].values
            valid_data = reader._valid_data[['h_id','r_id','t_id']].values

            adj, radj = ep_load_adjmat(list(train_data), reader._ent_num, reader._rel_num, is_decentRL=True, return_radj=True, is_two=False)
            adj_mats = self.convert_to_sparse_mat(adj, radj)
            return reader, test_data, train_data, valid_data, adj_mats
            
        if args.openEP:
            full_reader, _, _, _, full_adj_mats = load_from_dataset(data_path, args)
            # read the training data without unseen entities
            args.data_path = data_path.replace('data','opendata')
            reader, test_data, train_data, valid_data, adj_mats = load_from_dataset(data_path, args)
            # validate and test on the full adjacency matrix
            reader._tail_test_filter_mat = full_reader._tail_test_filter_mat
            self.test_adj_mats = full_adj_mats
        else:
            reader, test_data, train_data, valid_data, adj_mats = load_from_dataset(data_path, args)
            self.test_adj_mats = adj_mats


        self.reader = reader
        self.test_data = test_data
        self.train_data = train_data
        self.valid_data = valid_data
        self.adj_mats = adj_mats


    def fit(self, Model, args=None):
        if args is None:
            args = self.args

        self.logger.info(vars(self.args))
        reader = self.reader
        model = Model(args)

        model.ent_num, model.rel_num = reader._ent_num, reader._rel_num
        # model.adj_mats = self.adj_mats

        model.init_variables()

        self.model = model


    def test(self, data, filter_mat, model, args, adj_mats=None, method='min'):
        target = data[:, -1]
        
        if adj_mats is None:
            adj_mats = self.adj_mats

        probs = []
        for i, batch in enumerate(sample_mini_batches(data, self.reader._ent_num, permutation=False)):
            h, r, t = [col for col in batch.T]
            probs.append(model.evaluate(h, r, t, adj_mats=adj_mats)) 
        
        probs = np.concatenate(probs)
        assert len(probs) == len(data)
        
        # copy the probs of target
        target_probs = probs[range(len(data)), target].copy()
        
        # mask true candidates
        probs[filter_mat[0], filter_mat[1]] = -1.
        
        # recover the target probs
        probs[range(len(data)), target] = target_probs
        
        
        
        filter_ranks = cal_ranks(probs, method=method, label=target)
        (h_1, h_3, h_10), mrr, mr = cal_performance(filter_ranks)
        
        return (h_1, h_3, h_10, mrr, mr)

    def train(self):
        data, model, args = self.train_data, self.model, self.args

        
        results = []
        FLAG, best_epoch, best_MRR = 50, 0, 0.
        for epoch in range(1, args.max_epoch):
            for j, batch in enumerate(sample_mini_batches(data, self.reader._ent_num, batch_size=args.batch_size)):
                h, r, t = [col for col in batch.T]
                
                with tf.GradientTape() as tape:
                    batch_loss = model(h, r, t, self.adj_mats)
                    grads = tape.gradient(batch_loss, model.trainable_variables)
                    model.optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                            
                print('Epoch:%i, Batch:%i, Loss:%.3f' % (epoch, j, batch_loss), end='\r')

            if epoch % 1 == 0:
                result = self.test(self.valid_data, self.reader._tail_valid_filter_mat, model, args, adj_mats=self.adj_mats)
                self.logger.info('\n%s, Epoch:%i, Hits@1:%.3f, Hits@3:%.3f, Hits@10:%.3f, MRR:%.3f, MR:%i\n' % ('VALID', epoch, *result))

                # early stop
                if best_MRR < result[-2]:
                    self.save()
                    best_epoch = epoch
                    best_MRR = result[-2]
                    FLAG = 50
                else:
                    FLAG -= 1

                
            if FLAG <= 0:
                self.logger.info('EARLY STOP')
                break
        
        # restore the best model
        self.logger.info('Restore the best model')
        self.restore()
        # evaluate on testing data
        result = self.test(self.test_data, self.reader._tail_test_filter_mat, model, args, method='average', adj_mats=self.test_adj_mats)
        self.logger.info('\n%s, Epoch:%i, Hits@1:%.3f, Hits@3:%.3f, Hits@10:%.3f, MRR:%.3f, MR:%i\n' % ('TEST', best_epoch, *result))
        
                

    
