import random
import time

import argparse, gc

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy



from ..gnn.gcn.layers import InputLayer
from ..align.test import greedy_alignment, sim
from ..align.sample import generate_neighbours

class DecentralizedAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, dim, rate=0.5):
        super(DecentralizedAttentionLayer, self).__init__()
        
        self.w = tf.keras.layers.Dense(dim)
        self.w1 = tf.keras.layers.Dense(dim)
        self.w2 = tf.keras.layers.Dense(dim)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.key_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    

        
    def call(self, query, adj_mat, key=None, training=True, mean_sum=False):
        query = self.layernorm1(query)
        
        if key is None:
            key = query
        else:
            key = self.key_layernorm(key)
        
        value = self.w(key)
        
        # transpose without axis
        if mean_sum:
            col, row = tf.ones((query.shape[0], 1))*adj_mat, tf.ones((1, query.shape[0]))*adj_mat
        else:
            
        
            at1, at2 = self.w1(query), self.w2(key)
            
            #watch query
            sum1, sum2 = tf.reduce_sum(at1, axis=1, keepdims=True), tf.reduce_sum(at2, axis=1, keepdims=True)

            sum1, sum2 = tf.keras.activations.tanh(sum1), tf.keras.activations.tanh(sum2)

            sum1, sum2 = self.dropout1(sum1, training=training), self.dropout2(sum2, training=training)
            
            
            col, row = sum1*adj_mat, tf.transpose(sum2)*adj_mat
        
        sum_ = tf.sparse.add(col, row)
        logits = tf.SparseTensor(indices=sum_.indices,
                                  values=tf.nn.leaky_relu(sum_.values),
                                  dense_shape=sum_.dense_shape)
        attn_weights = tf.sparse.softmax(logits)
        out = tf.sparse.sparse_dense_matmul(attn_weights, value)
        
        return self.layernorm2(out)
    
    
class LinearAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, dim, rate=0.5):
        super(LinearAttentionLayer, self).__init__()
        
        self.w = tf.keras.layers.Dense(dim)
        self.w1 = tf.keras.layers.Dense(dim)
        self.w2 = tf.keras.layers.Dense(dim)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, query, input_, adj_mat, training=True):
        input_ = self.layernorm1(input_)
        
        dense = self.w(input_)
        
        at1, at2 = self.w1(query), self.w2(input_)
        sum1, sum2 = tf.reduce_sum(at1*query, axis=1, keepdims=True), tf.reduce_sum(at2*input_, axis=1, keepdims=True)
        
        sum1, sum2 = tf.keras.activations.tanh(sum1), tf.keras.activations.tanh(sum2)
        
        sum1, sum2 = self.dropout1(sum1, training=training), self.dropout2(sum2, training=training)
        
        # transpose without axis
#         col, row = tf.ones_like(sum1)*adj_mat, tf.transpose(tf.ones_like(sum2))*adj_mat
        col, row = sum1*adj_mat, tf.transpose(sum2)*adj_mat
        
        sum_ = tf.sparse.add(col, row)
        logits = tf.SparseTensor(indices=sum_.indices,
                                  values=tf.nn.leaky_relu(sum_.values),
                                  dense_shape=sum_.dense_shape)
        attn_weights = tf.sparse.softmax(logits)
        out = tf.sparse.sparse_dense_matmul(attn_weights, dense)
        
        return self.layernorm2(out)

class DecentralizedConv(tf.keras.layers.Layer):
    def __init__(self, dim, rate=0.5, rel_attn=True, operator='+'):
        super(DecentralizedConv, self).__init__()
        self.rel_attn = rel_attn
        
        self.local_attn = DecentralizedAttentionLayer(dim=dim, rate=rate)
        self.er_attn = LinearAttentionLayer(dim=dim, rate=rate)
        self.operator=operator

        self.dense1 = tf.keras.layers.Dense(dim,)
        self.dense2 = tf.keras.layers.Dense(dim,)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    
    def call(self, ent_em, rel_em, adj_mats, key=None, training=True, mean_sum=False):
        
        
        e_attn_out = self.local_attn(ent_em, adj_mats[0], key=key, training=training, mean_sum=mean_sum)
        
        if self.rel_attn:
            er_attn_out = self.er_attn(tf.stop_gradient(e_attn_out), rel_em, adj_mats[1], training)
            if self.operator=='+':
                out = e_attn_out + 0.6* self.dense1(er_attn_out)
            elif self.operator=='*':
                out = e_attn_out * self.dense1(er_attn_out)
        else:
            out = e_attn_out

        return out
        




class DecentRL(tf.keras.layers.Layer):
    def __init__(self, dim, depth=2, rate=.5, rel_attn=True, layernorm=True, operator='+'):
        super(DecentRL, self).__init__()
        
        self.depth= depth
        
        self.decentconvs = [DecentralizedConv(dim=dim, rel_attn=rel_attn, rate=rate, operator=operator) for i in range(self.depth)]
        
        self.layernorm = layernorm
        if layernorm:
            self.layernorms = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for i in range(self.depth)]



    def call(self, ent_em, rel_em, adj_mats, training=True):
        outs = []
        
        outs.append(ent_em)
        for i in range(self.depth):
            if i == 0:
                out = self.decentconvs[i](ent_em, rel_em, adj_mats[i%2], key=None, training=training, mean_sum=True)
                
            else:
                out = self.decentconvs[i](outs[-1], rel_em, adj_mats[i%2], key=outs[-2], training=training, mean_sum=False)
                # for layer 2+, the reidual is: initalized decentralized embedding g_0 +  prev output g_{k-1} + current output g_k
                out = outs[1]+outs[-1]+out

            if self.layernorm:
                out = self.layernorms[i](out)

            outs.append(out)
        
        # remove the orginal embeddings
        return outs[1:]



class KGEA(tf.keras.layers.Layer):


    '''
    We adapt our decentRL to the source code of AliNet (https://github.com/nju-websoft/AliNet). 
    We did not leverage its neighbor selection and truncated loss.  
    '''


    def __init__(self, adjs, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, tri_num, ent_num, rel_num, args):
        super(KGEA, self).__init__()

        self.one_hop_layers = None
        self.two_hop_layers = None
        self.layers_outputs = None
        
        self.args = args

        self.adj_mats = adjs
        self.kg1 = kg1
        self.kg2 = kg2
        self.sup_ent1 = sup_ent1
        self.sup_ent2 = sup_ent2
        self.ref_ent1 = ref_ent1
        self.ref_ent2 = ref_ent2
        self.tri_num = tri_num
        self.ent_num = ent_num
        self.rel_num = rel_num

        self.neg_multi = args.neg_multi
        self.neg_margin = args.neg_margin
        self.neg_param = args.neg_param

        self.layer_dims = args.layer_dims
        self.layer_num = len(args.layer_dims) - 1
        self.activation = args.activation
        self.dropout_rate = args.dropout_rate

        self.eval_metric = args.eval_metric
        self.hits_k = args.hits_k
        self.eval_threads_num = args.eval_threads_num
        self.eval_normalize = args.eval_normalize
        self.eval_csls = args.eval_csls

        self.new_edges1, self.new_edges2 = set(), set()
        self.new_links = set()
        self.pos_link_batch = None
        self.neg_link_batch = None
        self.sup_links_set = set()
        for i in range(len(sup_ent1)):
            self.sup_links_set.add((self.sup_ent1[i], self.sup_ent2[i]))
        self.new_sup_links_set = set()
        self.linked_ents = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)

        sup_ent1 = np.array(self.sup_ent1).reshape((len(self.sup_ent1), 1))
        sup_ent2 = np.array(self.sup_ent2).reshape((len(self.sup_ent1), 1))
        weight = np.ones((len(self.sup_ent1), 1), dtype=np.float)
        self.sup_links = np.hstack((sup_ent1, sup_ent2, weight))
        
        self.input_embeds, self.output_embeds_list = InputLayer(
            shape=[self.ent_num, self.layer_dims[0]]).init_embeds, None
        self.rel_embeds = InputLayer(
            shape=[self.rel_num, self.layer_dims[0]]).init_embeds
        
        self.optimizer = tf.keras.optimizers.Adam()

        
        # initialize decentRL
        all_dim = sum(self.layer_dims)
        reg_dim = self.layer_dims[0]
        self.reg_dense = tf.keras.layers.Dense(reg_dim,input_shape=(all_dim,), activation='relu')
        self.reg_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.reg_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        b_init = tf.zeros_initializer()
        self.reg_biases = tf.Variable(initial_value=b_init(shape=(self.ent_num,),
                                              dtype='float32'), name='reg_b',
                         trainable=True)

        # customize your model here
        self.model = DecentRL(dim=self.layer_dims[0],
                            depth=self.layer_num,
                            rel_attn=args.rel_attn,
                            layernorm=args.layernorm,
                            operator=args.operator
                            )

        

    def compute_loss(self, pos_links, neg_links, only_pos=False):
        index1 = pos_links[:, 0]
        index2 = pos_links[:, 1]
        neg_index1 = neg_links[:, 0]
        neg_index2 = neg_links[:, 1]
        embeds_list = list()
        for output_embeds in self.output_embeds_list + [self.input_embeds]:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds_list.append(output_embeds)
        output_embeds = tf.concat(embeds_list, axis=1)
        output_embeds = tf.nn.l2_normalize(output_embeds, 1)

        embeds1 = tf.nn.embedding_lookup(
            output_embeds, tf.cast(index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(
            output_embeds, tf.cast(index2, tf.int32))
        pos_loss = tf.reduce_sum(tf.reduce_sum(
            tf.square(embeds1 - embeds2), 1))
        
        loss = pos_loss
        
        if self.args.decentRL:
            reg_loss = self.decent_loss(embeds1, index1) + self.decent_loss(embeds2, index2)
            loss += reg_loss
            
        embeds1 = tf.nn.embedding_lookup(
            output_embeds, tf.cast(neg_index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(
            output_embeds, tf.cast(neg_index2, tf.int32))
        neg_distance = tf.reduce_sum(tf.square(embeds1 - embeds2), 1)
        neg_loss = tf.reduce_sum(tf.keras.activations.relu(
            self.neg_margin - neg_distance))
        
        loss += self.neg_param * neg_loss
        
        return loss


    def eval_embeds(self, adj_mat=None):
        if adj_mat is None:
            adj_mat = self.adj_mats

        output_embeds_list = self.model(
                        self.input_embeds, self.rel_embeds, adj_mat, training=False)

        return self.input_embeds, output_embeds_list

    def valid(self):
        embeds_list1, embeds_list2 = list(), list()
        input_embeds, output_embeds_list = self.eval_embeds()
        for output_embeds in output_embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds1 = tf.nn.embedding_lookup(output_embeds, self.ref_ent1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, self.ref_ent2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds1 = embeds1.numpy()
            embeds2 = embeds2.numpy()
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        embeds1 = np.concatenate(embeds_list1, axis=1)
        embeds2 = np.concatenate(embeds_list2, axis=1)
        alignment_rest, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                                   self.eval_metric, False, 0, False, self.logger)
        del embeds1, embeds2
        gc.collect()
        
        return hits1_12

    
    def get_sim_mat(self, adj_mat=None, ref_ent1=None, ref_ent2=None, is_open=False):
        if adj_mat is None:
            adj_mat = self.adj_mats
            ref_ent1 = self.ref_ent1
            ref_ent2 = self.ref_ent2
        
        embeds_list1, embeds_list2 = list(), list()
        input_embeds, output_embeds_list = self.eval_embeds(adj_mat)
        
        
        if is_open:
            embeds_list = output_embeds_list
        else:
            embeds_list = [input_embeds] + output_embeds_list
            
        for output_embeds in embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds1 = tf.nn.embedding_lookup(output_embeds, ref_ent1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, ref_ent2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds1 = embeds1.numpy()
            embeds2 = embeds2.numpy()
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)

        embeds1 = np.concatenate(embeds_list1, axis=1)
        embeds2 = np.concatenate(embeds_list2, axis=1)
        
        return sim(embeds1, embeds2, metric=self.eval_metric, normalize=False, csls_k=self.eval_csls)
        
        
    
    def test(self, adj_mat=None, ref_ent1=None, ref_ent2=None, is_open=False):
        if adj_mat is None:
            adj_mat = self.adj_mats
            ref_ent1 = self.ref_ent1
            ref_ent2 = self.ref_ent2
        
        embeds_list1, embeds_list2 = list(), list()
        input_embeds, output_embeds_list = self.eval_embeds(adj_mat)
        
        
        if is_open:
            embeds_list = output_embeds_list
        else:
            embeds_list = [input_embeds] + output_embeds_list
            
        for output_embeds in embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds1 = tf.nn.embedding_lookup(output_embeds, ref_ent1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, ref_ent2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds1 = embeds1.numpy()
            embeds2 = embeds2.numpy()
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        if self.args.eval_on_each_layer:
            for i in range(len(embeds_list1)):
                print("test embeddings of layer {}:".format(i))
                embeds1 = embeds_list1[i]
                embeds2 = embeds_list2[i]
                greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                 self.eval_metric, False, 0, True, self.logger)
                greedy_alignment(embeds1, embeds2,
                                 self.hits_k, self.eval_threads_num,
                                 self.eval_metric, False, self.eval_csls, True, self.logger)
        print("test all embeddings:")
        embeds1 = np.concatenate(embeds_list1, axis=1)
        embeds2 = np.concatenate(embeds_list2, axis=1)
        alignment_rest, _, _, _ = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                   self.eval_metric, False, 0, True, self.logger)
        hits, _, _, MRR = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                   self.eval_metric, False, self.eval_csls, True, self.logger)
        
        del embeds1, embeds2
        gc.collect()
        
        return (hits[0], hits[1], MRR)
    

    def generate_input_batch(self, batch_size, neighbors1=None, neighbors2=None):
        if batch_size > len(self.sup_ent1):
            batch_size = len(self.sup_ent1)
        index = np.random.choice(len(self.sup_ent1), batch_size)
        pos_links = self.sup_links[index, ]
        neg_links = list()
        if neighbors1 is None:
            neg_ent1 = list()
            neg_ent2 = list()
            for i in range(self.neg_multi):
                neg_ent1.extend(random.sample(
                    self.sup_ent1 + self.ref_ent1, batch_size))
                neg_ent2.extend(random.sample(
                    self.sup_ent2 + self.ref_ent2, batch_size))
            neg_links.extend([(neg_ent1[i], neg_ent2[i])
                              for i in range(len(neg_ent1))])
        else:
            for i in range(batch_size):
                e1 = pos_links[i, 0]
                candidates = random.sample(neighbors1.get(e1), self.neg_multi)
                neg_links.extend([(e1, candidate) for candidate in candidates])
                e2 = pos_links[i, 1]
                candidates = random.sample(neighbors2.get(e2), self.neg_multi)
                neg_links.extend([(candidate, e2) for candidate in candidates])
        neg_links = set(neg_links) - self.sup_links_set
        neg_links = neg_links - self.new_sup_links_set
        neg_links = np.array(list(neg_links))
        return pos_links, neg_links


    def decent_loss(self, inputs, labels):
        inputs = self.reg_layernorm(self.reg_dropout(self.reg_dense(inputs)))
        
        def l2_loss(inputs, labels):
            original_embeds = tf.nn.embedding_lookup(self.input_embeds, tf.cast(labels, tf.int32))
            return tf.math.reduce_sum(tf.keras.losses.cosine_similarity(original_embeds, inputs))
        def infoNCE_loss(inputs, labels):
            labels = tf.reshape(labels, [-1,1])
            return tf.math.reduce_sum(tf.nn.sampled_softmax_loss(
                self.input_embeds, self.reg_biases, labels, inputs, 64, self.ent_num, num_true=1,
            ))
        def auto_distiller_loss(inputs, labels):
            labels = tf.reshape(labels, [-1,1])
            return tf.math.reduce_sum(tf.nn.sampled_softmax_loss(
                tf.stop_gradient(self.input_embeds), self.reg_biases, labels, inputs, 64, self.ent_num, num_true=1,
            ))
        def none(inputs, labels):
            return 0.
        
        losses = {'L2':l2_loss,'InfoNCE':infoNCE_loss,'decentRL':auto_distiller_loss,'None':none}

        return losses[self.args.loss_type](inputs, labels)

    def train(self, batch_size, max_epochs=1000, start_valid=300, eval_freq=10):
        flag1 = 0
        flag2 = 0
        is_stop=False
        steps = len(self.sup_ent2) // batch_size

        

        neighbors1, neighbors2 = None, None
        
        
        if steps == 0:
            steps = 1
        for epoch in range(1, max_epochs + 1):
            start = time.time()
            epoch_loss = 0.0
            for step in range(steps):
                self.pos_link_batch, self.neg_link_batch = self.generate_input_batch(batch_size,
                                                                                     neighbors1=neighbors1,
                                                                                     neighbors2=neighbors2)
                with tf.GradientTape() as tape:
                    self.output_embeds_list = self.model(
                        self.input_embeds, self.rel_embeds, self.adj_mats, training=True)
                    batch_loss = self.compute_loss(
                        self.pos_link_batch, self.neg_link_batch) / batch_size
                    grads = tape.gradient(batch_loss, self.trainable_variables)
                    self.optimizer.apply_gradients(
                        zip(grads, self.trainable_variables))
                    epoch_loss += batch_loss
            self.logger.info('epoch {}, loss: {:.4f}, cost time: {:.4f}s'.format(
                epoch, epoch_loss, time.time() - start))
            if epoch % eval_freq == 0:
                flag = self.valid()
                if flag-flag1<0.3:
                    is_stop=True
                else:
                    flag1 = flag
                    is_stop=False
                    
                if is_stop and epoch >= start_valid:
                    print("\n == early stop == \n")
                    break



class KGEP(tf.keras.layers.Layer):
    def __init__(self, args):
        super(KGEP, self).__init__()

        self.args = args

        # for compatibility
        self._options = args

    def init_variables(self):
        hidden_size = self.args.hidden_size
        self.ent_embeds = self.add_weight('ent_embeds',
                                          shape=(self.ent_num, hidden_size),
                                          initializer='glorot_uniform',
                                          trainable=True,)

        
        
        self.decent_rel_embeds = self.add_weight('decent_rel_embeds',
                                          shape=(self.rel_num, hidden_size),
                                          initializer='glorot_uniform',
                                          trainable=True,)

        

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.args.learning_rate)

        
        
        # customize your model
        self.decentRL = DecentRL(
            dim=hidden_size, depth=self.args.num_layers, rel_attn=self.args.rel_attn,
            rate=self.args.decentRL_dp_rate, layernorm=False, operator='*')
        
        
        # init auto-distiller and the decoder
        in_dim = hidden_size*2
        out_dim = hidden_size
        self.dropout_rate = self.args.decoder_dp_rate
        self.reg_dense = tf.keras.layers.Dense(
            out_dim, input_shape=(in_dim,), activation='relu')
        self.reg_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.reg_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.reg_biases = self.add_weight('auto-distiller bias', shape=(self.ent_num,),
                                     initializer='zeros',
                                     trainable=True)
        
        self.out_w = self.add_weight('decoder out_w', shape=(self.ent_num, in_dim),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.out_b = self.add_weight('decoder out_b',shape=(self.ent_num,),
                                     initializer='zeros',
                                     trainable=True)
        
        self.in_reldp = tf.keras.layers.Dropout(self.dropout_rate)
        self.in_entdp = tf.keras.layers.Dropout(self.dropout_rate)
        
        # self.out_bn = tf.keras.layers.BatchNormalization()
        self.out_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        
        self.rel_embeds = self.add_weight('rel_embeds',
                                          shape=(self.rel_num, in_dim),
                                          initializer='glorot_uniform',
                                          trainable=True,)
        

    def decentralized_embeddings(self, adj_mats, training=True):
        ent_embeds = self.ent_embeds
        rel_embeds = self.decent_rel_embeds
        
        decent_embeds = self.decentRL(ent_embeds, rel_embeds, adj_mats, training=training)
        all_embeds = [ent_embeds, decent_embeds[-1]]#[decent_embeds[0], decent_embeds[-1]]
        return tf.concat(all_embeds, axis=1)

        
    def decent_loss(self, inputs, labels):
        inputs = self.reg_dropout(tf.linalg.normalize(self.reg_dense(inputs), 2, axis=1)[0])
        ori_embeds = tf.linalg.normalize(self.ent_embeds, 2, axis=1)[0] 
        
        def l2_loss(inputs, ori_embeds, labels):
            original_embeds = tf.nn.embedding_lookup(ori_embeds, tf.cast(labels, tf.int32))
            return tf.math.reduce_sum(tf.keras.losses.cosine_similarity(original_embeds, inputs))
        def infoNCE_loss(inputs, ori_embeds, labels):
            labels = tf.reshape(labels, [-1,1])
            return tf.math.reduce_mean(tf.nn.sampled_softmax_loss(
                ori_embeds, self.reg_biases, labels, inputs, 64, self.ent_num, num_true=1,
            ))
        
        def auto_distiller_loss(inputs, ori_embeds, labels):
            labels = tf.reshape(labels, [-1,1])
            return tf.math.reduce_mean(tf.nn.sampled_softmax_loss(
                tf.stop_gradient(ori_embeds), self.reg_biases, labels, inputs, 64, self.ent_num, num_true=1,
            ))

        def none(inputs, ori_embeds, labels):
            return 0.

        losses = {'L2':l2_loss,'InfoNCE':infoNCE_loss,'decentRL':auto_distiller_loss,'None':none}

        return losses[self.args.loss_type](inputs, ori_embeds, labels)

    def lookup_entembeds(self, x, adj_mats, training=True):
        ent_embeds = self.decentralized_embeddings(adj_mats=adj_mats, training=training)
 
        ent_embeds += 1e-10
        ent_embeds = tf.linalg.normalize(ent_embeds, 2, axis=1)[0]
        
        
        return tf.nn.embedding_lookup(params=ent_embeds, ids=x)

    def lookup_relembeds(self, x):
        rel_embeds = tf.linalg.normalize(self.rel_embeds, 2, axis=1)[0]
        # rel_embeds = self.rel_embeds
        return tf.nn.embedding_lookup(
            params=rel_embeds, ids=x)

    def predict(self, h, r, training=True):
        h = self.in_entdp(h, training=training)
        r = self.in_reldp(r, training=training)
        
        if self.args.decoder == 'ComplEx':
            r = tf.reshape(r, (-1, 2, self.args.hidden_size))
            rr, ir = r[:, 0], r[:, 1]
            
            h = tf.reshape(h, (-1, 2, self.args.hidden_size))
            rh, ih = h[:, 0], h[:, 1]
            
            rscore = rh*rr - ih*ir
            iscore = rh*ir + ih*rr
            score = tf.concat([rscore, iscore], axis=-1)
        elif self.args.decoder == 'TransE':
            score = h+r
        elif self.args.decoder == 'DistMult':
            score = h*r
        
        return self.out_dropout(tf.linalg.normalize(score, 2, axis=1)[0], training=training)
    
    def score(self, pred):
        logits = tf.matmul(pred, tf.transpose(self.out_w))
        logits = tf.nn.bias_add(logits, self.out_b)
        return logits

    def decoder_loss(self, labels, out):
        loss = tf.nn.sampled_softmax_loss(
            weights=self.out_w,
            biases=self.out_b,
            labels=tf.cast(labels[:, tf.newaxis],tf.int64),
            inputs=out,
            num_sampled=self.args.num_sampled,
            num_classes=self.ent_num,
            remove_accidental_hits=True,
        )

        return loss

    def call(self, h, r, t, adj_mats):
        he = self.lookup_entembeds(h, adj_mats=adj_mats)
        re = self.lookup_relembeds(r)

        out = self.predict(he, re)

        loss = self.decoder_loss(t, out)
        
        loss += 1e-3 * self.decent_loss(he, h)

        return tf.reduce_mean(loss)

    def evaluate(self, h, r, t, adj_mats):
        he = self.lookup_entembeds(h, adj_mats=adj_mats, training=False)

        re = self.lookup_relembeds(r)

        out = self.predict(he, re, training=False)

        prob = tf.nn.softmax(self.score(out))

        return prob