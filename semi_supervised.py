import sys

import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import logging
tf.get_logger().setLevel(logging.ERROR)
import argparse
import numpy as np
from scipy.signal import fftconvolve
from sklearn.decomposition import PCA
from collections import namedtuple,Counter
from reconstruction import ReconstructionViewer
from latent import LatentSpaceViewer

import pickle

from data_loader import load_data, custom_load_data
from os.path import join,split
from os import makedirs
from datetime import datetime
#tf.config.experimental_run_functions_eagerly(True)
#tf.debugging.enable_check_numerics()
ConvSpec = namedtuple("ConvSpec","channel_width channels_in channels_out")
TRAIN_DIR = "./training_runs"
DATASET_ROOT = "Datasets"


SUPERVISED = False
REFINE = True


def split_class_wise(labels, exp_per_class=5, print_log=True):
    counts = Counter(labels)
    # Todo explain idcs
    all_idcs = []
    try:
        for unq_val in counts:
            val_idcs = np.argwhere(labels == unq_val).reshape(-1)
            print("Choosing %d values for class label %d." %(exp_per_class,unq_val))
            if counts[unq_val] < exp_per_class:
                print("WARN: Class label %d has %d samples, but %d examples were requested. Operating with replacement" % (unq_val, counts[unq_val],exp_per_class))
                chosen_idcs = np.random.choice(val_idcs,size=exp_per_class,replace=True)
            else:
                chosen_idcs = np.random.choice(val_idcs,size=exp_per_class,replace=False)
            all_idcs.append(chosen_idcs)
    except ValueError:
        print(ValueError)
    return np.concatenate(all_idcs)


def make_training_episode(labels, query_pct = 0.5):
    unq_set = np.unique(labels)
    all_support_idcs = []
    all_query_idcs = []
    for i,unq_val in enumerate(unq_set):
        idcs_to_choose = np.argwhere(labels == unq_val).reshape(-1)
        num_query_pts = int(idcs_to_choose.shape[0]*query_pct)
        # Get number of values for this
        query_idcs = np.random.choice(idcs_to_choose,num_query_pts,replace=False)
        support_idcs = np.setdiff1d(idcs_to_choose,query_idcs)
        all_support_idcs.append(support_idcs)
        all_query_idcs.append(query_idcs)
    return (np.concatenate(all_support_idcs),np.concatenate(all_query_idcs)),(num_query_pts,idcs_to_choose.shape[0]-num_query_pts)


class SemiSupervised(tf.keras.Model):
    def __init__(self,data_size,supervised=SUPERVISED,att_refine=REFINE, batch_size=25, filt_width=None):
        super(SemiSupervised, self).__init__()
        self.data_size = data_size
        self.supervised = supervised
        self.att_refine = att_refine
        self.batch_size = batch_size
        self.train_filters = True
        self.initializer = tf.initializers.GlorotUniform(5)
        self.mse_loss = tf.losses.MeanSquaredError()
        self.class_loss = tf.losses.SparseCategoricalCrossentropy()
        self.proto_loss = tf.losses.CategoricalCrossentropy()
        self.is_build_call = True
        self.loss_lambda = tf.Variable(tf.initializers.RandomUniform(minval=0.1,maxval=0.9)((1,)),trainable=True,constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1,max_value=0.9))

        # Calculate C1 Size
        #filt_width = the filter with
        if filt_width is None:
            c1_width = int(self.data_size[1]/10)
        else:
            c1_width = filt_width

        self.c1_specs = ConvSpec(channel_width=c1_width, channels_in=self.data_size[-1], channels_out=5)
        self.c1_filter = tf.Variable(self.initializer(self.c1_specs),trainable=self.train_filters,name="ae_c1_filter_kernel")
        self.c1_bias = tf.Variable(self.initializer((self.c1_specs.channels_out,)),trainable=self.train_filters,name="ae_c1_filter_bias")

        self.c1_de_filter = tf.Variable(self.initializer(self.c1_specs),trainable=self.train_filters,name="ae_c1_de_filter_kernel")
        self.c1_de_bias = tf.Variable(self.initializer((self.c1_specs.channels_out,)),trainable=self.train_filters,name="ae_c1_de_filter_bias")

        self.c1_stride = 1
        self.c1_pool_size = 2
        c1_out_size = (self.data_size[1])//self.c1_pool_size
        c2_width =  c1_width
        self.c2_specs = ConvSpec(channel_width=c2_width, channels_in=self.c1_specs.channels_out, channels_out=10)
        self.c2_filter = tf.Variable(self.initializer(self.c2_specs),trainable=self.train_filters,name="ae_c2_filter_kernel")
        self.c2_bias = tf.Variable(self.initializer((self.c2_specs.channels_out,)),trainable=self.train_filters,name="ae_c2_filter_bias")

        self.c2_de_filter = tf.Variable(self.initializer(self.c2_specs),trainable=self.train_filters,name="ae_c2_de_filter_kernel")
        self.c2_de_bias = tf.Variable(self.initializer((self.c2_specs.channels_out,)),trainable=self.train_filters,name="ae_c2_de_filter_bias")

        self.c2_stride = 1
        c2_out_size = int(c1_out_size)//2

        self.const_filt = tf.Variable(self.initializer(ConvSpec(channel_width=12,channels_in=7,channels_out=10)),trainable=self.train_filters,name="const_filter")
        self.const_bias = tf.Variable(self.initializer((10,)),trainable=self.train_filters,name="const_bias")
        self.conv_drop = tf.keras.layers.Dropout(0.0)
        self.drop = tf.keras.layers.Dropout(0.0)
        self.regularizer = tf.keras.regularizers.l2()

    def get_dense_layer(self,data):
        rsp_data = self.in_rsp(data)
        c1_raw = tf.nn.conv1d(rsp_data, self.c1_filter, self.c1_stride, "SAME") + self.c1_bias
        c1 = tf.nn.tanh(c1_raw)
        c1_pool = tf.nn.max_pool1d(c1,ksize=self.c1_pool_size,strides=self.c1_pool_size,padding="SAME")
        flat_c1 = self.f1(c1_pool)

        dense_layer = self.d2(self.d1(flat_c1))

        return dense_layer

    def get_feature_maps(self,data):
        rsp_data = self.in_rsp(data)
        c1_raw = tf.nn.conv1d(rsp_data, self.c1_filter, self.c1_stride, "VALID") + self.c1_bias
        c1 = tf.nn.tanh(c1_raw)

        c2_raw = tf.nn.conv1d(c1, self.c2_filter, self.c2_stride, "VALID") + self.c2_bias
        c2 = tf.nn.tanh(c2_raw)
        return c1,c2

    def get_ae_variables(self):
        return [var for var in self.trainable_variables if var.name.startswith("ae")]

    def get_sup_refine_variables(self):
        return [var for var in self.trainable_variables if var.name.startswith("dense")]

    @tf.function
    def max_pool_argmax_valid(self,data,ks=2):
        # Size of batch dimension
        batch_size = data.shape[0]
        # Closest multiple of ks
        trim_tsteps = (data.shape[1]//ks)*ks
        # Number of channels
        num_channel = data.shape[-1]
        # Trimmed data to closest multiple of ks.
        data_trim = data[:,:trim_tsteps]

        # Reshape to (batch_size, -1, ks, num_dims)
        data_rsp = tf.reshape(data_trim,(batch_size,-1,ks,num_channel))

        data_max = tf.reduce_max(data_rsp,axis=2)
        pool_tsteps = data_max.shape[1]
        data_argmax = tf.argmax(data_rsp,axis=2,output_type=tf.int32) + tf.reshape(tf.repeat(tf.range(trim_tsteps,delta=ks),num_channel),(pool_tsteps,num_channel))
        # Return indices into flattened array.
        data_argmax_flat = ((data_argmax * num_channel) + tf.range(num_channel))
        return data_max,data_argmax_flat

    @tf.function
    def unpool_random(self,data,out_size,ks=2,mode="once"):
        # Data will be upsampled to be ks*tsteps of the previous size
        pass
    def encoder(self, data, training=True):

        if self.is_build_call:
            self.f1 = tf.keras.layers.Flatten()
            self.in_rsp = tf.keras.layers.Reshape((data.shape[1], data.shape[-1]), input_shape=(data.shape[1],))
        self.rsp_data = self.in_rsp(data)

        ## CONV LAYER C1
        self.c1_raw = tf.nn.conv1d(self.rsp_data, self.c1_filter, self.c1_stride, "SAME") + self.c1_bias
        self.c1 = tf.nn.tanh(self.c1_raw)
        self.c1_drop = self.conv_drop(self.c1, training=training)
        #self.c1_pool,self.c1_pool_argmax = self.max_pool_argmax_valid(self.c1_drop)
        self.c1_pool = tf.nn.avg_pool1d(self.c1_drop,ksize=self.c1_pool_size,strides=self.c1_pool_size,padding="SAME")
        # self.flat_c1 = self.f1(self.c1_pool)



        ## CONV LAYER C2
        self.c2_raw = tf.nn.conv1d(self.c1_pool, self.c2_filter, self.c2_stride, "SAME") + self.c2_bias
        self.c2 = tf.nn.tanh(self.c2_raw)
        self.c2_drop = self.conv_drop(self.c2, training=training)
        self.c2_pool = tf.nn.avg_pool1d(self.c2_drop,ksize=4,strides=4,padding="SAME")
        self.flat_c2 = self.f1(self.c2_pool)

        if self.is_build_call:
            print("C1 Raw Size: %s" % self.c1_raw.shape)
            print("C1 Pool Size: (K=%d,S=%d) %s" % (4,4,str(self.c1_pool.shape)))
            print("C2 Raw Size: %s" % str(self.c2_raw.shape))
            print("C2 Pool Size: (K=%d,S=%d) %s" % (2,2,str(self.c2_pool.shape)))

        if self.is_build_call:

            self.d1 = tf.keras.layers.Dense(128,activation=tf.nn.tanh,name="l1_dense")
            self.d2 = tf.keras.layers.Dense(32,activation=tf.nn.tanh,name="l2_dense")
            self.smax_layer = tf.keras.layers.Dense(5,activation=tf.nn.softmax,name="smax_assignment")
            self.d3 = tf.keras.layers.Dense(128,activation=tf.nn.tanh,name="l3_dense")

            self.d4 = tf.keras.layers.Dense(self.flat_c2.shape[-1], activation=tf.nn.tanh, name="l4_dense")
            self.rsp = tf.keras.layers.Reshape(self.c2_pool.shape[1:])


        self.latent = self.drop(self.d2(self.d1(self.flat_c2)))

        return self.latent

    def get_smax_dense(self, encoder_out):
        return self.smax_layer(encoder_out)

    def full_ae_path(self, encoder_out,training=True):

        self.unflat = self.rsp(self.d4(self.d3(encoder_out))) + self.const_bias

        ### DECONV C2
        self.c2_unpool = tf.squeeze(tf.image.resize(tf.expand_dims(self.unflat,axis=1),size=[1,self.c2_drop.shape[1]],method="nearest"),axis=1)
        self.c2_decode_raw = tf.nn.conv1d_transpose(self.c2_unpool, self.c2_de_filter,
                                                    (encoder_out.shape[0], self.c1_pool.shape[1], self.c1_pool.shape[2]),
                                                    self.c2_stride, padding="SAME")
        self.c2_decode = tf.nn.tanh(self.c2_decode_raw) + self.c1_de_bias

        ### DECONV C1
        self.c1_unpool = tf.squeeze(tf.image.resize(tf.expand_dims(self.c2_decode,axis=1),size=[1,self.c1_drop.shape[1]],method="nearest"),axis=1)
        self.c1_decode_raw = tf.nn.conv1d_transpose(self.c1_unpool,self.c1_de_filter,
                                                    (encoder_out.shape[0],self.rsp_data.shape[1],self.rsp_data.shape[2]),
                                                    self.c1_stride,padding="SAME")
        self.c1_decode = tf.nn.tanh(self.c1_decode_raw)

        self.reconstruction = self.f1(self.c1_decode)

        self.is_build_call = False

        return self.reconstruction

class Trainer(object):
    def __init__(self):
        pass
    def reset_training(self,filt_width=None):

        self.train_data,self.train_labels,self.train_subj_ids = load_data(DATASET,DATASET_ROOT,normalize=True)
        #self.train_data, self.train_labels, self.train_subj_ids = custom_load_data(DATASET), [], []
        if hasattr(self,"model"):
            del self.model
        self.model = SemiSupervised(self.train_data.shape, batch_size=10, filt_width=filt_width)

        self.lt_viewer = LatentSpaceViewer(self.model,out_dir=self.full_path)
        self.rc_viewer = ReconstructionViewer(self.model,out_dir=self.full_path)

        preds = self.model.encoder(self.train_data[:10])
        self.model.full_ae_path(preds)
        tf.compat.v1.global_variables_initializer()

        self.ae_optimizer = tf.optimizers.Adam(0.001)
        self.proto_optimizer = tf.optimizers.Adam()
        self.att_optimizer = tf.optimizers.Adam(0.0001)

        self.train_ae_loss = tf.keras.metrics.Mean()
        self.test_ae_loss = tf.keras.metrics.Mean()

        self.train_proto_loss = tf.keras.metrics.Mean()
        self.test_proto_loss = tf.keras.metrics.Mean()

        self.train_silh_loss = tf.keras.metrics.Mean()

        self.train_db_loss = tf.keras.metrics.Mean()

        self.train_hub_loss = tf.keras.metrics.Mean()

        self.train_proto_acc = tf.keras.metrics.Accuracy()
        self.test_proto_acc = tf.keras.metrics.Accuracy()

        self.train_silh_loss = tf.keras.metrics.Mean()

        self.ae_vars = self.model.trainable_variables
        self.att_vars = self.model.trainable_variables

        self.accum_grad = None


    @tf.function
    def square_euc_dist(self,A, B = None,batched=False):
        l2_norm_A = tf.reduce_sum(tf.square(A), axis=-1, keepdims=True)
        if B is None:
            B = A
        l2_norm_B = tf.reduce_sum(tf.square(B),axis=-1,keepdims=True)

        if batched:
            return l2_norm_A - 2*tf.matmul(A,tf.transpose(B,(0,2,1))) + tf.transpose(l2_norm_B,(0,2,1))
        else:
            return l2_norm_A - 2*tf.matmul(A,tf.transpose(B)) + tf.transpose(l2_norm_B)

    # Todo
    @tf.function
    def ae_loss(self, data, addnoise=False, flatten=False):
        if addnoise:
            enc_input = tf.clip_by_value(
                data + tf.random.normal(shape=data.shape, stddev=0.5), -1.0,
                1.0)
        else:
            enc_input = data
        encoder_out = self.model.encoder(enc_input)
        decoder_out = self.model.full_ae_path(encoder_out)

        if flatten:
            enc_input = tf.reshape(enc_input,(enc_input.shape[0],-1))
        reconst_loss = tf.cast(self.model.mse_loss(decoder_out,enc_input),dtype=tf.float32)

        return encoder_out,reconst_loss

    # Todo
    @tf.function
    def silh_loss(self,unsup_embd,query_embd,supp_embd,num_classes,num_query,num_support):
        query_embd_rsp = tf.reshape(query_embd, (num_classes, num_query, query_embd.shape[-1]))
        supp_embd_rsp = tf.reshape(supp_embd, (num_classes, num_support, supp_embd.shape[-1]))

        all_sup_embds = tf.concat([query_embd_rsp, supp_embd_rsp], axis=1)
        all_sup_centrs = tf.reduce_mean(all_sup_embds, axis=1)

        euc_dist = self.square_euc_dist(all_sup_embds, batched=True)
        # Supervised Silhouette Loss
        mean_intra = tf.reshape(tf.reduce_sum(euc_dist, axis=1) / (num_support + num_query - 1), (-1,))
        bool_mask = tf.reshape(tf.tile(tf.eye(num_classes), [1, num_query + num_support]),
                               ((num_query + num_support) * num_classes, num_classes))
        all_centr_dist = tf.reduce_sum(
            (tf.expand_dims(tf.reshape(all_sup_embds, (-1, all_sup_embds.shape[-1])), axis=1) - all_sup_centrs) ** 2,
            axis=-1)
        max_inter_dist = tf.reduce_max(all_centr_dist, axis=-1, keepdims=True)
        masked = all_centr_dist + (bool_mask * max_inter_dist)
        second_closest_idcs = tf.argmin(masked, axis=-1)
        second_closest_embds = tf.gather(all_sup_embds, second_closest_idcs)
        mean_inter = tf.reduce_mean(
            tf.reduce_sum((tf.reshape(all_sup_embds, (-1, 1, all_sup_embds.shape[-1])) - second_closest_embds) ** 2,
                          axis=-1), axis=-1)

        # Unsupervised Silhouette Loss
        unsup_centr_dists = tf.reduce_sum((tf.expand_dims(unsup_embd, axis=1) - all_sup_centrs) ** 2, axis=-1)
        unsup_embd_dists = tf.reduce_sum(
            (tf.expand_dims(unsup_embd, axis=1) - tf.gather(all_sup_embds, tf.argmin(unsup_centr_dists, axis=-1))) ** 2,
            axis=-1)
        unsup_mean_intra = tf.reduce_sum(unsup_embd_dists, axis=-1) / (num_support + num_query - 1)

        # TRY
        unsup_second_closest = tf.gather(tf.eye(num_classes), tf.argmin(unsup_centr_dists, axis=-1))
        unsup_max_dist = tf.reduce_max(unsup_centr_dists, axis=-1, keepdims=True)
        unsup_max_dist_mask = unsup_centr_dists + (unsup_second_closest * unsup_max_dist)
        unsup_second_embd_dists = tf.reduce_sum(
            (tf.expand_dims(unsup_embd, axis=1) - tf.gather(all_sup_embds, tf.argmin(unsup_max_dist_mask, axis=-1))) ** 2,
            axis=-1)
        unsup_mean_inter = tf.reduce_mean(unsup_second_embd_dists, axis=-1)

        combined_mean_inter = tf.concat([mean_inter, unsup_mean_inter], axis=-1)
        combined_mean_intra = tf.concat([mean_intra, unsup_mean_intra], axis=-1)

        silh_loss = tf.reduce_mean((combined_mean_intra - combined_mean_inter) / tf.clip_by_value(
            tf.maximum(combined_mean_intra, combined_mean_inter), clip_value_min=tf.keras.backend.epsilon(),
            clip_value_max=tf.float32.max))
        return silh_loss

    #Todo
    @tf.function
    def db_loss(self,unsup_embd,query_embd,supp_embd,num_classes,num_query,num_support,soft_weight=False):
        query_embd_rsp = tf.reshape(query_embd, (num_classes, num_query, query_embd.shape[-1]))
        supp_embd_rsp = tf.reshape(supp_embd, (num_classes, num_support, supp_embd.shape[-1]))

        all_sup_embds = tf.concat([query_embd_rsp, supp_embd_rsp], axis=1)
        left_out_idcs = tf.reshape(
            tf.boolean_mask(tf.reshape(tf.tile(tf.range(num_classes), [num_classes, ]), [num_classes, num_classes]),
                            ~tf.eye(num_classes, dtype=tf.bool)), [num_classes, num_classes - 1])

        centrs = tf.reduce_mean(all_sup_embds, axis=1, keepdims=True)


        #Euclidean method here?
        clust_pts_to_centr = tf.reduce_sum((all_sup_embds - centrs) ** 2,
                                           axis=-1)

        clust_var_part = tf.reduce_sum(clust_pts_to_centr, axis=-1)

        #Todo what is unsup_embd
        unsup_pts_to_centr = tf.reduce_sum((unsup_embd - centrs)**2,axis=-1)

        if soft_weight:
            unsup_pts_weights = tf.expand_dims(tf.nn.softmax(-unsup_pts_to_centr,axis=0),axis=-1)
        else:
            unsup_pts_weights = tf.expand_dims(tf.cast(tf.tile(tf.expand_dims(tf.range(num_classes),axis=-1),[1,256]) == tf.argmin(unsup_pts_to_centr,axis=0,output_type=tf.int32),dtype=tf.float32),axis=-1)
        unsup_pts_weighted = tf.tile(tf.expand_dims(unsup_embd,axis=0),[num_classes,1,1]) * unsup_pts_weights

        unsup_var_part = tf.reduce_sum((centrs - unsup_pts_weighted)**2,axis=[-1,1])
        total_var = 1

        #print(tf.reduce_sum(unsup_pts_weights,axis=1))
        #Todo figure how to make tf.reduce_sum works
        try:
            #total_var = (clust_var_part + unsup_var_part)/(num_query+num_support + tf.reduce_sum(unsup_pts_weights,axis=0))
            total_var = (clust_var_part + unsup_var_part) / (num_query + num_support )
        except ValueError:
            raise ValueError

        centr_to_centr_dists = self.square_euc_dist(tf.squeeze(centrs))
        r = tf.range(num_classes)
        I, J = tf.meshgrid(r, r)
        cond = tf.where(I < J)
        # Only take the unique pairs from the matrix.
        centr_centr_k = tf.gather_nd(centr_to_centr_dists, cond)
        total_var_k = tf.gather_nd(tf.expand_dims(total_var,axis=0) + tf.expand_dims(total_var,axis=1),cond)
        db_loss = tf.reduce_mean(total_var_k / centr_centr_k) / num_classes
        return db_loss

    @tf.function
    def proto_loss(self,unsup_embd,query_embd,supp_embd,num_classes,num_query,num_support):
        query_embd_rsp = tf.reshape(query_embd, (query_embd.shape[0], 1, -1))
        supp_embd_class = tf.reshape(supp_embd, (num_classes, num_support, supp_embd.shape[-1]))
        #support_centrs = tf.reduce_mean(supp_embd_class, axis=1)
        support_centrs_sum = tf.reduce_sum(supp_embd_class,axis=1)
        support_centrs = support_centrs_sum/num_support
        # Distance between query point and support centroid
        query_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_classes), axis=-1), [1, num_query]), (-1, 1))
        dists = tf.reduce_sum((query_embd_rsp - support_centrs) ** 2, axis=-1)

        # Unsupervised pts weighting
        unsup_weights = tf.nn.softmax(-self.square_euc_dist(unsup_embd,support_centrs)) # (batch,num_classes)

        unsup_sum = tf.expand_dims(unsup_weights,axis=-1) * tf.expand_dims(unsup_embd,axis=1) # (batch, num_classes, embd_dim)

        update_centrs_numer = tf.reduce_sum(unsup_sum,axis=0) + support_centrs_sum
        update_centrs_denom = tf.expand_dims(num_support + tf.reduce_sum(unsup_weights,axis=0),axis=-1)

        update_centrs = update_centrs_numer/update_centrs_denom

        update_dists = tf.reduce_sum((query_embd_rsp - update_centrs) **2,axis=-1)
        # Combine losses
        lsmax = tf.nn.softmax(-update_dists, axis=-1)

        proto_loss = self.model.class_loss(query_indices, lsmax)
        return proto_loss

    @tf.function
    def train_model(self, batch_unsup, batch_query,batch_support, num_classes,num_support, num_query, flatten=False, ae=True,proto=False,silh=False,db=False,hubert=False,update_eoe=False,last_batch=False):
        with tf.GradientTape(persistent=True) as g:
            total_loss = 0

            all_data = tf.concat([batch_unsup,batch_query,batch_support],axis=0)
            # Embeddings are always required.
            embeddings,reconst_loss = self.ae_loss(all_data,flatten=flatten)
            #Todo unsup_embd origin
            unsup_embd,query_embd,support_embd = tf.split(embeddings,[batch_unsup.shape[0],batch_query.shape[0],batch_support.shape[0]])
            #an_array = unsup_embd.eval(session=tf.compat.v1.Session())


            if ae:
                total_loss += reconst_loss
                self.train_ae_loss(reconst_loss)
            if silh:
                silh_loss = self.silh_loss(unsup_embd,query_embd,support_embd,num_classes,num_query,num_support)
                total_loss += silh_loss
                self.train_silh_loss(silh_loss)
            if db:
                db_loss = self.db_loss(unsup_embd,query_embd,support_embd,num_classes,num_query,num_support)
                total_loss += db_loss
                self.train_db_loss(db_loss)
            if proto:
                proto_loss = self.proto_loss(unsup_embd,query_embd,support_embd,num_classes,num_query,num_support)
                total_loss += proto_loss
                self.train_proto_loss(proto_loss)
        if update_eoe and any([silh,db,proto]):
            # Always update ae_grads
            ae_grad = g.gradient(reconst_loss,self.ae_vars)
            # Accumulate SSE grad.
            if silh:
                sse_grad = g.gradient(silh_loss,self.ae_vars)
            elif db:
                sse_grad = g.gradient(db_loss,self.ae_vars)
            elif proto:
                sse_grad = g.gradient(proto_loss,self.ae_vars)
            if self.accum_grad is None:
                self.accum_grad = sse_grad
            else:
                self.accum_grad = [x+y if (x is not None and y is not None) else None for x, y in zip(self.accum_grad, sse_grad)]

            if last_batch:
                total_grad = [x+y if (x is not None and y is not None) else None for x,y in zip(self.accum_grad,ae_grad)]
                self.accum_grad = None
            else:
                total_grad = ae_grad

            self.proto_optimizer.apply_gradients(zip(total_grad,self.ae_vars))
        else:
            # Gradient of loss
            grads = g.gradient(total_loss,self.ae_vars)
            self.proto_optimizer.apply_gradients(zip(grads,self.ae_vars))


    @tf.function
    def test_batch(self, batch, flatten = False):
        encoder_out = self.model.encoder(batch)
        ae_batch_preds = self.model.full_ae_path(encoder_out, training=False)
        if flatten:
            batch = tf.reshape(batch, (batch.shape[0], -1))
        ae_loss_val = self.model.mse_loss(batch,ae_batch_preds)

        self.test_ae_loss(ae_loss_val)

    @tf.function
    def test_sup_batch(self, batch,labels):
        encoder_out = self.model.encoder(batch,training=False)
        sup_batch_preds = self.model.supervised_path(encoder_out,training=False)

        sup_loss_val = self.model.class_loss(labels,sup_batch_preds)

        self.test_sup_loss(sup_loss_val)
        return sup_batch_preds

    @tf.function
    def get_silh_preds(self,data,query_data,supp_data,num_classes,num_support,num_query):
        concat_data = tf.concat([data,query_data,supp_data],axis=0)

        encoder_out = self.model.encoder(concat_data)

        unsup = encoder_out[:data.shape[0]]
        sup_query = encoder_out[data.shape[0]:data.shape[0] + query_data.shape[0]]
        sup_supp = encoder_out[data.shape[0] + query_data.shape[0]:]
        sup_supp_rsp = tf.reshape(sup_supp, (num_classes, num_support, sup_query.shape[-1]))
        sup_query_rsp = tf.reshape(sup_query, (num_classes, num_query, sup_query.shape[-1]))

        all_sup_embds = tf.concat([sup_query_rsp, sup_supp_rsp], axis=1)
        all_sup_centrs = tf.reduce_mean(all_sup_embds, axis=1)
        # Unsupervised Silhouette Loss
        unsup_centr_dists = tf.reduce_sum((tf.expand_dims(unsup, axis=1) - all_sup_centrs) ** 2, axis=-1)
        # TRY THIS!
        unsup_centr_smax = tf.expand_dims(tf.nn.softmax(-unsup_centr_dists, axis=-1), axis=-1)

        return tf.squeeze(unsup_centr_smax)


    def make_path(self,name):

        return join(self.full_path,name)

    def setup_folders(self,dataset_name,exmpls_per_class):
        # Make training directory
        train_path = join(TRAIN_DIR,dataset_name)
        if not os.path.isdir(train_path):
            makedirs(train_path)
        out_dir_name = "%d_exmpls_" % exmpls_per_class
        if self.ae:
            out_dir_name += "ae_"
        if self.proto:
            out_dir_name += "proto_"
        if self.silh:
            out_dir_name += "silh_"
        if self.db:
            out_dir_name += "db_"

        out_dir_name  += datetime.now().strftime("%b_%d_%Y_%H_%M_%S_%p")
        self.full_path = join(train_path,out_dir_name)
        makedirs(self.full_path)
        makedirs(join(self.full_path,"figs"))
        makedirs(join(self.full_path,"reconst"))
        makedirs(join(self.full_path,"weights"))
        makedirs(join(self.full_path,"conf"))
        makedirs(join(self.full_path,"labels"))
        # Write seeds
        with open(self.make_path("seed.txt"),"w") as sf:
            sf.write(str(self.seed))

    def reset_losses(self):
        self.train_proto_loss.reset_states()
        self.train_proto_acc.reset_states()
        self.train_ae_loss.reset_states()
        self.train_silh_loss.reset_states()
        self.train_db_loss.reset_states()
        self.train_hub_loss.reset_states()

    def print_log_string(self,epoch,ae,proto,silh,db,hubert):
        log_str = "Epoch %d: " % epoch
        if ae:
            log_str += ("\n\tAE Loss: %f" % self.train_ae_loss.result())
        if proto:
            log_str += ("\n\tProto Loss: %f" % self.train_proto_loss.result())
        if silh:
            log_str += ("\n\tSilh Loss: %f" % self.train_silh_loss.result())
        if db:
            log_str += ("\n\tDB Loss: %f" % self.train_db_loss.result())
        if hubert:
            log_str += ("\n\tHubert Loss: %f" % self.train_hb_loss.result())

        with open(self.make_path("log.txt"), "a+") as log:
            log.write(log_str + "\n")

    def train_semi_sup(self, epochs=200,start_at=0,pct_for_query=0.5,exmpls_per_class=15,seed=None,proto=True,ae=True,silh=False,db=False,hubert=False,dataset_name=None,update_eoe=False,filt_width=None):

        self.proto = proto
        self.ae = ae
        self.silh = silh
        self.db = db
        self.seed = seed

        if not any([ae,proto,silh,db,hubert]):
            raise ValueError("Must train with at least one loss.")
        # Generate a seed if none provided
        if self.seed is None:
            self.seed = np.random.randint(low=0,high=999999)

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.setup_folders(dataset_name = dataset_name, exmpls_per_class=exmpls_per_class)
        ### RESET TRAINING HERE
        self.reset_training(filt_width=filt_width)

        train_loss = []
        train_ae_loss = []
        train_acc = []
        # Choose supervised data for semi-supervised loss
        sup_train_idcs = split_class_wise(self.train_labels,exp_per_class=exmpls_per_class,print_log=True)
        np.save(self.make_path("sup_train_idcs.npy"),sup_train_idcs)
        sup_train_data = self.train_data[sup_train_idcs] # Choose a list of row from train_data
        sup_train_labels =self.train_labels[sup_train_idcs] #Choose a list of row from train_label corresponding to the above train_data
        # Choose unsupervised data for unsupervised loss
        unsup_train_idcs = np.setdiff1d(np.arange(self.train_labels.shape[0]),sup_train_idcs) # return the list of row that in train_labels but not in sup_train_indcs
        #unsup_train_idcs = np.arange(self.train_labels.shape[0])
        np.save(self.make_path("unsup_train_idcs.npy"),unsup_train_idcs)

        if not any([silh,proto,hubert,db]):
            unsup_train_data = self.train_data
            unsup_train_labels = self.train_labels
            unsup_train_subj_ids = self.train_subj_ids
        else:
            unsup_train_data = self.train_data[unsup_train_idcs]
            unsup_train_labels = self.train_labels[unsup_train_idcs]
            if self.train_subj_ids is not None:
                unsup_train_subj_ids = self.train_subj_ids[unsup_train_idcs]
            else:
                unsup_train_subj_ids = None

        num_batches = max(1,unsup_train_data.shape[0] // self.model.batch_size)
        num_classes = np.unique(self.train_labels).shape[0]

        for epoch in range(start_at,epochs+start_at):
            self.reset_losses()

            train_shuffle = np.random.choice(len(unsup_train_data),size=(len(unsup_train_data),),replace=False)
            s_unsup_train_data = unsup_train_data[train_shuffle]
            last_batch=False

            for batch in range(num_batches):
                if batch == num_batches -1:
                    last_batch = True
                # Get semi-sup training data for batch
                (sup_idcs, query_idcs),(num_query,num_support) = make_training_episode(sup_train_labels,pct_for_query)
                sup_data = sup_train_data[sup_idcs]
                sup_labels = sup_train_labels[sup_idcs]

                query_data = sup_train_data[query_idcs]
                query_labels = sup_train_labels[query_idcs]
                # Get unsupervised training data for batch
                unsup_batch = s_unsup_train_data[batch*self.model.batch_size:(batch+1)*self.model.batch_size]
                # Train.
                self.train_model(unsup_batch,query_data,sup_data,num_classes,num_support,num_query,
                                 flatten=True,
                                 ae=ae,
                                 proto=proto,
                                 silh=silh,
                                 db=db,
                                 hubert=hubert,
                                 update_eoe=update_eoe,
                                 last_batch=last_batch)
            self.print_log_string(epoch,ae,proto,silh,db,hubert)

            train_loss.append(self.train_proto_loss.result())
            train_ae_loss.append(self.train_ae_loss.result())
            train_acc.append(self.train_proto_acc.result())
            self.lt_viewer.on_epoch_end(epoch, None,train_tup=(unsup_train_data,unsup_train_labels,unsup_train_subj_ids),
                                               query_tup=(query_data,query_labels),
                                               support_tup=(sup_data,sup_labels),
                                               kmeans_seed=self.seed)
            self.rc_viewer.on_epoch_end(epoch, self.train_data,None)


        latent_space = self.model.encoder(trainer.train_data)
        np.save(self.make_path("latent"), latent_space)

        model_weights = self.model.variables

        train_rand = self.lt_viewer.train_rand
        test_rand = self.lt_viewer.test_rand

        if len(train_rand) != 0:
            #print("Max Test RI was: %f at Epoch %f." % (max(test_rand), np.argmax(test_rand)))
            #cb_rand = np.stack([train_rand, test_rand], axis=-1)
            np.save(self.make_path("train_test_rand.npy"), train_rand)

        comb_losses = np.stack([train_loss,train_ae_loss],axis=-1)
        np.savetxt("losses.txt", comb_losses)

        weights_map = {v.name: v.numpy() for v in model_weights}
        pickle.dump(weights_map, open(self.make_path("weights/saved_weights.pkl"), "wb"))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",required=True,type=str)
parser.add_argument("--number_examples",required=True,type=int)
parser.add_argument("--number_epochs",required=False,default=200,type=int)
parser.add_argument("--proto",action="store_true")
parser.add_argument("--ae",action="store_true")
parser.add_argument("--silh",action="store_true")
parser.add_argument("--db",action="store_true")
parser.add_argument("--seed",type=int,required=False)
parser.add_argument("--update_eoe",type=int,default=0)
parser.add_argument("--filt_width",type=int)
args = parser.parse_args()

DATASET = args.dataset
num_examples = args.number_examples
num_epochs= args.number_epochs
use_ae = args.ae
use_proto = args.proto
use_silh = args.silh
use_db = args.db
seed = args.seed
update_eoe = False
if args.update_eoe == 1:
    update_eoe = True
filt_width=args.filt_width
trainer = Trainer()
os.environ["PYTHONHASHSEED"] = str(seed)
trainer.train_semi_sup(num_epochs,exmpls_per_class=num_examples,proto=use_proto,ae=use_ae,silh=use_silh,db=use_db,seed=seed,dataset_name=DATASET,update_eoe=update_eoe,filt_width=filt_width)