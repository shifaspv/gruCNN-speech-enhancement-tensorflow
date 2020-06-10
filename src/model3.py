#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:57:15 2018

@author: Muhammed Shifas Pv
University of Crete (UoC)
"""
from __future__ import division
import os
import sys
import math
import logging
import time
import numpy as np
import lib.util as util
import tensorflow as tf
from datetime import datetime
from lib.ops import conv, conv2D
from lib.util import l1_l2_loss
from lib.precision import _FLOATX
from lib.model_io import save_variables, get_info
from lib.util import compute_receptive_field_length
import pdb
from numpy.lib import stride_tricks
from numpy import inf
def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(), ema=None):
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX), ema=None): 
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)
   
def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter




class RNN_SE(object):
    
    def __init__(self,cfg,model_id=None):
        self.cfg=cfg        
        self.num_unit_fc = cfg['num_unit_fc'] 
#        self.network_depth=cfg['network_depth']
        self.use_biases = cfg['use_biases']
        self.l2 = cfg['L2_regularization'] 
        self.use_dropout = cfg['use_dropout']
        self.use_ema = cfg['use_ema']
        self.polyak_decay = cfg['polyak_decay']
        self.num_lstm_layers=cfg['num_lstm_layers']
        self.model_id = model_id    
        self.masker_length= cfg['masker_length']
        self.filter_size_conv_lstm=cfg[ 'filter_size_conv_lstm']
        self.num_input_frames=cfg['num_input_frames']
        self.num_channel_lstm=cfg['num_channel_lstm']
        self.batch_size=cfg['batch_size']
        self.frame_size=int(cfg['frame_size']*cfg['sample_rate'])
        self.frame_shift=int(cfg['frame_shift']*cfg['sample_rate'])
        self.create_variables()
        if self.use_ema:
            self.ema = tf.train.ExponentialMovingAverage(decay=self.polyak_decay)
        else:
            self.ema = None 
    
        
    def create_variables(self):       
        fc_unit = self.num_unit_fc
        fw_lstm=self.filter_size_conv_lstm
        DFm=int((self.masker_length/4+1))
        r_lstm=self.num_channel_lstm
        dim_out = self.masker_length+1
        with tf.variable_scope('RNN_SE'):
                      
                
            with tf.variable_scope('ConvLSTM_layer'):  
                for i, feature_dim in enumerate(self.num_lstm_layers):   
                    with tf.variable_scope('block{}'.format(i)):
#            with tf.variable_scope('Conv_LSTM_layer'): # implementing LSTM manually
                        if i==0:
                            get_weight_variable('W_zx',( fw_lstm[0], fw_lstm[1], 1, r_lstm))
                            get_weight_variable('W_zh',( fw_lstm[0], fw_lstm[1], r_lstm, r_lstm))
                            get_weight_variable('W_rx',(fw_lstm[0], fw_lstm[1], 1, r_lstm))
                            get_weight_variable('W_rh',(fw_lstm[0], fw_lstm[1], r_lstm, r_lstm))
                            get_weight_variable('W_hx',(fw_lstm[0],fw_lstm[1], 1, r_lstm))
                            get_weight_variable('W_hh',(fw_lstm[0], fw_lstm[1], r_lstm, r_lstm)) 
    #                        get_weight_variable('TD_W', (DFm*r_lstm,dim_out))  
                        else:
                            get_weight_variable('W_zx',( fw_lstm[0],fw_lstm[1], r_lstm, r_lstm))
                            get_weight_variable('W_zh',( fw_lstm[0], fw_lstm[1], r_lstm, r_lstm))
                            get_weight_variable('W_rx',(fw_lstm[0], fw_lstm[1], r_lstm, r_lstm))
                            get_weight_variable('W_rh',(fw_lstm[0], fw_lstm[1], r_lstm, r_lstm))
                            get_weight_variable('W_hx',(fw_lstm[0], fw_lstm[1], r_lstm, r_lstm))
                            get_weight_variable('W_hh',(fw_lstm[0], fw_lstm[1], r_lstm, r_lstm))    
                        get_weight_variable('alpha',(r_lstm))                          
                          
                
            with tf.name_scope('TD_layer'):
                # final Fully connected layer
                get_weight_variable('TD_W1', (DFm*r_lstm,fc_unit))
                get_weight_variable('TD_W2', (DFm*r_lstm,dim_out))
                if self.use_biases['TD_layer']:
                    get_bias_variable('TD', (dim_out))                           



    def parametric_relu(self,_x,r,ema=None):
#      r = self.n_channels  
      alphas = get_bias_variable('alpha', shape=( r), ema=ema)
      pos = tf.nn.relu(_x)
      neg = alphas * (_x - abs(_x)) * 0.5
    
      return pos + neg
      
    def TimeDistributed(self, X, ema=None):
#        pdb.set_trace()    
        # post processing is a Fully connected network (FC)
        # Fully Connected Layer
        DFm=X.shape[1].value
#        fc_unit = self.num_unit_fc

#        DFm=int((self.masker_length/2+1))
        r_lstm=self.num_channel_lstm
        dim_out = self.masker_length+1
        X_ticks=tf.unstack(X,axis=2)
        out=[]
        with tf.name_scope('TD_layer'):
            for X_t in X_ticks:
                X_t=tf.reshape(X_t,(self.batch_size,-1))
#                TD_W1=get_weight_variable('TD_W1', (DFm*r_lstm,fc_unit))
#                out_t1=tf.matmul(X_t,TD_W1)
#                out_t1=tf.nn.relu(out_t1)
                TD_W2=get_weight_variable('TD_W2', (DFm*r_lstm,dim_out))
                out_t2=tf.matmul(X_t,TD_W2)
                out_t2=tf.nn.relu(out_t2)
                out.append(out_t2)
        out=tf.transpose(tf.stack(out),(1,2,0)) 
        
#        pdb.set_trace()
        return out
      
        
    def ConvLSTM_manual(self,X,h_t0,index):
#        pdb.set_trace()

        fw=self.filter_size_conv_lstm
        r_lstm = self.num_channel_lstm
        if index==0:
            r=1
        else:
            r=r_lstm
        with tf.variable_scope('block{}'.format(index)):
            # z_t implement
#            pdb.set_trace()
            W_zx=get_weight_variable('W_zx',(fw[0],fw[1], r, r_lstm))
            W_zh=get_weight_variable('W_zh',(fw[0], fw[1], r_lstm, r_lstm))
            z_t=conv(X,W_zx)+conv(h_t0,W_zh)
            z_t=tf.sigmoid(z_t)
            
            # r_t implement
            W_rx=get_weight_variable('W_rx',(fw[0], fw[1], r, r_lstm))
            W_rh=get_weight_variable('W_rh',(fw[0], fw[1], r_lstm, r_lstm))
            r_t=conv(X,W_rx)+conv(h_t0,W_rh)
            r_t=tf.sigmoid(r_t)
            
            # h_t_hat implementation
            W_hx=get_weight_variable('W_hx',(fw[0], fw[1], r, r_lstm))
            W_hh=get_weight_variable('W_hh',(fw[0], fw[1], r_lstm, r_lstm))
            r_t_h_t=tf.multiply(r_t,h_t0)
            h_t_hat=conv(X,W_hx)+conv(r_t_h_t,W_hh)
            h_t_hat=tf.tanh(h_t_hat)
            
            
            h_t=tf.multiply(z_t,h_t0)+tf.multiply((1-z_t),h_t_hat)
            o_t=self.parametric_relu(h_t,r_lstm) ## Should move this out of loop        

    #        h_t=tf.layers.dropout(h_t,rate=0.2)
        return o_t,o_t
    
    def ConvLSTM_layer(self, layer_input,index,is_training=True,ema=None):
        r_lstm=self.num_channel_lstm
        fw=self.filter_size_conv_lstm
        DFm=layer_input.shape[1].value
#        DFm=int((self.masker_length+1))
#        X=tf.transpose(layer_input,[2,0,1,3])
        padding=tf.constant([[0,0],[0,0],[1,1],[0,0]])
        X=tf.pad(layer_input,padding, mode="CONSTANT")
#        pdb.set_trace()
        # Conv_LSTM manual implementation
#        inputs=tf.unstack(X)
        h_t0=tf.zeros(shape=(self.batch_size,DFm,fw[1],r_lstm))
        rnn_outputs=[]
        with tf.variable_scope('ConvLSTM_layer',reuse=True):
            for i in range(layer_input.shape[2].value):
                input=tf.slice(X,[0,0,i,0],[-1,-1,fw[1],-1])
#                pdb.set_trace()
                o_t, h_t0= self.ConvLSTM_manual(input,h_t0,index)

                rnn_outputs.append(o_t[:,:,1,:])
        #        pdb.set_trace()    
        rnn_outputs=tf.transpose(tf.stack(rnn_outputs),(1,2,0,3))

#        rnn_outputs=tf.nn.relu(rnn_outputs) 
        return rnn_outputs                                                                        #onput=[batch,max_time,op_length]

        
        
    
    def get_out_1_loss(self, Y_true, Y_pred):

        weight = self.cfg['loss']['out_1']['weight']
        l1_weight = self.cfg['loss']['out_1']['l1']
        l2_weight = self.cfg['loss']['out_1']['l2']


        if weight == 0:
            return Y_true * 0

        return weight * l1_l2_loss(Y_true, Y_pred, l1_weight, l2_weight)
    
    
    
    def inference(self, X, is_training, ema): 
        # Input X is mixed signal (clean_speech + noise) 
        with tf.variable_scope('RNN_SE', reuse=True):
            #X ->Causal_layer -> FFT_layer0-> ... FFT_layerN 

            # Conv Layers  

#            for i, dilation in enumerate(self.num_conv_layers):         
#                X= self.Conv_layer(X, i, is_training, ema)
            # post processing

#            X=tf.nn.max_pool(X,ksize=(1,2,1,1),strides=(1,2,1,1),padding='SAME')
            #ConvLSTM layers
            for i, dilation in enumerate(self.num_lstm_layers):         
                X= self.ConvLSTM_layer(X, i, is_training, ema)
                if ((i==1) or (i==3)):
                    X=tf.nn.max_pool(X,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')

#                pdb.set_trace()
            # Time Distributed layer
            clean_mask_pred= self.TimeDistributed(X, ema)
#            pdb.set_trace()
            clean_mask_pred=tf.expand_dims(clean_mask_pred,axis=3)
        return clean_mask_pred        
        
        
        
    def define_train_computations(self, optimizer, train_audio_conditions_reader, valid_audio_conditions_reader, global_step):
        # Train operations 
        self.train_audio_conditions_reader = train_audio_conditions_reader

        mixed_audio_train, clean_audio_train= train_audio_conditions_reader.dequeue()
        self.input=mixed_audio_train
        self.target=clean_audio_train
        
#        clean_audio_train = clean_audio_train[:, :, self.half_receptive_field:-self.half_receptive_field, :]  # target1
#        pdb.set_trace()
        clean_audio_pred = self.inference(mixed_audio_train, is_training=True, ema=None)
        self.predi=clean_audio_pred
        # Loss of train data (Time domain)
        self.train_loss = self.get_out_1_loss(clean_audio_train, clean_audio_pred)
        trainable_variables = tf.trainable_variables()
        # Loss of train dat (STFT domain)
#        pdb.set_trace()
#        # Regularization loss 
#        if self.l2 is not None:
#            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('_b' in v.name)])
#            self.train_loss += self.l2*l2_loss

        trainable_variables = tf.trainable_variables()
        
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0) # clipping the gradient
        
#        pdb.set_trace()
        self.gradients_update_op = optimizer.minimize(self.train_loss, global_step=global_step, var_list=trainable_variables)
        if self.use_ema:
            self.maintain_averages_op = tf.group(self.ema.apply(trainable_variables)) 
            self.update_op = tf.group(self.gradients_update_op, self.maintain_averages_op)
        else:
            self.update_op = self.gradients_update_op

         

        # Validation operations
        self.valid_audio_conditions_reader = valid_audio_conditions_reader

        mixed_audio_valid, clean_audio_valid = valid_audio_conditions_reader.dequeue()

#        clean_audio_valid = clean_audio_valid[:, :, self.half_receptive_field:-self.half_receptive_field, :]   # target 1

        clean_audio_pred_valid = self.inference(mixed_audio_valid, is_training=False, ema=self.ema)
     

        # Loss of validation data
        self.valid_loss = self.get_out_1_loss(clean_audio_valid, clean_audio_pred_valid)        
        
      
    def train_epoch(self, coord, sess, logger):
        self.train_audio_conditions_reader.reset()
        thread = self.train_audio_conditions_reader.start_enqueue_thread(sess) 

        total_train_loss = 0
        total_batches = 0 
        
        while (not coord.should_stop()) and self.train_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss, _ = sess.run([self.train_loss, self.update_op]) 
#            input, target, predi= sess.run([self.input,self.target,self.predi])
            
            
            if math.isnan(batch_loss):
                logger.critical('train cost is NaN')
                coord.request_stop() 
                break 
            total_train_loss += batch_loss
            total_batches += 1  
##            pdb.set_trace()  
#            axis=plt.subplot2grid([1,3],loc=[0,0])
#            ax=sns.heatmap(input[0,:,:,0], cmap="Reds", ax=axis)
#            plt.title('input image')
#            ax.invert_yaxis()
#            axis=plt.subplot2grid([1,3],loc=[0,1])
#            ax=sns.heatmap(target[0,:,:,0], cmap="Reds",ax=axis)    
#            plt.title('target image')
#            ax.invert_yaxis()
#            axis=plt.subplot2grid([1,3],loc=[0,2])
#            ax=sns.heatmap(predi[0,:,:,0], cmap="Reds",ax=axis)    
#            plt.title('target image')
#            ax.invert_yaxis()
#            plt.show()
            print( 'average_training_loss is %s' %(batch_loss))
        coord.join([thread])
        
        if total_batches > 0:  
            average_train_loss = total_train_loss/total_batches 
        
        # Plots of features
        print( 'AVARAGE TRAIN LOSS IS %s' %(average_train_loss))

        return average_train_loss         
        
    
    def valid_epoch(self, coord, sess, logger):
        self.valid_audio_conditions_reader.reset()
        thread = self.valid_audio_conditions_reader.start_enqueue_thread(sess) 

        total_valid_loss = 0
        total_batches = 0 

        while (not coord.should_stop()) and self.valid_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss = sess.run(self.valid_loss)
            if math.isnan(batch_loss):
                logger.critical('valid cost is NaN')
                coord.request_stop()
                break  
            total_valid_loss += batch_loss
            total_batches += 1  

        coord.join([thread])  

        if total_batches > 0:  
            average_valid_loss = total_valid_loss/total_batches  

        return average_valid_loss        
        
        
    def train(self, cfg, coord, sess):
        logger = logging.getLogger("msg_logger") 

        started_datestring = "{0:%Y-%m-%d, %H-%M-%S}".format(datetime.now())
        logger.info('Training of FFTNet started at: ' + started_datestring + ' using Tensorflow.\n')
        logger.info(get_info(cfg))

#        if self.use_batch_normalization and self.use_biases['filter_gate']:
#            print('Warning: Batch normalization should not be used in combination with filter and gate biases.')
#            logger.warning('Warning: Batch normalization should not be used in combination with filter and gate biases. Change the configuration file.')

        start_time = time.time()

        n_early_stop_epochs = cfg['n_early_stop_epochs']
        n_epochs = cfg['n_epochs']
       
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=4)

        early_stop_counter = 0

        min_valid_loss = sys.float_info.max
        epoch = 0
        while (not coord.should_stop()) and (epoch < n_epochs):
            epoch += 1
            epoch_start_time = time.time() 
            train_loss = self.train_epoch(coord, sess, logger) 
            valid_loss = self.valid_epoch(coord, sess, logger) 

            epoch_end_time = time.time()
                         
            info_str = 'Epoch=' + str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' + str(epoch_end_time - epoch_start_time)  
            logger.info(info_str)

            if valid_loss < min_valid_loss: 
                logger.info('Best epoch=' + str(epoch)) 
                save_variables(sess, saver, epoch, cfg, self.model_id) 
                min_valid_loss = valid_loss 
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                logger.debug('stopping early')
                break

        end_time = time.time()
        logger.info('Total time = ' + str(end_time - start_time))

        if (not coord.should_stop()):
            coord.request_stop()

    def define_generation_computations(self): 
        self.noisy_audio_test = tf.placeholder(shape=(1, 161, 850, 1), dtype=_FLOATX)
   
#        if self.lc_enabled:
#            self.lc_test = tf.placeholder(shape=(1, 1, None, self.label_dim), dtype=_FLOATX)  
#        else:
#            self.lc_test = None
#    
#        if self.gc_enabled:
#            self.gc_test = tf.placeholder(shape=(1, None), dtype=tf.int32) 
#        else:
#            self.gc_test = None   
       
        if self.use_ema:
            self.inference(self.noisy_audio_test, is_training=False, ema=None)
            self.ema.apply(tf.trainable_variables()) 

        self.clean_audio_pred_computation_graph = self.inference(self.noisy_audio_test, is_training=False, ema=self.ema) 
        self.test_loss = self.get_out_1_loss(self.noisy_audio_test, self.clean_audio_pred_computation_graph )


    def generation(self, sess, noisy_audio):
#        regain=self.cfg['regain']
#        rms_sig=util.rms(noisy_audio)
#        noisy_audio=(regain/rms_sig)*noisy_audio
#        pdb.set_trace()
        fft_noisy= self.data_segmentations_test(noisy_audio)
        fft_noisy=fft_noisy
        # input to the model 
        t=fft_noisy.shape[2]
        placeholder=np.zeros((1,161,850,1))
        
        feature_input=10*np.log10(10**2*np.abs(fft_noisy)**2+1)
#        inp_len=feature_input.shape[0]
        placeholder[:,:,0:t,:]=feature_input
#        pdb.set_trace()      
        feed_dict = {self.noisy_audio_test:placeholder}
        
        model_out = sess.run(self.clean_audio_pred_computation_graph, feed_dict=feed_dict)
        test_loss=sess.run(self.test_loss,feed_dict={self.noisy_audio_test:placeholder,self.clean_audio_pred_computation_graph:model_out})
        print('test loss', test_loss)
        model_out=model_out[:,:,0:t,:];
        fft_mag=np.sqrt((10**(model_out/10)-1)/10**2)
#        pdb.set_trace()
        fft_mag=np.squeeze(fft_mag,axis=3)
        fft_mag=np.transpose(fft_mag,(0,2,1))
        fft_mag=np.reshape(fft_mag,(-1,161))      
        fft_noisy=np.squeeze(fft_noisy,axis=3)
        fft_noisy=np.transpose(fft_noisy,(0,2,1))
        fft_noisy=np.reshape(fft_noisy,(-1,161))  
#        pdb.set_trace() 
#        mask_input_batch=np.squeeze(mask_input_batch,axis=2)
        # signal reconstruction with noisy phase
        fft_mag_noisy=np.maximum(np.abs(fft_noisy), 1e-6)
        fft_phase=fft_noisy/fft_mag_noisy 
        clean_audio_segments=np.fft.irfft(np.multiply(fft_mag,fft_phase)).astype(np.float64)
        clean_audio_pred= self.overlap_add(clean_audio_segments)          
#        clean_audio_pred = clean_audio_segments.reshape((-1, ))
        noisy_audio_segments=np.fft.irfft(np.multiply(fft_mag_noisy,fft_phase)).astype(np.float64)
        noisy_audio_pred=self.overlap_add(noisy_audio_segments)
#        noisy_audio_pred = noisy_audio_segments.reshape((-1, ))        
        return clean_audio_pred, noisy_audio_pred
    
    
    
    
    def data_segmentations_test(self,noisy_audio):
        
        n_samples = noisy_audio.shape[0] 
        num_itr=int((n_samples-self.frame_size)/self.frame_shift)  
#        pdb.set_trace()
        noisy_audio_segments = stride_tricks.as_strided(noisy_audio,
                                                        shape=(num_itr,self.frame_size),
                                                        strides=(                                                                
                                                                noisy_audio.strides[0]*self.frame_shift,
                                                                noisy_audio.strides[0]))
#        win=np.hamming(self.frame_size)
#        noisy_audio_segments *=win
        
#        n=int(num_itr/128+1)
#        arr=np.zeros((n*128,self.frame_size))
#        arr[:noisy_audio_segments.shape[0],:noisy_audio_segments.shape[1]]=noisy_audio_segments
#        noisy_audio_segments=arr
#        noisy_audio_segments=np.reshape(noisy_audio_segments,(n,self.num_input_frames,self.frame_size))
        noisy_audio_segments=np.fft.rfft(noisy_audio_segments)
#        noisy_audio_segments=np.transpose(noisy_audio_segments,[0,2,1]) 
        noisy_audio_segments=np.transpose(noisy_audio_segments,[1,0]) 
        noisy_audio_segments=np.expand_dims(noisy_audio_segments,axis=0)
        noisy_audio_segments=np.expand_dims(noisy_audio_segments,axis=3)         
        return noisy_audio_segments
    
    def overlap_add(self,audio_segments):
#        pdb.set_trace()
        L_shift=(self.frame_shift)
        L_frame=self.frame_size
        L_sig=L_shift*audio_segments.shape[0]
        offsets = range(0, L_sig, L_shift)
        res = np.zeros(L_sig+L_shift, dtype=np.float64)
#        win=np.hamming(self.frame_size)
#        audio_segments *=win
        for i, n in enumerate(offsets):
            res[n:n+L_frame] += audio_segments[i,:]
            
        return res