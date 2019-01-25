import numpy as np
from tensorflow.contrib.layers import optimize_loss
import tensorflow as tf
import time
import os

class DeepStateSpaceModel(object):
    """
    This class contains functions to build, train, and evaluate a deep state space model
        according to "Deep State Space Model for Time Series Forecasting"
    """

    def __init__(self, sess):
        self.all_z = np.load('data/formatted_traffic.npy') # dims (963,35,24)==(laneID, Day, Hour)
        self.train_range = config.train_range # 14, 21, or 28
        self.test_range = 7 # For now, keep 7 as the testing range
        
        self.DoW_labels = np.load('data/Day_of_Week_labels.npy') # dims(35,7)==(Day, one hot encoding)
        self.ToD_labels = np.eye(24) # dim (24,24) Time of Day one hot encoding

        #TODO: add laneID as feature vector

        self.all_x = np.array((self.DoW_labels,self.ToD_labels)) # TODO: format feature vectors  

        # Training variables
        self.batch_size = 32
        self.training_range = 28
        self.feature_size = 32 # 24 hours + 7 days + 1 laneID
        self.lstm_sizes = [32,32]
        self.keep_prob = 0.8

        # Initialize State Space variables
        self.dim_l = 2
        self.dim_z = 1

        initial_variance = 20
        
        F = np.array(np.eye(self.dim_l).astype(np.float32))

        a = np.random.randn(self.dim_l,self.dim_z).astype(np.float32)
        b = np.random.randn(self.dim_z).astype(np.float32)

        Q = np.eye(self.dim_l, dtype=np.float32)*np.random.randn(self.dim_l).astype(np.float32)
        R = np.eye(self.dim_z, dtype = np.float32)*np.random.randn(self.dim_z).astype(np.float32)

        l_0 = R*np.random.randn(self.dim_l, dtype=np.float32) + Q
        P = np.tile(initial_variance*np.eye(self.dim_l, dtype = np.float32), (self.batch_size, 1, 1))

        z_0 = np.zeros((self.dim_z,), dtype = np.float32) #TODO: whats this for?


        # Collect initial variables
        self.init_vars = dict(mu=l_0, Sigma=P, F=F, Q=Q, a=a, b=b, R=R)

        self.activation_fn = tf.nn.relu

        self.sess = sess

        # Initialize placeholders and various variables
        self.lstm_input = tf.placeholder(tf.float32, shape = [None, self.training_range, self.feature_size],\
                                         name = 'LSTM_input')
        self.lstm_output = tf.placeholder(tf.float32, shape = [None, self.training_range, self.lstm_size],\
                                          name = 'LSTM_output')

        self.kf = None
        self.updates = None
        self.model_vars = None
        # TODO: Need these summaries?
        self.train_summary = None
        self.test_summary = None

    def build_lstm_layers(self, lstm_sizes, inputs, keep_prob, batch_size):
        """
        Creates LSTM layers
        """
        lstms = tf.contrib.rnn.LSTMCell(size) for size in lstm_sizes]
        dropouts = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob) for lstm in lstms]

        cell = tf.contrib.rnn.MultiRNNCell(dropouts)
        initial_state = cell.zero_state(batch_size, tf.float32)

        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state = initial_state)

        return initial_state, lstm_outputs, cell, final_state

    def build_model(self):
        initial_state, lstm_outputs, lstm_cell, final_state = \
                       self.build_lstm_layers(self.lstm_sizes, self.lstm_input,
                                              self.keep_prob, self.batch_size)
        y = # TODO: use placeholder to designate that this will be the current set of true z values

        """
        TODO: Convert lstm_outputs to Theta parameters through a variety of affine transformations
        """
        self.kf = KalmanFilter(dim_l = self.dim_l,
                               dim_z = self.dim_z,
                               l_0 = self.init_vars['mu'],
                               P = self.init_vars['Sigma'],
                               F = self.init_vars['F'],
                               Q = self.init_vars['Q'],
                               a = self.init_vars['a'],
                               b = self.init_vars['b'],
                               R = self.init_vars['R'],
                               y = 
                               )













        





        
