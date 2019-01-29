import numpy as np
from tensorflow.contrib.layers import optimize_loss
import tensorflow as tf
import time
import os
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import to_categorical


class DeepStateSpaceModel(object):
    """
    This class contains functions to build, train, and evaluate a deep state space model
        according to "Deep State Space Model for Time Series Forecasting"
    """

    def __init__(self, sess):
        self.all_z = np.load('data/formatted_traffic.npy') # dims (963,35,24)==(laneID, Day, Hour)
        self.train_range = 28 # 14, 21, or 28
        self.test_range = 7 # For now, keep 7 as the testing range

        self.train_z = np.reshape(self.all_z[:,:self.train_range,:], [self.all_z.shape[0], -1])
        self.test_z = np.reshape(self.all_z[:,self.train_range:,:], self.all_z.shape[0], -1])
        del self.all_z

        mlb = MultiLabelBinarizer()
        self.train_onehot_ToD_DoW = mlb.fit_transform([((time//24)%7, (time%24)+7) \
                                                       for time in range(self.train_z.shape[1])])
        self.test_onehot_ToD_DoW = mlb.fit_transform([((time//24)%7, (time%24)+7) \
                                                       for time in range(self.test_z.shape[1])])

        self.onehot_laneID = to_categorical(self.train_z.shape[0]).astype(int)

        # Training variables
        self.batch_size = 32
        self.training_range = 28
        self.feature_size = 32 # 24 hours + 7 days + 1 laneID
        self.lstm_sizes = [32,32]
        self.keep_prob = 0.8
        self.learning_rate = 0.2
        self.max_grad_norm = 20

        # Initialize State Space variables
        self.dim_l = 31 # seasonal model, so 24 hours + 7 days is the size of the season
        self.dim_z = 1

        initial_variance = 20
        
        F = np.array(np.eye(self.dim_l).astype(np.float32))

        a = np.random.randn(self.dim_l,self.dim_z).astype(np.float32)
        b = np.random.randn(self.dim_z).astype(np.float32)

        Q = np.eye(self.dim_l, dtype=np.float32)*np.random.randn(self.dim_l).astype(np.float32)
        R = np.eye(self.dim_z, dtype = np.float32)*np.random.randn(self.dim_z).astype(np.float32)


        # TODO: figure out best way to initialize these
        l_0 = R*np.random.randn(self.dim_l, dtype=np.float32) + Q
        P = np.tile(initial_variance*np.eye(self.dim_l, dtype = np.float32), (self.batch_size, 1, 1))

        y_0 = np.zeros((self.dim_z,), dtype = np.float32) #TODO: Is this as an initial prediction?


        # Collect initial variables
        self.init_vars = dict(mu=l_0, Sigma=P, F=F, Q=Q, a=a, b=b, R=R, y_0 = y_0)

        self.activation_fn = tf.nn.relu

        self.sess = sess

        # Initialize placeholders and various variables
        self.lstm_input = tf.placeholder(tf.float32, shape = [None, self.training_range, self.feature_size],\
                                         name = 'LSTM_input')
        self.lstm_output = tf.placeholder(tf.float32, shape = [None, self.training_range, self.lstm_size],\
                                          name = 'LSTM_output')
        self.train_z = tf.placeholder(tf.float32, shape = [None, self.training_range], name = 'train_y')
        self.test_z = tf.placeholder(tf.float32, shape = [None, self.test_range], name = 'test_y')

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

        self.train_z = # TODO: use placeholder to designate that this will be the current set of true z values

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
                               y_0 = self.init_vars['y_0'],
                               z = self.train_z
                               )

        l_a_posteriori, P_a_posteriori, F, Q, a, b, R = self.kf.filter()

        # TODO: May need more variables than this? For computation of loss
        self.model_vars = dict(l = l_a_posteriori, P = P_a_posteriori, F=F, Q=Q, a=a, b=b, R=R)

        return self

    def build_loss(self):
        loss, kf_log_probs, l_smooth = self.kf.loss(self.model_vars['l'],
                                                    self.model_vars['P'],
                                                    self.model_vars['F'],
                                                    self.model_vars['Q'],
                                                    self.model_vars['a'],
                                                    self.model_vars['R']
                                                    )

        _vars = tf.trainable_variables()

        self.updates = optimize_loss(loss = loss,
                                     learning_rate = self.learning_rate,
                                     optimizer = 'Adam',
                                     clip_gradients = self.max_grad_norm,
                                     variables = _vars,
                                     name = 'updates')

        return self

    def initialize_variables(self):
        # TODO: give option to load variables from previous training

        self.sess.run(tf.global_variables_initializer())

        return self

    def train(self):
        













        





        
