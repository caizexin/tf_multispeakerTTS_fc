# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


class ResNetHParams(namedtuple('ResNetHParams',
    ('num_classes', 'min_lrn_rate', 'lrn_rate',
        'decay_learning_rate', 'start_decay',
        'decay_steps', 'decay_rate',
        'num_residual_units', 'use_bottleneck',
        'weight_decay_rate', 'relu_leakiness',
        'optimizer', 'gv_dim', 'clip_gradients',
        'dropout_rate', 'num_gpus'))):
    def replace(self, **kwargs):
        return super(ResNetHParams, self)._replace(**kwargs)


tf.logging.warning("models/research/resnet is deprecated. "
                   "Please use models/official/resnet instead.")


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, fbanks, labels, mode):
        """ResNet constructor.
        Args:
            hps: Hyperparameters.
            fbanks: Batches of fbanks. [batch_size, time_step, fbank_dim]
            labels: Batches of labels. [batch_size, num_classes]
            mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self.fbanks = tf.expand_dims(fbanks, axis=-1)
        self.labels = labels
        self.mode = mode

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.train.get_or_create_global_step()
        # self._build_model(self.fbanks, self.labels)
        self._build_model_multi_gpu(self.fbanks, self.labels)
        if self.mode == 'train':
            # self._build_train_op()
            self._build_train_op_multi_gpu()
        self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model_multi_gpu(self, fbanks, labels):
        with tf.device('/cpu:0'):
            tower_fbanks = tf.split(fbanks, self.hps.num_gpus)
            tower_labels = tf.split(labels, self.hps.num_gpus)

        tower_gvs = list()
        tower_predictions = list()
        tower_costs = list()
        tower_accuracies = list()
        tower_gradients = list()
        for i in range(self.hps.num_gpus):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/cpu:0', worker_device='/gpu:{}'.format(i))):
                with tf.variable_scope(''):
                    self._build_model(tower_fbanks[i], tower_labels[i])
                    tower_gvs.append(self.gv)
                    tower_predictions.append(self.predictions)
                    tower_costs.append(self.cost)
                    tower_accuracies.append(self.accuracy)

        self.gv = tf.concat(axis=0, values=tower_gvs)
        self.predictions = tf.concat(axis=0, values=tower_predictions)
        self.cost = tf.reduce_mean(tf.stack(tower_costs))
        self.accuracy = tf.reduce_mean(tf.stack(tower_accuracies))
        self.costs = tower_costs

    def _build_model(self, fbanks, labels):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = fbanks
            x = self._conv('init_conv', x, 3, 1, 16, self._stride_arr(1))

        strides = [1, 2, 2, 2]
        activate_before_residual = [True, True, True, True]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual
            filters = [16, 32, 64, 128]
            # filters = [32, 64, 128, 256]
            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # filters = [16, 160, 320, 640]
            # Update hps.num_residual_units to 4

        with tf.variable_scope('unit_0_0'):
            x = res_func(x, 16, filters[0], self._stride_arr(strides[0]),
                                     activate_before_residual[0])
        for i in six.moves.range(1, self.hps.num_residual_units[0]):
            with tf.variable_scope('unit_0_%d' % i):
                x = res_func(x, filters[0], filters[0], self._stride_arr(1), False)

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[1]),
                                     activate_before_residual[1])
        for i in six.moves.range(1, self.hps.num_residual_units[1]):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[2]),
                                     activate_before_residual[2])
        for i in six.moves.range(1, self.hps.num_residual_units[2]):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[3]),
                                     activate_before_residual[3])
        for i in six.moves.range(1, self.hps.num_residual_units[3]):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_mean_std_pool(x)

        with tf.variable_scope('vector'):
            self.gv = self._fully_connected(x, self.hps.gv_dim)
            # self.gv = self.gv / tf.norm(self.gv, ord=2, axis=-1, keep_dims=True)
            dropout_gv = tf.nn.dropout(self.gv, keep_prob=1-self.hps.dropout_rate) if self.mode == 'train' else self.gv

        with tf.variable_scope('logit'):
            logits = self._fully_connected(dropout_gv, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            # tf.summary.scalar('cost', self.cost)

        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(tf.argmax(self.predictions, 1, output_type=tf.int32), tf.cast(labels, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            # tf.summary.scalar('accuracy', self.accuracy)

    def _compute_gradient(self, cost):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(cost, trainable_variables)
        return grads

    def _average_gradients(self, gradients):
        avg_grads = list()
        for grads_per_gpu in zip(*gradients):
            grads = list()
            for g in grads_per_gpu:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            avg_grads.append(grad)
        return avg_grads

    def _build_train_op_multi_gpu(self):
        # self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        self.lrn_rate = self._learning_rate_decay()
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        gradients = list()
        for i, cost in enumerate(self.costs):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/cpu:0', worker_device='/gpu:{}'.format(i))):
                gradients.append(self._compute_gradient(cost))

        with tf.device('/cpu:0'):
            grads = self._average_gradients(gradients)

            if self.hps.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hps.optimizer == 'mom':
                optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            elif self.hps.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lrn_rate, 0.9, 0.999, 1e-6)

            if self.hps.clip_gradients:
                grads, _ = tf.clip_by_global_norm(grads, 1.)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                apply_op = optimizer.apply_gradients(
                        zip(grads, trainable_variables),
                        global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        # self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        self.lrn_rate = self._learning_rate_decay()
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate, 0.9, 0.999, 1e-6)

        # grads, trainable_variables = zip(*optimizer.compute_gradients(self.cost))

        if self.hps.clip_gradients:
            grads, _ = tf.clip_by_global_norm(grads, 1.)

        apply_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                    'beta', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                    'gamma', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                moving_variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                        moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                        moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)
                # tf.summary.histogram(mean.op.name, mean)
                # tf.summary.histogram(variance.op.name, variance)
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                    x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _residual(self, x, in_filter, out_filter, stride,
                                activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('sub_add_conv', orig_x, 3, in_filter, out_filter, stride)
                '''
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'SAME')
                orig_x = tf.pad(
                        orig_x, [[0, 0], [0, 0], [0, 0],
                                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
                '''
            x += orig_x

        tf.logging.debug('fbank after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                                                     activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('fbank after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                    'DW', [filter_size, filter_size, in_filters, out_filters],
                    tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(2.0/n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        # x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
                'DW', [x.get_shape()[1], out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                                                initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _global_mean_std_pool(self, x):
        assert x.get_shape().ndims == 4
        mean, std = tf.nn.moments(x, axes=[1, 2])
        return tf.concat([mean, std], axis=-1)

    def _learning_rate_decay(self):
        if not self.hps.decay_learning_rate:
            return tf.constant(self.hps.lrn_rate, tf.float32)

        lr = tf.train.exponential_decay(self.hps.lrn_rate,
                self.global_step - self.hps.start_decay,
                self.hps.decay_steps,
                self.hps.decay_rate)

        return tf.minimum(tf.maximum(lr, self.hps.min_lrn_rate), self.hps.lrn_rate)
