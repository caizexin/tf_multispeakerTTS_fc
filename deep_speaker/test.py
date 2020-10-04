import os, time
import traceback
from datetime import datetime
import numpy as np
from tqdm import tqdm
from infolog import log
from utils import ValueWindow

import tensorflow as tf
from tensorflow.errors import OutOfRangeError
from feeder_wav import Feeder
from resnet import ResNet
from vox12_hparams import train_feeder_hparams, dev_feeder_hparams, resnet_hparams, train_hparams



def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def create_feeder(num_frames, handle):
    train_feeder = Feeder(hparams=train_feeder_hparams)
    train_iterator = train_feeder(num_frames)
    dev_feeder = Feeder(hparams=dev_feeder_hparams)
    dev_iterator = dev_feeder(num_frames)

    iterator = tf.data.Iterator.from_string_handle(handle,
        train_iterator.output_types,
        train_iterator.output_shapes)

    return iterator, train_iterator, dev_iterator

def create_model(fbanks, labels):
    with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
        train_resnet = ResNet(resnet_hparams, fbanks, labels, 'train')
        train_resnet.build_graph()
    with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
        eval_resnet = ResNet(resnet_hparams, fbanks, labels, 'eval')
        eval_resnet.build_graph()

    return train_resnet, eval_resnet

os.environ["CUDA_VISIBLE_DEVICES"]="4"

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True


tf.reset_default_graph()
graph = tf.Graph()

with graph.as_default() as g:

    num_frames = tf.placeholder(tf.int32, [], name='num_frames')

    handle = tf.placeholder(tf.string, shape=[])

    with tf.device('/cpu:0'):
        iterator, train_iterator, dev_iterator = create_feeder(num_frames, handle)
        fbanks, labels = iterator.get_next()
    
    min_frames = 400
    max_frames = 600

    real_frames = tf.cast(tf.random_uniform((), min_frames, max_frames), dtype=tf.int32)
    real_frames = tf.maximum(tf.minimum(real_frames, max_frames), min_frames)
    
    
    fbanks = fbanks[:, :real_frames, :]

    train_resnet, eval_resnet = create_model(fbanks, labels)

    init_op = tf.global_variables_initializer(), tf.local_variables_initializer()
    
    os.makedirs(train_hparams.checkpoint_path, exist_ok=True)
    checkpoint_path = os.path.join(train_hparams.checkpoint_path, 'gvector.ckpt')
    saver = tf.train.Saver(max_to_keep=20)

    log('Automatic speaker valification training set '
        'to a maximum of {} steps.'.format(train_hparams.train_steps))
    
    with tf.Session(graph=g, config=tfconfig)  as sess:
        try:
            sess.run(init_op)
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())

            train_iterator_variable = tf.data.experimental.make_saveable_from_iterator(train_iterator)
            tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, train_iterator_variable)

            sess.run(train_iterator.initializer, {num_frames: max_frames})
            sess.run(dev_iterator.initializer, {num_frames: max_frames})

            try:
                checkpoint_state = tf.train.get_checkpoint_state(train_hparams.checkpoint_path)

                if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                    log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
                    saver.restore(sess, checkpoint_state.model_checkpoint_path)
                else:
                    log('No model to load at {}'.format(train_hparams.checkpoint_path), slack=True)
                    saver.save(sess, checkpoint_path, global_step=train_resnet.global_step)
            except OutOfRangeError as e:
                log('Cannot restore checkpoint: {}'.format(e), slack=True)
            print(tf.global_variables())

        except Exception as e:
            log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
    




