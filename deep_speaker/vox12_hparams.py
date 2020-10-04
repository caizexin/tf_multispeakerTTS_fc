from feeder import FeederHParams
from resnet import ResNetHParams
import os
from collections import namedtuple
import tensorflow as tf

class TrainHparams(namedtuple('TrainHparams', ('checkpoint_path',
                                              'train_steps',
                                              'checkpoint_interval',
                                              'eval_interval',
                                              ))):
    def replace(self, **kwargs):
        return super(FeederHParams, self)._replace(**kwargs)

data_dir = 'data_vox12'

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print("number gpu: " + str(num_gpus) + '\n')
else:
    import GPUtil
    available_gpus = GPUtil.getAvailable(order='last', limit=8, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    num_gpus = len(available_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))

# best when total batch size equals 128 or 256, if use one gpu, set to 128
batch_per_gpu = 64

train_feeder_hparams = FeederHParams(
    scp = os.path.join(data_dir, 'vox12_train_aug.csv'),
    spkfile = os.path.join(data_dir, 'spklist'),
    record_defaults = [tf.string] * 2,
    field_delim = '\t',
    select_cols = [0, 1],
    dtypes = [tf.float32, tf.int64],
    padded_shapes = ((None, 80), []),
    batch_size = batch_per_gpu*num_gpus,
    shuffle = True,
    shuffle_size = 1024,
    is_repeat = False,
    times = 2
)

dev_feeder_hparams = FeederHParams(
    scp = os.path.join(data_dir, 'test.csv'),
    spkfile = os.path.join(data_dir, 'test_spklist'),
    record_defaults = [tf.string] * 2,
    field_delim = '\t',
    select_cols = [0, 1],
    dtypes = [tf.float32, tf.int64],
    padded_shapes = ((None, 80), []),
    batch_size = batch_per_gpu*num_gpus,
    shuffle = False,
    shuffle_size = 1024,
    is_repeat = False,
    times = 2
)

resnet_hparams = ResNetHParams(
    num_classes = 7323,
    min_lrn_rate = 1e-4,
    lrn_rate = 0.01,
    decay_learning_rate = True,
    start_decay = 147500,
    decay_steps = 30000,
    decay_rate = 0.5,
    num_residual_units = [3, 4, 6, 3],
    use_bottleneck = False,
    weight_decay_rate = 1e-4,
    relu_leakiness = 0.0,
    optimizer = 'mom',
    # optimizer = 'sgd',
    clip_gradients = False,
    # spekaer embedding dimension
    gv_dim = 256,
    dropout_rate = 0.5,
    num_gpus = num_gpus
)

train_hparams = TrainHparams(
    checkpoint_path = 'vox12_resnet34',
    train_steps = 300000,
    checkpoint_interval = 5000,
    eval_interval = 2500000
)
