from Resnet import ResNetHParams
import os
from collections import namedtuple
import tensorflow as tf


if 'CUDA_VISIBLE_DEVICES' in os.environ:
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print("number gpu: " + str(num_gpus) + '\n')
else:
    import GPUtil
    available_gpus = GPUtil.getAvailable(order='last', limit=8, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    num_gpus = len(available_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))



resnet_hparams = ResNetHParams(
    num_classes = 7323,
    min_lrn_rate = 1e-4,
    lrn_rate = 4e-3,
    decay_learning_rate = True,
    start_decay = 147500,
    decay_steps = 30000,
    decay_rate = 0.5,
    num_residual_units = [3, 4, 6, 3],
    use_bottleneck = False,
    weight_decay_rate = 1e-9,
    relu_leakiness = 0.0,
    optimizer = 'adam',
    # optimizer = 'sgd',
    clip_gradients = False,
    gv_dim = 256,
    dropout_rate = 0.5,
    num_gpus = num_gpus
)

