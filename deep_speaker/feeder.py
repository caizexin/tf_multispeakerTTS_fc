import tensorflow as tf
import numpy as np
import os
import random

from collections import namedtuple
class FeederHParams(namedtuple('FeederHParams',
                               ('scp', 'record_defaults', 'field_delim',
                               'select_cols', 'dtypes', 'padded_shapes',
                               'batch_size', 'shuffle', 'shuffle_size',
                               'is_repeat', 'times', 'spkfile'))):
    def replace(self, **kwargs):
        return super(FeederHParams, self)._replace(**kwargs)

class Feeder:
    def __init__(self, hparams):
        self.hp = hparams

    def _parse_func(self, spkid, mel_file, num_frames):
        fbanks = np.load(os.path.join(mel_file.decode()))
        remainder = len(fbanks) - num_frames

        if remainder > 0:
            sp = random.randint(0, remainder)
            fbanks = fbanks[sp: sp+num_frames]
        elif remainder < 0:
            while len(fbanks) < num_frames:
                fbanks = np.concatenate([fbanks, fbanks[:num_frames-len(fbanks)]], axis=0)

        return fbanks, self.spk_dict[spkid.decode()]

    def __call__(self, num_frames):
        hp = self.hp

        with open(hp.spkfile) as fid:
            spklist = [line.strip() for line in fid.readlines() if len(line.strip())!=0]

        self.spk_dict = {spkid: idx for idx, spkid in enumerate(spklist)}

        dataset = tf.data.experimental.CsvDataset(hp.scp,
                                                 record_defaults=hp.record_defaults,
                                                 field_delim=hp.field_delim,
                                                 select_cols=hp.select_cols)

        dataset = dataset.map(lambda spkid, mel_file:
                              tuple(tf.py_func(self._parse_func, [spkid, mel_file, num_frames], hp.dtypes)), num_parallel_calls=-1)

        dataset = dataset.shuffle(buffer_size=hp.shuffle_size, reshuffle_each_iteration=True) if hp.shuffle==True else dataset

        dataset = dataset.repeat(hp.times) if hp.is_repeat==True else dataset

        # dataset = dataset.padded_batch(hp.batch_size, padded_shapes=hp.padded_shapes)
        dataset = dataset.batch(hp.batch_size, drop_remainder=True)

        iterator = dataset.make_initializable_iterator()

        return iterator
