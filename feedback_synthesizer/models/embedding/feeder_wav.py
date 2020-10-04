import tensorflow as tf
import numpy as np
import os
import random

from collections import namedtuple

from datasets import audio
from hparams import hparams as audio_hparams

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

    def _process_wave(self, wav_file, num_frames):
        try:
            wav = audio.load_wav(wav_file, sr=audio_hparams.sample_rate)
        except FileNotFoundError:
            print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_file))

        if audio_hparams.trim_silence:
            wav = audio.trim_silence(wav, audio_hparams)

        expect_len =  num_frames * audio_hparams.hop_size + audio_hparams.win_size
        if len(wav) < expect_len:
            wav = np.concatenate([wav] * np.math.ceil(expect_len / len(wav)))

        if len(wav) > expect_len:
            sp = random.randint(0, len(wav)-expect_len)
            wav = wav[sp: sp+expect_len]

        wav = audio.preemphasis(wav, audio_hparams.preemphasis, audio_hparams.preemphasize)

        if audio_hparams.rescale:
            wav = wav / np.abs(wav).max() * audio_hparams.rescaling_max

        mels = audio.melspectrogram(wav, audio_hparams).astype(np.float32).T
        return mels

    def _parse_func(self, spkid, wav_file, num_frames):
        fbanks = self._process_wave(wav_file.decode(), num_frames)
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

        dataset = dataset.map(lambda spkid, wav_file:
                              tuple(tf.py_func(self._parse_func, [spkid, wav_file, num_frames], hp.dtypes)), num_parallel_calls=-1)

        dataset = dataset.shuffle(buffer_size=hp.shuffle_size, reshuffle_each_iteration=True) if hp.shuffle==True else dataset

        dataset = dataset.repeat(hp.times) if hp.is_repeat==True else dataset

        # dataset = dataset.padded_batch(hp.batch_size, padded_shapes=hp.padded_shapes)
        dataset = dataset.batch(hp.batch_size, drop_remainder=True)

        iterator = dataset.make_initializable_iterator()

        return iterator
