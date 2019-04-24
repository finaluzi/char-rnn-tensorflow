#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
from six.moves import cPickle


from six import text_type


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='save/tangpin2',
                    help='model directory to store checkpointed models')
parser.add_argument('-n', type=int, default=1000,
                    help='number of characters to sample')
parser.add_argument('--prime', type=text_type, default=u'l',
                    help='prime text')
parser.add_argument('--sample', type=int, default=1,
                    help='0 to use max at each timestep, 1 to sample at '
                         'each timestep, 2 to sample on spaces')
parser.add_argument('--rand', type=float, default=1.0,
                    help='sample in rand')
parser.add_argument('--block', type=int, default=100,
                    help='sample block')
parser.add_argument('--input', type=bool, default=False,
                    help='human input after block')

args = parser.parse_args()

import tensorflow as tf
from model import Model


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    # Use most frequent char if no prime is given
    if args.prime == '':
        args.prime = chars[0]
    model = Model(saved_args, training=False)
    sample_block = args.block
    need_sample = args.n
    real_sample = 0
    prime_next = args.prime
    need_run_prime = True
    state_last = None

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            while need_sample > 0:
                if need_sample > sample_block:
                    real_sample = sample_block
                else:
                    real_sample = need_sample
                if need_sample < args.n:
                    need_run_prime = False
                need_sample -= sample_block
                temp_output = model.sample(sess, chars, vocab, real_sample, prime_next,
                                           args.sample, args.rand, state_last, need_run_prime)
                prime_next = temp_output[0].encode(
                    'utf-8').decode(encoding='utf-8')
                state_last = temp_output[1]
                print(prime_next, end='')
                if args.input:
                    text = input("input:")
                    if len(text) > 0:
                        # text = prime_next[-1]+text
                        model.input_sample(
                            sess, vocab, prime_next[-1]+text, state_last)
                        prime_next = text
            # print(out_put)


if __name__ == '__main__':
    sample(args)
