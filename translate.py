"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import codecs  # in order to write utf-8 files

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import seq2seq_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 640, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 101414, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 92577, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "dialog_data_new", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "dialog_data_new", "Training directory.")
tf.app.flags.DEFINE_string("ckpt_dir", "check_point", "Checkpoint directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_integer("segment_length", 10, "Value 'K' in original paper.")
tf.app.flags.DEFINE_integer("beam_num", 2, "Used in BeamSampling.")
tf.app.flags.DEFINE_integer("samples_per_beam", 10, "Value D. Create D words to append to each beam.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
buckets_pair_size = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
_buckets = [(20, FLAGS.segment_length + 1),
            (40, FLAGS.segment_length + 1),
            (60, FLAGS.segment_length + 1),
            (80, FLAGS.segment_length + 1),
            (100, FLAGS.segment_length + 1)]


def choose_bucket(max_input_len):
    for bucket_id in range(len(_buckets)):
        input_bound, _ = _buckets[bucket_id]
        if max_input_len < input_bound:
            return bucket_id

def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
            output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets_pair_size):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.en_vocab_size,
        FLAGS.fr_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    """Train a en->fr translation model using WMT data."""
    # Prepare WMT data.
    print("Preparing WMT data in %s" % FLAGS.data_dir)
    en_train, fr_train, en_dev, fr_dev = data_utils.prepare_data(FLAGS.data_dir)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)
        dev_set = read_data(en_dev, fr_dev)
        train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets_pair_size))]
        train_total_size = float(sum(train_bucket_sizes))  # 总共用于训练的pair数

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]  # 其实是按每个bucket里面的东西多少分类的……

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))   # 落到哪个bucket的区间里就用哪个bucket 其实是随机抽的
                            if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs_ori, decoder_inputs_ori, decoder_outputs,\
                max_encoder_input_len, max_decoder_input_len = model.get_batch(train_set, bucket_id)

            for segNum in range(int(max_decoder_input_len / 10)):
                tmp_max_encoder_input_len = segNum * 10 + max_encoder_input_len
                #  choose a bucket(depends on the encoder length)
                bucket_id = choose_bucket(tmp_max_encoder_input_len)

                encoder_inputs, decoder_inputs, target_weights = model.pad_pair(encoder_inputs_ori,
                                                                                decoder_inputs_ori,
                                                                                segNum, FLAGS.segment_length,
                                                                                bucket_id, segNum == 0)
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint

            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                  sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.ckpt_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                  if len(dev_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % (bucket_id))
                    continue
                  encoder_inputs_ori, decoder_inputs_ori, decoder_outputs,\
                  max_encoder_input_len, max_decoder_input_len = model.get_batch(train_set, bucket_id)

                  eval_loss = 0.0
                  for segNum in range(int(max_decoder_input_len / 10)):
                    tmp_max_encoder_input_len = segNum * 10 + max_encoder_input_len
                    #  choose a bucket(depends on the encoder length)
                    bucket_id = choose_bucket(tmp_max_encoder_input_len)

                    encoder_inputs, decoder_inputs, target_weights = model.pad_pair(encoder_inputs_ori,
                                                                                decoder_inputs_ori,
                                                                                segNum, FLAGS.segment_length,
                                                                                bucket_id, segNum == 0)
                    _, tmp_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_loss += tmp_loss

                  eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                  print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = FLAGS.beam_num  # 在beamSample的时候，所有的beam一起丢进去，产生接上下一个词的概率

        # Load vocabularies.
        q_vocab, r_vocab = data_utils.get_vocab(FLAGS.data_dir)
        r_vocab_reversed = data_utils.get_reverse_vocab_dict(r_vocab)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        question = sys.stdin.readline()
        while question:
            if question == "quit":
                quit()
            # Get token-ids for the input sentence.
            src_i = data_utils.tokenize_sentence(q_vocab, question)
            response = model.generate_response(sess, src_i, _buckets, FLAGS.beam_num,
                                               FLAGS.samples_per_beam, FLAGS.segment_length)
            # Print out respond corresponding to outputs.
            rst = " ".join([tf.compat.as_str(r_vocab_reversed[response_word]) for response_word in response])
            rst.encode('utf-8')
            fout = codecs.open("chinese.txt", 'a', 'utf-8')
            fout.write(rst + '\n')
            fout.close()
            print(rst)
            print(" ".join([tf.compat.as_str(r_vocab_reversed[response_word]) for response_word in response]))
            print("> ", end="")
            sys.stdout.flush()
            question = sys.stdin.readline()
            

def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                           5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tf.initialize_all_variables())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                     bucket_id, False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
  tf.app.run()
