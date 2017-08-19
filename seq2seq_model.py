"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import scipy.misc
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
import seq2seq_for_MT
import Stack_Residual_RNNCell


def isNan(x):
    return x != x


def print_candidate(cand):
    print("---------------\nnext_t = " + str(cand[2]))
    print("logp[]: " + str(cand[0]))
    print("cand_t: " + str(cand[3]))


def normalize_possibility(p):
    p = np.exp(p)
    sump = sum(p)
    p = p / sump
    return p


def safe_log(x):
    if x <= 0:
        return -(1e10)
    return np.log(x)


class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
    """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=True,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
          dtype: the data type to use to store internal variables.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                               num_samples, self.target_vocab_size),
                    dtype)
            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        list_of_cell = []
        for layer in xrange(num_layers):
            if layer % 2 == 0:
                with tf.device('/gpu:0'):
                    single_cell = tf.nn.rnn_cell.LSTMCell(size)
                list_of_cell.append(single_cell)
            else:
                with tf.device('/gpu:1'):
                    single_cell = tf.nn.rnn_cell.LSTMCell(size)
                list_of_cell.append(single_cell)

        if num_layers > 1:
            cell = Stack_Residual_RNNCell.Stack_Residual_RNNCell(list_of_cell)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return seq2seq_for_MT.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_layers=num_layers,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode,
              dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one. 感觉这里有问题
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
              softmax_loss_function=softmax_loss_function)
          # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                        ]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets,
              lambda x, y: seq2seq_f(x, y, False),
              softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                         self.gradient_norms[bucket_id],  # Gradient norm.
                         self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
        The chosen data will be padded in pad_pair later.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            data: A tuple of size len(self.buckets) in which each element contains
                lists of pairs of input and output data that we use to create a batch.
            bucket_id: Integer, which bucket to get the batch from.

        Returns:
            encoder_inputs, decoder_inputs, decoder_outputs: corresponding to the
                chosen batch of data.
                Each of these three lists are all of length batch_size.
            max_encoder_input_length: the length of the longest input in the chosen
                batch. Will be used to decide which bucket to use in encode-decode
                process later.
            max_decoder_input_length: the length of the longest output in the chosen
                batch. Will be used to decide the number of segments that need to be
                generated in encode-decode process.
        """
        encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
        max_encoder_input_length = 0
        max_decoder_input_length = 0

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            decoder_output = decoder_input + [data_utils.EOS_ID]
            decoder_input = [data_utils.GO_ID] + decoder_input
            max_encoder_input_length = max(max_encoder_input_length, len(encoder_input))
            max_decoder_input_length = max(max_decoder_input_length, len(decoder_input))
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            decoder_outputs.append(decoder_output)

        return encoder_inputs, decoder_inputs, decoder_outputs,\
               max_encoder_input_length, max_decoder_input_length

    def pad_pair(self, encoder_inputs_ori, decoder_inputs_ori, seg_num, seg_len, bucket_id, BOS):
        """
        To pad the encoder_inputs and decoder_inputs to fit certain buckets in
        encode-decode process.

        Args:
            encoder_inputs_ori: A list of length batch_size. each element is an sentence
                in which each word is represented by its own id.
            decoder_inputs_ori: A list of length batch_size. each element is an sentence
                in which each word is represented by its own id.
            seg_num: The number of the segment that is going through the decode process.
            seg_len: The length of the segment that is going through the decode process.
                Equals to FLAGS.segment_length.
            bucket_id: The bucket to use in the encode-decode process.
            BOS: Whether this is the first segment to decode.

        Returns:
            batch_encoder_inputs, batch_decoder_inputs: Padded inputs that are ready
                to be fed in nn.
            batch_weights: Implemented as vanilla Seq2Seq.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs = []
        decoder_inputs = []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i in xrange(self.batch_size):
            dst_i = decoder_inputs_ori[i]
            src_i = encoder_inputs_ori[i] + dst_i[:min(len(dst_i), seg_num * seg_len)]

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(src_i))
            encoder_inputs.append(list(reversed(src_i + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then. GO可能和BOS是一样的？
            decoder_pad_size = decoder_size - len(dst_i) - 1
            if BOS:
                decoder_inputs.append([data_utils.GO_ID] + dst_i +
                                      [data_utils.PAD_ID] * decoder_pad_size)
            else:
                decoder_inputs.append(dst_i + [data_utils.PAD_ID] * (decoder_pad_size + 1))

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs. 这里好像就是给展开成1维的了？
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def get_segment(self, session, encoder_inputs, decoder_inputs, buckets, beam_num, samples_per_beam, seg_len, BOS):
        """
        Generate a segment using stochastic sampling with segment-by-segment reranking
        as described in "Generating High-Quality and Informative Conversation Responses
        with Sequence-to-Sequence Models".
        A beam sampling process.

        Args:
            session: The session used to encode-decode
            encoder_inputs: A list of length batch_size. Each element is generated by
                concatenating the original input sentence and the segments that have
                been decoded.
            decoder_inputs: A list of length batch_size. Each element is an fracture
                that only consists of the last word of previous segment generated by
                the decoder.
            buckets: The buckets used in encode-decode.
            beam_num: FLAGS.beam_num. 'B' in original paper.
            samples_per_beam: FLAGS.samples_per_beam. 'D' in original paper.
            seg_len: FLAGS.segment_length. 'K' in original paper.
            BOS: Whether this is the first segment to decode.

        Returns:
            dst_os: A list of length beam_num. Each element is a segment generated
                through beam sampling.
            logp: A list of length beam_num. Each element represent the 'possibility'
                of the segment in dst_os correspondingly.
        """
        def choose_bucket(max_input_len):
            for bid in range(len(buckets)):
                input_bound, _ = buckets[bid]
                if max_input_len < input_bound:
                    return bid

        done = [False] * beam_num  # 每个分支是否已经产生结束
        logp = [0.0] * beam_num  # 每个分支选每个词的概率取log后求和的结果……吧
        dst_os = [decoder_inputs] * beam_num  # 现在已经得到的每个beam的decode的输出，初始时是上一次被选中的segment的最后一个词（或者EOS）
        # beamSample的时候每次在每个beam后面接上一个词
        for _ in range(seg_len):
            print("=================================LOOP COUNT: " + str(_) + "=============================================")
            print("ALL BEAMS:")
            print(dst_os)
            candidates = {}  # 存放所有可能的B * D个candidates,key是decode的结果,val=(logp, 来自第几个分支，最后一个符号，整个句子)

            # 对每一个beam，产生接上下一个词的概率
            src_i = []
            dst_i =[]
            for j in range(beam_num):
                src_i.append(encoder_inputs + dst_os[j])
                dst_i.append([dst_os[j][-1]])
            bucket_id = choose_bucket(len(encoder_inputs) + _)  # '_' 就是现在decode出来的长度咯
            src_i, dst_i, target_weights = self.pad_pair(src_i, dst_i, 0, 0, bucket_id, (_ == 0 and BOS))
            _, _, output_logits = self.step(session, src_i, dst_i, target_weights, bucket_id, True)
           
            """fin = open("output_logits.txt", 'w') 
            fin.write("***********************\n Output_logits:\n ")
            for ii in range(len(output_logits[0][0])):
                fin.write(str(output_logits[0][0][ii]) + '\n')
            fin.close()"""

            # 遍历每一个beam，分别接上D个词，产生D个candidates，存进candidates（dict）里面
            for j in range(beam_num):
                if done[j]:
                    cand_t = tuple(dst_os[j])  # 把decoder的输出弄成一个tuple作为key
                    candidates[cand_t] = (logp[j], j, data_utils.EOS_ID, cand_t)
                else:
                    # 选出D个最可能的词，D = samples_per_beam,这里是下标【或许要的就是下标？因为下标和词是一一对应的
                    p = normalize_possibility(output_logits[j][0])
                    selected = np.random.choice(range(len(output_logits[j][0])), size=samples_per_beam,
                                                replace=False, p=p)  # * (1.0 - 1e-7))
                    print("******************************************\nFor beam " + str(j) +": " + str(dst_os[j]) + ", selected " + str(len(selected)) + " words: ")
                    print(selected)
                    
                    # 把这D个词接在当前的beam后面，产生D个candidates
                    for next_t in selected:
                        # cand_t = tuple(dst_os[j][:-1] + [next_t])  # 把D个词之一接在当前分支后面，得到了一个candidate
                        cand_t = tuple(dst_os[j] + [next_t])  # 把D个词之一接在当前分支后面，得到了一个candidate
                        candidates[cand_t] = (logp[j] + safe_log(output_logits[0][j][next_t]), j, next_t, cand_t)
                        print_candidate(candidates[cand_t])

            # 现在从得到的所有candidates（应该是B*D个，可能更少）当中选择B个
            candidates = list(candidates.values())  # 变成list
            p = np.array([x[0] for x in candidates])  # 所有产生的片段所对应的logp
            print("***************************\nPossibilities:")
            print(p)
            # p = p - scipy.misc.logsumexp(p)  # 一个不懂的操作
            p = normalize_possibility(p)
            print(p)
            print("length of candidates: " + str(len(candidates)))
            selected = np.random.choice(range(len(candidates)), size=beam_num, replace=False, p=p)  # 随机选B个
            print("****************************\nSelect B candidates to use in next loop:")
            print(selected)
            selected = [candidates[k] for k in selected]  # 选中的那B个
            for s in selected:
                print_candidate(s)

            # 一些复制操作，为下一轮做准备
            new_examples = [None] * beam_num
            new_logp = [0.0] * beam_num
            ndone = 0  # 结束的beam数
            for j in range(beam_num):
                new_logp[j], old_j, next_t, new_examples[j] = selected[j]  # 继承了之前的logp
                new_examples[j] = list(new_examples[j])
                if next_t == data_utils.EOS_ID:
                    done[j] = True
                    ndone += 1
                else:
                    done[j] = False
            logp = new_logp
            dst_os = new_examples
            if ndone >= beam_num:  # 所有beam都已经生成结束了
                return dst_os, logp
        return dst_os, logp

    def generate_response(self, session, question, buckets, beam_num, samples_per_beam, seg_len):
        """
        Generate a response for the inputed question using stochastic sampling with segment-by-
        segment reranking as described in "Generating High-Quality and Informative Conversation
        Responses with Sequence-to-Sequence Models"

        Args:
            session: The session used to encode-decode
            question: A list. The input sentence that have been tokenized. Not padded yet.
            buckets: The buckets used in encode-decode.
            beam_num: FLAGS.beam_num. 'B' in original paper.
            samples_per_beam: FLAGS.samples_per_beam. 'D' in original paper.
            seg_len: FLAGS.segment_length. 'K' in original paper.

        Returns:
            dst_generated: A list of length beam_num. Each element is a segment generated
                through beam sampling.
        """
        dst_generated = []  # 存放已经decode出来的内容
        max_s_len = 100  # 100是encoder那边最大的输入大小
        while not dst_generated or dst_generated[-1] != data_utils.EOS_ID:  # deocde过程
            if dst_generated:
                if len(dst_generated) + len(src_i) + 1 > max_s_len:
                    src_i = [data_utils.GO_ID] + dst_generated[:max_s_len - 2] + [data_utils.EOS_ID]
                else:
                    src_i = question + [data_utils.GO_ID] + dst_generated[:-1] + [data_utils.EOS_ID]
                dst_i = dst_generated[-1:]  # decoder的输入是之前已经decode出来的内容中的最后一个/none
            else:
                src_i = question + [data_utils.GO_ID]  # 初始时encoder那边只有输入的src
                dst_i = [data_utils.GO_ID]  # decoder输入相当于只有一个>
            if len(src_i) > max_s_len:
                src_i = src_i[-max_s_len:]

            # 产生B个segment片段
            candidate_response, _ = self.get_segment(session, src_i, dst_i, buckets, beam_num,
                                                     samples_per_beam, seg_len, (not dst_generated))
            dst_generated += candidate_response[0]  # 从备选回复中挑一个，更新decode结果，接上了这次deocde出来的内容【这里感觉有问题！选择方法不是reranking吗】
            # print self.infer.IdsToText(dst_generated)
            if len(dst_generated) + len(src_i) + 1 > max_s_len:
                break
        if dst_generated[-1] == data_utils.EOS_ID:
            dst_generated = dst_generated[:-1]
        return dst_generated
