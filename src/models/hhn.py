# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import namedtuple
import numpy as np
import tensorflow as tf
import lib
import math
import pdb

FLAGS = tf.app.flags.FLAGS

# NB: batch_size could be unspecified (None) in decode mode
HParams = namedtuple("HParams", "mode, lr, min_lr, dropout, batch_size,"
                     "max_num_doc, max_doc_len, max_entity_len,"
                     "max_num_cands, emb_dim, type_emb_dim, word_conv_filter,"
                     "word_conv_width, hop_net_rnn_layers, hop_net_rnn_num_hid,"
                     "max_grad_norm, decay_step, decay_rate")


def CreateHParams():
  """Create Hyper-parameters from tf.app.flags.FLAGS"""

  assert FLAGS.mode in ["train", "predict"], "Invalid mode."

  hps = HParams(
      mode=FLAGS.mode,
      lr=FLAGS.lr,
      min_lr=FLAGS.min_lr,
      dropout=FLAGS.dropout,
      batch_size=FLAGS.batch_size,
      max_num_doc=FLAGS.max_num_doc,  # max number of supporting documents
      max_doc_len=FLAGS.max_doc_len,  # max number of words in a document
      max_entity_len=FLAGS.max_entity_len,  # max number of tokens in an entity
      max_num_cands=FLAGS.max_num_cands,  # max number of candidates
      emb_dim=FLAGS.emb_dim,
      type_emb_dim=FLAGS.type_emb_dim,  # emb_dim for type_vocab
      word_conv_filter=FLAGS.word_conv_filter,
      word_conv_width=FLAGS.word_conv_width,
      hop_net_rnn_layers=FLAGS.hop_net_rnn_layers,
      hop_net_rnn_num_hid=FLAGS.hop_net_rnn_num_hid,
      max_grad_norm=FLAGS.max_grad_norm,
      decay_step=FLAGS.decay_step,
      decay_rate=FLAGS.decay_rate)
  return hps


class HierHopNet(object):
  """ A baseline for QAngaroo dataset called ``Hierarchical Hopping Network''. """

  def __init__(self, hps, input_vocab, type_vocab, num_gpus=0):
    if hps.mode not in ["train", "decode"]:
      raise ValueError("Only train and decode mode are supported.")

    self._hps = hps
    self._input_vocab = input_vocab
    self._type_vocab = type_vocab
    self._num_gpus = num_gpus

  def build_graph(self):
    self._allocate_devices()
    self._add_placeholders()
    self._build_model()
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    if self._hps.mode == "train":
      self._add_loss()
      self._add_train_op()

    self._summaries = tf.summary.merge_all()

  def _allocate_devices(self):
    num_gpus = self._num_gpus

    if num_gpus == 0:
      raise ValueError("Current implementation requires at least one GPU.")
    elif num_gpus == 1:
      self._device_0 = "/gpu:0"
      self._device_1 = "/gpu:0"
    elif num_gpus > 1:
      self._device_0 = "/gpu:0"
      self._device_1 = "/gpu:1"

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self._hps
    # Input documents
    self._docs = tf.placeholder(
        tf.int32, [hps.batch_size, hps.max_num_doc, hps.max_doc_len],
        name="docs")  # [B, N, L]
    self._doc_lens = tf.placeholder(
        tf.int32, [hps.batch_size, hps.max_num_doc], name="doc_lens")  #[B, N]
    self._num_docs = tf.placeholder(tf.int32, [hps.batch_size], name="num_docs")

    # Query-related
    self._query_type = tf.placeholder(
        tf.int32, [hps.batch_size], name="query_type")
    self._query_subject = tf.placeholder(
        tf.int32, [hps.batch_size, hps.max_entity_len],
        name="query_subject")  #[B, EL]
    self._query_sub_lens = tf.placeholder(
        tf.int32, [hps.batch_size], name="query_sub_lens")

    # Candidate-related
    self._candidates = tf.placeholder(
        tf.int32, [hps.batch_size, hps.max_num_cands, hps.max_entity_len],
        name="candidates")  # [B, C, EL]
    self._cand_lens = tf.placeholder(
        tf.int32, [hps.batch_size, hps.max_num_cands], name="cand_lens")
    self._num_cand = tf.placeholder(tf.int32, [hps.batch_size], name="num_cand")

    # Answer
    self._answers = tf.placeholder(tf.int32, [hps.batch_size], name="answers")

  def _add_embeddings(self):
    hps = self._hps
    input_vsize = self._input_vocab.NumIds
    type_vsize = self._type_vocab.NumIds

    with tf.device(self._device_0):
      # Input word embeddings
      self._input_embed = tf.get_variable(
          "input_embed", [input_vsize, hps.emb_dim],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4))
      # Query type embeddings
      self._type_embed = tf.get_variable(
          "type_embed", [type_vsize, hps.type_emb_dim],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4))

  def _build_model(self):
    hps = self._hps

    with tf.variable_scope("hhn") as self._vs:
      with tf.variable_scope("embeddings"):
        self._add_embeddings()

      # Document encoder
      with tf.variable_scope("encoder",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_0):
        # Embed the docs
        emb_docs = tf.nn.embedding_lookup(self._input_embed,
                                          self._docs)  # [B, N, L, D]
        # Masking the paddings at document level
        doc_masks = tf.expand_dims(
            tf.expand_dims(
                tf.sequence_mask(
                    self._num_docs, maxlen=hps.max_num_doc, dtype=tf.float32),
                2), 3)  # [B, N, 1, 1]
        emb_docs *= doc_masks  # [B, N, L, D]
        emb_docs_rsp = tf.reshape(emb_docs, [-1, hps.max_doc_len,
                                             hps.emb_dim])  # [B*N, L, D]

        # Masking the paddings at word level
        doc_lens_rsp = tf.reshape(self._doc_lens, [-1])  # [B*N]
        word_masks = tf.expand_dims(
            tf.sequence_mask(
                doc_lens_rsp, maxlen=hps.max_doc_len, dtype=tf.float32),
            2)  # [B*N, L, 1]
        emb_docs_rsp *= word_masks  # apply the masks, [B*N, L, D]

        # Add the word-level CNN for documents
        docs_conv = tf.layers.conv1d(
            emb_docs_rsp,
            hps.word_conv_filter,
            hps.word_conv_width,
            padding="same",
            name="docs_conv",
            reuse=None,
            kernel_initializer=tf.random_uniform_initializer(
                -0.1, 0.1))  # [B*N, L, F]

        # Use the same CNN to model query subject
        emb_subject = tf.nn.embedding_lookup(self._input_embed,
                                             self._query_subject)  # [B, EL, D]
        # Mask the subject and calculate mean-pooled representation
        subject_lens = self._query_sub_lens  # [B]
        subject_mask = tf.expand_dims(
            tf.sequence_mask(
                subject_lens, maxlen=hps.max_entity_len, dtype=tf.float32),
            2)  # [B, EL, 1]
        emb_subject *= subject_mask

        subject_conv = tf.layers.conv1d(
            emb_subject,
            hps.word_conv_filter,
            hps.word_conv_width,
            padding="same",
            name="docs_conv",
            reuse=True)  # [B, EL, F]

        mean_subject = tf.reduce_sum(subject_conv, 1) / tf.expand_dims(
            tf.to_float(subject_lens) + 1e-3, 1)  # [B, F]

        # Embed the query type
        emb_query_type = tf.nn.embedding_lookup(self._type_embed,
                                                self._query_type)  # [B, TD]

        # Model the candidates
        emb_cands = tf.nn.embedding_lookup(self._input_embed,
                                           self._candidates)  # [B, C, EL, D]
        # Masking the paddings at candiadte level
        cands_masks = tf.expand_dims(
            tf.expand_dims(
                tf.sequence_mask(
                    self._num_cand, maxlen=hps.max_num_cands, dtype=tf.float32),
                2), 3)  # [B, C, 1, 1]
        emb_cands *= cands_masks  # [B, C, EL, D]
        emb_cands_rsp = tf.reshape(
            emb_cands, [-1, hps.max_entity_len, hps.emb_dim])  # [B*C, EL, D]
        # Masking the paddings at word level
        cand_lens_rsp = tf.reshape(self._cand_lens, [-1])  # [B*C]
        cand_word_masks = tf.expand_dims(
            tf.sequence_mask(
                cand_lens_rsp, maxlen=hps.max_entity_len, dtype=tf.float32),
            2)  # [B*C, EL, 1]
        emb_cands_rsp *= cand_word_masks  # [B*C, EL, D]

        # Pass the candidates through CNN and mean-pooling
        cands_conv = tf.layers.conv1d(
            emb_cands_rsp,
            hps.word_conv_filter,
            hps.word_conv_width,
            padding="same",
            name="docs_conv",
            reuse=True)  # [B*C, EL, F]
        mean_cands_rsp = tf.reduce_sum(cands_conv, 1) / tf.expand_dims(
            tf.to_float(cand_lens_rsp) + 1e-3, 1)  # [B*C, F]
        mean_cands = tf.reshape(mean_cands_rsp,
                                [-1, hps.max_num_cands,
                                 hps.word_conv_filter])  # [B, C, F]


      with tf.variable_scope("select_net",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_0):
        # Linear tranform for each word
        docs_conv_2 = tf.layers.conv1d(
            docs_conv,
            hps.word_conv_filter,
            1,
            padding="same",
            name="select_net_conv",
            reuse=None,
            kernel_initializer=tf.random_uniform_initializer(
                -0.1, 0.1))  # [B*N, L, F]

        # Introduce bilinear and scaled multiplicative attention
        mapped_docs = docs_conv_2 + docs_conv / math.sqrt(hps.word_conv_filter)
        mapped_docs_rsp = tf.reshape(mapped_docs, [
            -1, hps.max_num_doc, hps.max_doc_len, hps.word_conv_filter
        ])  # [B, N, L, F]
        mean_subject_rsp = tf.expand_dims(tf.expand_dims(mean_subject, 1),
                                          2)  # [B, 1, 1, F]
        select_att_logit = tf.reduce_sum(mapped_docs_rsp * mean_subject_rsp,
                                         3)  # [B, N, L]
        select_att_masks = tf.reshape(
            word_masks, [-1, hps.max_num_doc, hps.max_doc_len])  # [B, N, L]
        select_att_softmax = lib.masked_softmax(
            select_att_logit, select_att_masks, axis=-1)  # [B, N, L]

      with tf.variable_scope("hop_net",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_1):
        # Concat the query into document modeling in HopNet RNN
        query_concat = tf.concat([mean_subject, emb_query_type], 1)  # [B, TD+F]
        query_rsp = tf.expand_dims(tf.expand_dims(query_concat, 1),
                                   2)  # [B, 1, 1, TD+F]
        query_tiled = tf.tile(
            query_rsp, [1, hps.max_num_doc, hps.max_doc_len, 1])  # [B,N,L,TD+F]
        query_tiled_rsp = tf.reshape(query_tiled, [
            -1, hps.max_doc_len, hps.type_emb_dim + hps.word_conv_filter
        ])  # [B*N, L, TD+F]

        hop_net_rnn_input = tf.concat([emb_docs_rsp, query_tiled_rsp],
                                      2)  # [B*N, L, D+TD+F]

        # Run the RNN
        hop_net_rnn_output, _ = lib.cudnn_rnn_wrapper(
            hop_net_rnn_input,
            "gru",
            hps.hop_net_rnn_layers,
            hps.hop_net_rnn_num_hid,
            hps.emb_dim + hps.type_emb_dim + hps.word_conv_filter,
            "hop_net_gru_var",
            direction="bidirectional",
            dropout=hps.dropout)  # [L, B*N, H*2]

        # Linear tranform for each word
        hop_net_doc_linear = tf.layers.conv1d(
            hop_net_rnn_output,
            hps.word_conv_filter,
            1,
            padding="same",
            name="hop_net_att_bilinear",
            reuse=None,
            kernel_initializer=tf.random_uniform_initializer(
                -0.1, 0.1))  # [L, B*N, F]
        hop_net_doc_t = tf.transpose(hop_net_doc_linear, [1, 2,
                                                          0])  # [B*N, F, L]
        hop_net_att_logit = tf.matmul(docs_conv, hop_net_doc_t)  # [B*N, L, L]
        column_masks = tf.transpose(word_masks, [0, 2, 1])  # [B*N, 1, L]
        row_masks = word_masks  # [B*N, L, 1]

        hop_net_att_softmax = lib.masked_softmax(
            hop_net_att_logit, column_masks, axis=-1)  # [B*N, L, L]
        hop_net_att_softmax *= row_masks  # [B*N, L, L]

      # Apply hierarchical attention over the documents
      with tf.variable_scope("hier_att",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_0):
        # Compute the word-level attention after hopping
        select_att_rsp = tf.expand_dims(select_att_softmax, 2)  # [B, N, 1, L]
        hop_net_att_rsp = tf.reshape(hop_net_att_softmax, [
            -1, hps.max_num_doc, hps.max_doc_len, hps.max_doc_len
        ])  # [B, N, L, L]
        hop_att_weight = tf.squeeze(
            tf.matmul(select_att_rsp, hop_net_att_rsp), 2)  # [B, N, L], masked

        # Compute the weighted document representation
        hop_att_weight_rsp = tf.reshape(hop_att_weight,
                                        [-1, hps.max_doc_len, 1])  # [B*N, L, 1]
        weighted_docs = docs_conv * hop_att_weight_rsp  # [B*N, L, F], masked
        sum_docs = tf.reduce_sum(weighted_docs, 1)  # [B*N, F], masked
        sum_docs_rsp = tf.reshape(sum_docs,
                                  [-1, hps.max_num_doc,
                                   hps.word_conv_filter])  # [B, N, F], masked

        # Weight the doc representation with document-level attention
        doc_att_logit = tf.reduce_sum(select_att_logit * select_att_masks,
                                      2)  # [B, N]
        doc_att_mask = tf.sequence_mask(
            self._num_docs, maxlen=hps.max_num_doc, dtype=tf.float32)  # [B, N]
        doc_att_weight = lib.masked_softmax(doc_att_logit,
                                            doc_att_mask)  # [B, N], masked
        doc_att_weight_rsp = tf.expand_dims(doc_att_weight, 2)  # [B, N, 1]
        final_output = tf.reduce_sum(sum_docs_rsp * doc_att_weight_rsp,
                                     1)  # [B, F], masked

      with tf.variable_scope("prob_output"), tf.device(self._device_0):
        # Compute the probability of each candidate by inner product
        cand_prob_logits = tf.matmul(mean_cands, tf.expand_dims(
            final_output, 2))  # [B, C, 1]
        cand_prob_masks = tf.sequence_mask(
            self._num_cand, maxlen=hps.max_num_cands, dtype=tf.float32)
        self._cand_prob_logits = tf.squeeze(cand_prob_logits,
                                            2) * cand_prob_masks  # [B, C]

  def _add_loss(self):
    hps = self._hps

    with tf.variable_scope("loss"), tf.device(self._device_0):
      # Masking the loss
      # loss_mask = tf.sequence_mask(
      #     self._num_cand, maxlen=hps.max_num_cands, dtype=tf.float32)  # [B, C]

      xe_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self._answers,
          logits=self._cand_prob_logits,
          name="cand_xe_loss")  # [B]

      # batch_loss = tf.reduce_sum(xe_loss * loss_mask, 1)
      loss = tf.reduce_mean(xe_loss)
      tf.summary.scalar("loss", loss)

      # Calculate the accuracy
      prediction = tf.argmax(
          self._cand_prob_logits, axis=1, output_type=tf.int32)
      equality = tf.equal(prediction, self._answers)
      accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
      tf.summary.scalar("accuracy", accuracy)

    self._loss = loss
    self._accuracy = accuracy

  def _add_train_op(self):
    """Sets self._train_op for training."""
    hps = self._hps

    self._lr_rate = tf.maximum(
        hps.min_lr,  # minimum learning rate.
        tf.train.exponential_decay(hps.lr, self.global_step, hps.decay_step,
                                   hps.decay_rate))
    tf.summary.scalar("learning_rate", self._lr_rate)

    tvars = tf.trainable_variables()
    with tf.device(self._device_0):
      # Compute gradients
      grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self._loss, tvars), hps.max_grad_norm)
      tf.summary.scalar("global_norm", global_norm)

      # Create optimizer and train ops
      optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
      self._train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=self.global_step, name="train_step")

  def run_train_step(self, sess, batch):
    # pdb.set_trace()
    (doc_batch, doc_lens_batch, num_doc_batch, q_type_ids, q_subject_batch,
     q_subject_lens, candidates_batch, cand_lens_batch, num_cand_batch,
     answer_idxs, others) = batch

    to_return = [
        self._train_op, self._summaries, self._loss, self._accuracy,
        self.global_step
    ]

    results = sess.run(
        to_return,
        feed_dict={
            self._docs: doc_batch,
            self._doc_lens: doc_lens_batch,
            self._num_docs: num_doc_batch,
            self._query_type: q_type_ids,
            self._query_subject: q_subject_batch,
            self._query_sub_lens: q_subject_lens,
            self._candidates: candidates_batch,
            self._cand_lens: cand_lens_batch,
            self._num_cand: num_cand_batch,
            self._answers: answer_idxs
        })

    return results[1:]

  def run_eval_step(self, sess, batch):
    # pdb.set_trace()
    (doc_batch, doc_lens_batch, num_doc_batch, q_type_ids, q_subject_batch,
     q_subject_lens, candidates_batch, cand_lens_batch, num_cand_batch,
     answer_idxs, others) = batch

    to_return = [self._loss, self._accuracy]

    results = sess.run(
        to_return,
        feed_dict={
            self._docs: doc_batch,
            self._doc_lens: doc_lens_batch,
            self._num_docs: num_doc_batch,
            self._query_type: q_type_ids,
            self._query_subject: q_subject_batch,
            self._query_sub_lens: q_subject_lens,
            self._candidates: candidates_batch,
            self._cand_lens: cand_lens_batch,
            self._num_cand: num_cand_batch,
            self._answers: answer_idxs
        })

    return results

  def train_loop(self, sess, batcher, valid_batcher, summary_writer):
    """Runs model training."""
    step, losses, accuracies = 0, [], []
    while step < FLAGS.max_run_steps:
      next_batch = batcher.next()
      summaries, loss, accuracy, train_step = self.run_train_step(
          sess, next_batch)

      losses.append(loss)
      accuracies.append(accuracy)
      summary_writer.add_summary(summaries, train_step)
      step += 1

      # Display current training loss
      if step % FLAGS.display_freq == 0:
        avg_loss = lib.compute_avg(losses, summary_writer, "avg_loss",
                                   train_step)
        avg_acc = lib.compute_avg(accuracies, summary_writer, "avg_acc",
                                  train_step)
        tf.logging.info("Train step %d: avg_loss %f avg_acc %f" %
                        (train_step, avg_loss, avg_acc))
        losses, accuracies = [], []
        summary_writer.flush()

      # Run evaluation on validation set
      if step % FLAGS.valid_freq == 0:
        valid_losses, valid_accuracies = [], []
        for _ in xrange(FLAGS.num_valid_batch):
          next_batch = valid_batcher.next()
          valid_loss, valid_acc = self.run_eval_step(sess, next_batch)
          valid_losses.append(valid_loss)
          valid_accuracies.append(valid_acc)

        gstep = self.get_global_step(sess)
        avg_valid_loss = lib.compute_avg(valid_losses, summary_writer,
                                         "valid_loss", gstep)
        avg_valid_acc = lib.compute_avg(valid_accuracies, summary_writer,
                                        "valid_acc", gstep)
        tf.logging.info("\tValid step %d: avg_loss %f avg_acc %f" %
                        (gstep, avg_valid_loss, avg_valid_acc))

        summary_writer.flush()

  def train(self, data_batcher, valid_batcher):
    """Runs model training."""
    hps = self._hps
    assert hps.mode == "train", "This method is only callable in train mode."

    with tf.device("/gpu:0"):  # GPU by default
      self.build_graph()

    # Restore pretrained model if necessary
    load_pretrain = None

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    sv = tf.train.Supervisor(
        logdir=FLAGS.ckpt_root,
        saver=saver,
        summary_op=None,
        save_summaries_secs=FLAGS.checkpoint_secs,
        save_model_secs=FLAGS.checkpoint_secs,
        global_step=self.global_step,
        init_fn=load_pretrain)  # TODO: could exploit more Supervisor features

    config = tf.ConfigProto(allow_soft_placement=True)
    # Turn on JIT compilation if necessary
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = sv.prepare_or_wait_for_session(config=config)

    # Summary dir is different from ckpt_root to avoid conflict.
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

    # Start the training loop
    self.train_loop(sess, data_batcher, valid_batcher, summary_writer)

    sv.Stop()

  def get_global_step(self, sess):
    """Get the current number of training steps."""
    return sess.run(self.global_step)
