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
from hhn import HierHopNet

FLAGS = tf.app.flags.FLAGS

# NB: batch_size could be unspecified (None) in decode mode
HParams = namedtuple("HParams", "mode, lr, min_lr, dropout, batch_size,"
                     "max_num_doc, max_doc_len, max_entity_len,"
                     "max_num_cands, emb_dim, type_emb_dim, word_conv_filter,"
                     "word_conv_width, hop_net_rnn_layers, hop_net_rnn_num_hid,"
                     "max_grad_norm, decay_step, decay_rate,"
                     "num_hops, hop_mod_reuse")


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
      decay_rate=FLAGS.decay_rate,
      num_hops=FLAGS.num_hops,  # max number of hops
      hop_mod_reuse=FLAGS.hop_mod_reuse)  # whether reuse hop module parameters
  return hps


class MultiHierHopNet(HierHopNet):
  """ Hierarchical Hopping Network with multiple hops. """

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
        # docs_conv = tf.tanh(docs_conv)

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

      self._multi_hop_vs = []
      # Create multi-layer hopping network
      with tf.variable_scope("multi_hop",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_0):
        query_subjects, query_types = [], []

        query_subject, query_type = mean_subject, emb_query_type
        for i in range(hps.num_hops):
          # Not reusing parameters
          with tf.variable_scope("hop_module_%d" % i) as vs:
            query_subject, query_type = self._add_hop_module(
                docs_conv, word_masks, query_subject, query_type)

          # Record the variable scope of each hop module
          self._multi_hop_vs.append(vs)

          query_subjects.append(query_subject)
          query_types.append(query_type)

        q_sub_sum = tf.add_n(query_subjects)
        q_type_sum = tf.add_n(query_types)

        # Use an MLP to integrate all the query outputs
        mlp_output_1 = tf.nn.relu(
            lib.linear(
                [q_sub_sum, q_type_sum],
                hps.word_conv_filter,
                True,
                scope="final_mlp_1"))
        mlp_output_2 = tf.nn.relu(
            lib.linear(
                mlp_output_1, hps.word_conv_filter, True,
                scope="final_mlp_2"))  # [B, F]
        final_output = tf.nn.dropout(mlp_output_2, 1 - hps.dropout)

      with tf.variable_scope("prob_output"), tf.device(self._device_0):
        # Compute the probability of each candidate by inner product
        cand_prob_logits = tf.matmul(mean_cands, tf.expand_dims(
            final_output, 2))  # [B, C, 1]
        cand_prob_masks = tf.sequence_mask(
            self._num_cand, maxlen=hps.max_num_cands, dtype=tf.float32)
        self._cand_prob_logits = tf.squeeze(cand_prob_logits,
                                            2) * cand_prob_masks  # [B, C]

  def _add_hop_module(self,
                      docs,
                      doc_word_masks,
                      query_subject,
                      query_type,
                      scope=""):
    """Create a hopping module.

    Args:
      docs: a float32 Tensor with shape [B*N, L, D1].
      doc_word_masks: a float32 Tensor with shape [B*N, L, 1].
      query_subject: a float32 Tensor with shape [B, D1].
      query_type: a float32 Tensor with shape [B, D2].
      scope: str, scope name

    Returns:
      next_q_subject: a float32 Tensor with shape [B, D1].
      next_q_type: a float32 Tensor with shape [B, D2].
    """
    hps = self._hps
    with tf.variable_scope("select_net%s" % scope,
        initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
        tf.device(self._device_0):
      # Linear tranform for each word
      docs_conv_2 = tf.layers.conv1d(
          docs,
          hps.word_conv_filter,
          1,
          padding="same",
          name="select_net_conv",
          reuse=None,
          kernel_initializer=tf.random_uniform_initializer(-0.1,
                                                           0.1))  # [B*N, L, F]

      # Introduce bilinear and scaled multiplicative attention
      mapped_docs = docs_conv_2 + docs / math.sqrt(hps.word_conv_filter)
      mapped_docs_rsp = tf.reshape(mapped_docs, [
          -1, hps.max_num_doc, hps.max_doc_len, hps.word_conv_filter
      ])  # [B, N, L, F]
      mean_subject_rsp = tf.expand_dims(tf.expand_dims(query_subject, 1),
                                        2)  # [B, 1, 1, F]
      select_att_logit = tf.reduce_sum(mapped_docs_rsp * mean_subject_rsp,
                                       3)  # [B, N, L]
      select_att_masks = tf.reshape(
          doc_word_masks, [-1, hps.max_num_doc, hps.max_doc_len])  # [B, N, L]
      select_att_softmax = lib.masked_softmax(
          select_att_logit, select_att_masks, axis=-1)  # [B, N, L]

    with tf.variable_scope("hop_net%s" % scope,
        initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
        tf.device(self._device_1):
      # Concat the query into document modeling in HopNet RNN
      query_concat = tf.concat([query_subject, query_type], 1)  # [B, TD+F]
      query_rsp = tf.expand_dims(tf.expand_dims(query_concat, 1),
                                 2)  # [B, 1, 1, TD+F]
      query_tiled = tf.tile(query_rsp, [1, hps.max_num_doc, hps.max_doc_len,
                                        1])  # [B, N, L, TD+F]
      query_tiled_rsp = tf.reshape(query_tiled, [
          -1, hps.max_doc_len, hps.type_emb_dim + hps.word_conv_filter
      ])  # [B*N, L, TD+F]

      hop_net_rnn_input = tf.concat([docs, query_tiled_rsp],
                                    2)  # [B*N, L, F+TD+F]

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
          kernel_initializer=tf.random_uniform_initializer(-0.1,
                                                           0.1))  # [L, B*N, F]
      hop_net_doc_t = tf.transpose(hop_net_doc_linear, [1, 2, 0])  # [B*N, F, L]
      hop_net_att_logit = tf.matmul(docs, hop_net_doc_t)  # [B*N, L, L]
      column_masks = tf.transpose(doc_word_masks, [0, 2, 1])  # [B*N, 1, L]
      row_masks = doc_word_masks  # [B*N, L, 1]

      hop_net_att_softmax = lib.masked_softmax(
          hop_net_att_logit, column_masks, axis=-1)  # [B*N, L, L]
      hop_net_att_softmax *= row_masks  # [B*N, L, L]

    # Apply hierarchical attention over the documents
    with tf.variable_scope("hier_att%s" % scope,
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
      hop_att_weight_rsp = tf.reshape(hop_att_weight, [-1, hps.max_doc_len,
                                                       1])  # [B*N, L, 1]
      weighted_docs = docs * hop_att_weight_rsp  # [B*N, L, F], masked
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
      next_q_subject = tf.reduce_sum(sum_docs_rsp * doc_att_weight_rsp,
                                     1)  # [B, F], masked

      # Produce the representation for next query type
      next_q_type_linear = lib.linear(
          [next_q_subject, query_subject, query_type],
          hps.type_emb_dim,
          True,
          scope="next_q_type_linear")
      next_q_type = tf.tanh(next_q_type_linear)

    return next_q_subject, next_q_type

  def train(self, data_batcher, valid_batcher):
    """Runs model training."""
    hps = self._hps
    assert hps.mode == "train", "This method is only callable in train mode."

    with tf.device("/gpu:0"):  # GPU by default
      self.build_graph()

    # Restore pretrained model if necessary
    if hps.hop_mod_reuse:
      all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      last_hop_vars = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=self._multi_hop_vs[-1].name)
      pretrain_vars = [v for v in all_vars if v not in last_hop_vars]
      restorer = tf.train.Saver(pretrain_vars)
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
      if not (ckpt_state and ckpt_state.model_checkpoint_path):
        restorer = None

      def load_pretrain(sess):
        restorer.restore(sess, ckpt_state.model_checkpoint_path)
        if restorer:
          restorer.restore(sess, ckpt_state.model_checkpoint_path)
    else:
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
