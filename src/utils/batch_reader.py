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
"""Batch reader to seq2seq attention model, with bucketing support."""

import Queue
import random
from random import shuffle, randint
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import glob
# import cPickle as pkl
import json
from collections import namedtuple
import pdb
FLAGS = tf.app.flags.FLAGS

QAngarooSample = namedtuple('QAngarooSample', 'doc_input, doc_lens, num_doc,'
                            'q_type_id, q_subject, q_subject_len,'
                            'candidates, cand_lens, num_candidate, answer_idx,'
                            'origin_doc, origin_query, origin_candidates')
QAngarooBatch = namedtuple('QAngarooBatch',
                           'doc_batch, doc_lens_batch, num_doc_batch,'
                           'q_type_ids, q_subject_batch, q_subject_lens, '
                           'candidates_batch, cand_lens_batch, num_cand_batch,'
                           'answer_idxs, others')
# others=(origin_doc origin_candidate origin_type)

QUEUE_NUM_BATCH = 100  # Number of batches kept in the queue
BUCKET_NUM_BATCH = 20  # Number of batches per bucketing iteration fetches
GET_TIMEOUT = 240


class QAngarooBatcher(object):
  """Batch reader for extractive summarization data."""

  def __init__(self,
               data_path,
               enc_vocab,
               type_vocab,
               hps,
               bucketing=False,
               truncate_input=True,
               num_epochs=None,
               shuffle_batches=True):
    """Batcher constructor.

    Args:
      data_path: tf.Example filepattern.
      enc_vocab: Encoder vocabulary.
      type_vocab: query type vocabulary.
      hps: model hyperparameters.
      bucketing: Whether bucket inputs of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
      shuffle_batches: True if the examples would be randomly shuffled.
    """
    if not data_path:
      raise ValueError("Data path must be specified.")
    self._data_path = data_path
    self._enc_vocab = enc_vocab
    self._type_vocab = type_vocab
    self._hps = hps
    self._bucketing = bucketing  # deprecated
    self._truncate_input = truncate_input
    assert self._truncate_input == True, 'truncate=False not implemented.'

    self._num_epochs = num_epochs
    self._shuffle_batches = shuffle_batches

    self._sample_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
    self._batch_queue = Queue.Queue(QUEUE_NUM_BATCH)

    # Create input reading threads
    fn = self._data_path
    self._input_thread = Thread(target=self._FillInputQueue, args=(fn,))
    self._input_thread.daemon = True
    self._input_thread.start()

    # Create bucketing threads
    self._bucketing_threads = []
    for _ in xrange(2):
      self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()

    # Create watch threads
    if self._hps.mode == 'train':
      # Keep input threads running in train mode,
      # but they are not needed in eval and decode mode.
      self._watch_thread = Thread(target=self._WatchThreads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

  def __iter__(self):
    return self

  def next(self):
    """Returns a batch of inputs for seq2seq attention model.

    Returns:
        batch: a AbsModelBatch object.
    """
    try:
      batch = self._batch_queue.get(timeout=GET_TIMEOUT)
    except Queue.Empty as e:
      raise StopIteration('batch_queue.get() timeout: %s' % e)

    return batch

  def _FillInputQueue(self, data_path):
    """Fill input queue with QAngarooSample."""
    hps = self._hps
    enc_vocab = self._enc_vocab
    type_vocab = self._type_vocab

    enc_pad_id = enc_vocab.pad_id
    enc_empty_doc = [enc_pad_id] * hps.max_doc_len
    empty_cand = [enc_pad_id] * hps.max_entity_len

    data_generator = self._DataGenerator(data_path, self._num_epochs)

    # pdb.set_trace()
    for data_sample in data_generator:
      supports = data_sample['supports']
      enc_input = []
      for s in supports:
        enc_input.append([enc_vocab.WordToId(w) for w in s])

      # Truncate the documents
      trunc_enc_input = [
          d[:hps.max_doc_len] for d in enc_input[:hps.max_num_doc]
      ]

      # Calculate the length statistics
      enc_doc_lens = [len(s) for s in trunc_enc_input]
      enc_num_doc = len(trunc_enc_input)

      # Pad trunc_enc_input if necessary
      padded_enc_input = [
          s + [enc_pad_id] * (hps.max_doc_len - l)
          for s, l in zip(trunc_enc_input, enc_doc_lens)
      ]
      padded_enc_input += [enc_empty_doc] * (hps.max_num_doc - enc_num_doc)
      np_enc_input = np.array(
          padded_enc_input, dtype=np.int32)  # [max_num_doc x max_doc_len]

      # Pad the lengths
      pad_doc_lens = enc_doc_lens + [0] * (hps.max_num_doc - enc_num_doc)
      np_doc_lens = np.array(pad_doc_lens, dtype=np.int32)

      # Get query type
      type_id = type_vocab.WordToId(data_sample['query_type'])

      # Get query subject and its length
      subject_words = data_sample['query_subject']
      subject_ids = [enc_vocab.WordToId(w) for w in subject_words]
      subject_ids = subject_ids[:hps.max_entity_len]  # truncate if too long
      subject_len = len(subject_ids)

      padded_sub = subject_ids + [enc_pad_id] * (
          hps.max_entity_len - subject_len)
      np_subject = np.array(padded_sub, dtype=np.int32)

      # Get candidates
      candidates = data_sample['candidates']
      cand_ids = []
      for c in candidates:
        cand_ids.append([enc_vocab.WordToId(w) for w in c])
      # Truncate the candiates
      trunc_cand_ids = [
          c[:hps.max_entity_len] for c in cand_ids[:hps.max_num_cands]
      ]
      cand_lens = [len(c) for c in trunc_cand_ids]
      num_cands = len(trunc_cand_ids)

      # Pad trunc_cand_ids if necessary
      padded_cands = [
          c + [enc_pad_id] * (hps.max_entity_len - l)
          for c, l in zip(trunc_cand_ids, cand_lens)
      ]
      padded_cands += [empty_cand] * (hps.max_num_cands - num_cands)
      np_cands = np.array(padded_cands, dtype=np.int32)

      padded_cand_lens = cand_lens + [0] * (hps.max_num_cands - num_cands)
      np_cand_lens = np.array(padded_cand_lens, dtype=np.int32)

      # Get the answer
      answer_idx = data_sample['answer_index']

      # Wrap with namedtuple
      element = QAngarooSample(np_enc_input, np_doc_lens, enc_num_doc, type_id,
                               np_subject, subject_len, np_cands, np_cand_lens,
                               num_cands, answer_idx, supports, (
                                   subject_words, type_id), candidates)
      self._sample_queue.put(element)

  def _DataGenerator(self, path, num_epochs=None):
    """An (infinite) iterator that outputs data samples."""
    epoch = 0
    with open(path, 'r') as f:
      dataset = json.load(f)

    while True:
      if num_epochs is not None and epoch >= num_epochs:
        return

      if self._shuffle_batches:
        shuffle(dataset)

      for d in dataset:
        yield d

      epoch += 1

  def _FillBucketInputQueue(self):
    """Fill bucketed batches into the bucket_input_queue."""
    hps = self._hps
    while True:
      samples = []
      for _ in xrange(hps.batch_size * BUCKET_NUM_BATCH):
        samples.append(self._sample_queue.get())

      # if self._bucketing:
      #   samples = sorted(samples, key=lambda inp: inp.enc_doc_len)

      batches = []
      for i in xrange(0, len(samples), hps.batch_size):
        batches.append(samples[i:i + hps.batch_size])

      if self._shuffle_batches:
        shuffle(batches)

      for b in batches:
        self._batch_queue.put(self._PackBatch(b))

  def _PackBatch(self, batch):
    """ Pack the batch into numpy arrays.

    Returns:
        model_batch: QAngarooBatch
    """
    field_lists = [[], [], [], [], [], [], [], [], [], []]
    origin_docs, origin_queries, origin_cands = [], [], []

    for ex in batch:
      for i in range(10):
        field_lists[i].append(ex[i])
      origin_docs.append(ex.origin_doc)
      origin_queries.append(ex.origin_query)
      origin_cands.append(ex.origin_candidates)

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return QAngarooBatch(
        stacked_fields[0], stacked_fields[1], stacked_fields[2],
        stacked_fields[3], stacked_fields[4], stacked_fields[5],
        stacked_fields[6], stacked_fields[7], stacked_fields[8],
        stacked_fields[9], (origin_docs, origin_queries, origin_cands))

  def _WatchThreads(self):
    """Watch the daemon input threads and restart if dead."""
    while True:
      time.sleep(60)

      if not self._input_thread.is_alive():
        tf.logging.error('Found input thread dead.')
        new_t = Thread(target=self._FillInputQueue, args=(self._data_path,))
        new_t.daemon = True
        new_t.start()
        self._input_thread = new_t

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._FillBucketInputQueue)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()
      self._bucketing_threads = bucketing_threads
