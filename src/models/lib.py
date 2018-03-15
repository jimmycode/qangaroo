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
# ============================================================================
"""seq2seq library codes copied from elsewhere for customization."""

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
import re


def compute_avg(values, summary_writer, tag_name, step):
  """Calculate the average of losses and write to summary."""
  avg_value = float(np.mean(values))
  avg_summary = tf.Summary()
  avg_summary.value.add(tag=tag_name, simple_value=avg_value)
  summary_writer.add_summary(avg_summary, step)
  return avg_value


def parse_list_str(list_str):
  l = [int(x) for x in re.split("[\[\]\s,]", list_str) if x]
  if not l:
    raise ValueError("List is empty.")
  return l


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or tf.get_variable_scope()) as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return tf.nn.bias_add(res, biases)


def cudnn_rnn_wrapper(input_data,
                      rnn_mode,
                      num_layers,
                      num_units,
                      input_size,
                      variable_name,
                      direction="unidirectional",
                      time_major=False,
                      dropout=0.0):

  if rnn_mode == "lstm":
    model = cudnn_rnn_ops.CudnnLSTM(
        num_layers, num_units, input_size, direction=direction, dropout=dropout)
  elif rnn_mode == "gru":
    model = cudnn_rnn_ops.CudnnGRU(
        num_layers, num_units, input_size, direction=direction, dropout=dropout)
  else:
    raise ValueError("Invalid rnn_mode: %s" % rnn_mode)

  # Compute the total size of RNN params (Tensor)
  params_size_ts = model.params_size()
  params = tf.Variable(
      tf.random_uniform([params_size_ts], minval=-0.1, maxval=0.1),
      validate_shape=False,
      name=variable_name)

  if not time_major:
    batch_size_ts = tf.shape(input_data)[0]  # batch size Tensor
    input_data = tf.transpose(input_data, [1, 0, 2])
  else:
    batch_size_ts = tf.shape(input_data)[1]  # batch size Tensor
  # NB: input_data should has shape [batch_size, num_timestep, d]

  if direction == "unidirectional":
    dir_count = 1
  elif direction == "bidirectional":
    dir_count = 2
  else:
    raise ValueError("Invalid direction: %s" % direction)

  init_h = tf.zeros(
      tf.stack([num_layers * dir_count, batch_size_ts, num_units]))
  has_input_c = (rnn_mode == "lstm")

  # Call the CudnnRNN
  if has_input_c:
    init_c = tf.zeros(
        tf.stack([num_layers * dir_count, batch_size_ts, num_units]))
    output, output_h, output_c = model(
        input_data=input_data, input_h=init_h, input_c=init_c, params=params)
  else:
    output, output_h = model(
        input_data=input_data, input_h=init_h, params=params)

  # output: [num_timestep, batch_size, num_units*dir_count]
  # output_h/c: [batch_size, num_units*dir_count]
  return output, output_h


def masked_softmax(logits, masks, axis=-1):
  """Compute softmax along the specified axis with masks applied."""
  probs = tf.nn.softmax(logits * masks, dim=axis)  # [?, C]
  #NOTE: v1.3 uses dim, v1.5+ uses axis
  masked_probs = probs * masks  # [?, C]
  sum_masked_probs = tf.reduce_sum(
      masked_probs, axis=axis, keep_dims=True) + 1e-7  # [?, 1], avoid NaN
  #NOTE: v1.3 uses keep_dims, v1.5+ uses keepdims
  final_probs = masked_probs / sum_masked_probs  # [?, C]
  return final_probs
