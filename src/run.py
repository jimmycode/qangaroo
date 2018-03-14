"""Script for training and testing models."""
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple

from utils import batch_reader, vocab, evaluate

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model", "", "The name of model runned.")
tf.app.flags.DEFINE_string("data_path", "", "Path expression to data file.")
tf.app.flags.DEFINE_string("input_vocab", "", "Path to input vocabulary file.")
tf.app.flags.DEFINE_string("type_vocab", "", "Path to type vocabulary file.")
tf.app.flags.DEFINE_integer("input_vsize", 0,
                            "Number of words in input vocabulary.")
tf.app.flags.DEFINE_integer("type_vsize", 0,
                            "Number of words in type vocabulary.")
tf.app.flags.DEFINE_string("ckpt_root", "", "Directory for checkpoint root.")
tf.app.flags.DEFINE_string("summary_dir", "", "Directory for summary files.")
tf.app.flags.DEFINE_string("mode", "train", "train/decode mode")
tf.app.flags.DEFINE_integer("batch_size", 16, "Size of minibatch.")
# ----------- Train mode related flags ------------------
tf.app.flags.DEFINE_float("lr", 0.15, "Initial learning rate.")
tf.app.flags.DEFINE_float("min_lr", 0.01, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_grad_norm", 1.0,
                          "Maximum gradient norm for gradient clipping.")
tf.app.flags.DEFINE_integer("decay_step", 10000, "Exponential decay step.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Exponential decay rate.")
tf.app.flags.DEFINE_integer("max_run_steps", 1000000,
                            "Maximum number of run steps.")
tf.app.flags.DEFINE_float("dropout", 0.0, "Dropout rate.")
tf.app.flags.DEFINE_string("valid_path", "",
                           "Path expression to validation set.")
tf.app.flags.DEFINE_integer("valid_freq", 500, "How often to run eval.")
tf.app.flags.DEFINE_integer("num_valid_batch", 20,
                            "Number valid batches in each _Valid step.")
tf.app.flags.DEFINE_integer("checkpoint_secs", 1200, "How often to checkpoint.")
tf.app.flags.DEFINE_integer("max_to_keep", None,
                            "Maximum number of checkpoints to keep. "
                            "Keep all by default")
tf.app.flags.DEFINE_integer("display_freq", 200, "How often to print.")
tf.app.flags.DEFINE_integer("verbosity", 20,
                            "tf.logging verbosity (default INFO).")
# ----------- Data reading related flags ------------------
tf.app.flags.DEFINE_bool("use_bucketing", False,
                         "Whether bucket articles of similar length.")
tf.app.flags.DEFINE_bool("truncate_input", True,
                         "Truncate inputs that are too long. If False, "
                         "examples that are too long are discarded.")
# ----------- general flags ----------------
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of gpus used.")
# ----------- model related flags ----------------
tf.app.flags.DEFINE_integer("emb_dim", 128, "Dim of word embedding.")
tf.app.flags.DEFINE_integer("type_emb_dim", 128, "Dim of type embedding.")
tf.app.flags.DEFINE_integer("max_num_doc", 64, "Maximum number of documents.")
tf.app.flags.DEFINE_integer("max_doc_len", 300,
                            "Maximum number of words in a document.")
tf.app.flags.DEFINE_integer("max_entity_len", 10,
                            "Maximum number of tokens in an entity.")
tf.app.flags.DEFINE_integer("max_num_cands", 70,
                            "Maximum number of candidates.")
tf.app.flags.DEFINE_integer("word_conv_filter", 128, "Number of CNN filters.")
tf.app.flags.DEFINE_integer("word_conv_width", 3, "Width of CNN kernel.")
tf.app.flags.DEFINE_integer("hop_net_rnn_layers", 1, "Number of layer of RNN.")
tf.app.flags.DEFINE_integer("hop_net_rnn_num_hid", 128,
                            "Number of hidden units of RNN.")


def main():
  # Configure the enviroment
  tf.logging.set_verbosity(FLAGS.verbosity)

  # Import model
  model_type = FLAGS.model
  if model_type == "hhn":
    from models.hhn import CreateHParams
    from models.hhn import HierHopNet as Model
  else:
    raise ValueError("%s model NOT defined." % model_type)
  tf.logging.info("Using model %s." % model_type.upper())

  # Build vocabs
  input_vocab = vocab.Vocab(FLAGS.input_vocab, FLAGS.input_vsize)
  type_vocab = vocab.Vocab(FLAGS.type_vocab, FLAGS.type_vsize)

  # Create model hyper-parameters
  hps = CreateHParams()
  tf.logging.info("Using the following hyper-parameters:\n%r" % str(hps))

  if FLAGS.mode == "train":
    num_epochs = None  # infinite loop
    shuffle_batches = True
    batcher_hps = hps
  else:
    num_epochs = 1  # only go through test set once
    shuffle_batches = False  # do not shuffle the batches
    batcher_hps = hps._replace(batch_size=1)  # ensure all examples are used
    batch_reader.BUCKET_NUM_BATCH = 1  # ensure all examples are used
    batch_reader.GET_TIMEOUT = 60

  # Create data reader
  if model_type == "hhn":
    batcher = batch_reader.QAngarooBatcher(
        FLAGS.data_path,
        input_vocab,
        type_vocab,
        batcher_hps,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)
    if FLAGS.mode == "train":
      # Create validation data reader
      valid_batcher = batch_reader.QAngarooBatcher(
          FLAGS.valid_path,
          input_vocab,
          type_vocab,
          batcher_hps,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)

  else:
    raise NotImplementedError()

  if FLAGS.mode == "train":
    model = Model(hps, input_vocab, type_vocab, num_gpus=FLAGS.num_gpus)
    model.train(batcher, valid_batcher)  # start training
  elif FLAGS.mode == "predict":
    raise NotImplementedError()
  else:
    raise ValueError("Invalid mode %s. Try train/predict instead." % FLAGS.mode)


if __name__ == "__main__":
  main()
