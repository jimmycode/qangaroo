import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
import json
from nltk.tokenize import word_tokenize
import operator
import os.path as osp


def update_vocab(vocab, word):
  if word in vocab:
    vocab[word] += 1
  else:
    vocab[word] = 1


def sort_vocab(vocab):
  return sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)


def write_vocab(vocab, path):
  # pdb.set_trace()
  with open(path, 'w') as f:
    for word, freq in vocab:
      s = '%s %d\n' % (word, freq)
      f.write(s.encode('utf-8'))
  print 'Finished writing %s.' % path


special_symbols = ['<EOD>', '<OOV>', '<PAD>']


def merge(file_A, file_B, file_C, output_file):
  vocabs = []
  for fn in [file_A, file_B, file_C]:
    v = {}
    with open(fn, 'r') as f:
      for line in f:
        token, freq = line.split()
        v[token] = int(freq)
    vocabs.append(v)

  vocab_A, vocab_B, vocab_C = vocabs

  # Merge B and C
  for k in vocab_C:
    if k in vocab_B:
      vocab_B[k] += vocab_C[k]
    else:
      vocab_B[k] = vocab_C[k]

  # Merge A into B+C
  rest_A = []
  for k in vocab_A:
    if k in vocab_B:
      vocab_B[k] += vocab_A[k]
    else:
      rest_A.append((k, vocab_A[k]))
  rest_A = sorted(rest_A, key=operator.itemgetter(1), reverse=True)

  # Write to file
  with open(output_file, 'w') as f:
    for w in special_symbols:
      f.write(w + " 0\n")

    vocab_B_list = sort_vocab(vocab_B)
    for word, freq in vocab_B_list:
      s = '%s %d\n' % (word, freq)
      f.write(s)
    print "Subject + candidate vocab: %s" % len(vocab_B_list)

    for word, freq in rest_A:
      s = '%s %d\n' % (word, freq)
      f.write(s)

  print 'Finished writing to %s.' % output_file


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Merge file B and C into A.')
  parser.add_argument('-A', default='A', help='file A')
  parser.add_argument('-B', default='B', help='file B')
  parser.add_argument('-C', default='C', help='file C')
  parser.add_argument('-O', default='O', help='output file')
  # parser.add_argument('-L', default=200000, help='limit')

  args = parser.parse_args()
  merge(args.A, args.B, args.C, args.O)
