import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
import json
from nltk.tokenize import word_tokenize
import operator


def update_vocab(vocab, word):
  if word in vocab:
    vocab[word] += 1
  else:
    vocab[word] = 1


def sort_vocab(vocab):
  return sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)


def write_vocab(vocab, path):
  with open(path, 'w') as f:
    for word, freq in vocab:
      f.write('%s %d\n' % (word, freq))
  print 'Finished writing %s.' % path


def plot_hist(count, title):
  plt.hist(count, bins='auto')
  plt.title(title)
  plt.show()


def preprocess(input_file):
  doc_len_count, num_doc_count, num_cand_count, cand_len_count = [], [], [], []
  doc_vocab, query_type_vocab, query_subject_vocab, candidate_vocab = {}, {}, {}, {}

  with open(input_file, 'r') as f:
    data = json.load(f)

  new_data = []

  for d in data:
    new_d = {}
    new_d['id'] = d['id']

    # Process the supports
    new_supports = []
    for s in d['supports']:
      tokens = word_tokenize(s)
      lower_tokens = [t.lower() for t in tokens]

      doc_len_count.append(len(lower_tokens))
      new_supports.append(lower_tokens)

      for t in lower_tokens:
        update_vocab(doc_vocab, t)

    new_d['supports'] = new_supports  # list of list

    num_doc_count.append(len(new_supports))

    # Process the query
    query_str = d['query']
    query_type = query_str.split()[0]
    query_subject = query_str.split()[1:]
    new_d['query_type'] = query_type  # str
    update_vocab(query_type_vocab, query_type)

    new_d['query_subject'] = query_subject  # list
    for t in query_subject:
      update_vocab(query_subject_vocab, t)

    # Process the candidates and the answer
    candidates = d['candidates']
    answer = d['answer']
    try:
      new_d['answer_index'] = candidates.index(answer)
    except ValueError:
      print "Index error:", candidates, answer
      continue

    new_candidates = []
    for c in candidates:
      cand_tokens = c.split()
      for t in cand_tokens:
        update_vocab(candidate_vocab, t)

      cand_len_count.append(len(cand_tokens))
      new_candidates.append(cand_tokens)
    num_cand_count.append(len(new_candidates))
    new_d['candidates'] = new_candidates  # list of list

    new_data.append(new_d)

  with open(input_file + '.proc', 'w') as f:
    json.dump(new_data, f, indent=1)
    print "Finish writing to %s." % (input_file + '.proc')

  sort_vocab(doc_vocab)
  write_vocab(doc_vocab, input_file + '.doc.vocab')

  sort_vocab(query_type_vocab)
  write_vocab(query_type_vocab, input_file + '.type.vocab')

  sort_vocab(query_subject_vocab)
  write_vocab(query_subject_vocab, input_file + '.sub.vocab')

  sort_vocab(candidate_vocab)
  write_vocab(candidate_vocab, input_file + '.cand.vocab')
  print "Finish writing vocab files."

  for var in [doc_len_count, num_doc_count, num_cand_count, cand_len_count]:
    print 'Stats: max %d min %d mean %f std %f' % (max(var), min(var),
                                                   np.mean(var), np.std(var))
    plot_hist(input_count, 'Count')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=
      'Preprocess the dataset, build vocabs, and calculate statistics.')
  parser.add_argument(
      '--in_file',
      default='data/wikihop/train.json',
      help='path to input data file')

  args = parser.parse_args()
  preprocess(args.in_file)
