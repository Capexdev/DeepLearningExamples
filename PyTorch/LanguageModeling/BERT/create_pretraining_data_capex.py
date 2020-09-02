# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

"""Create masked LM/next sentence masked_lm TF examples for BERT."""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open
import h5py
import numpy as np
from tqdm import tqdm, trange

from tokenization import BertTokenizer
import tokenization as tokenization
import sentencepiece as spm

import random
import collections




class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_file(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_file):
  """Create TF example files from `TrainingInstance`s."""
 

  total_written = 0
  features = collections.OrderedDict()
 
  num_instances = len(instances)
  features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
  features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
  features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
  features["masked_lm_positions"] =  np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
  features["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
  features["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")


  for inst_index, instance in enumerate(tqdm(instances)):
    input_ids = instance.tokens #tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = instance.masked_lm_labels #tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    

    features["input_ids"][inst_index] = input_ids
    features["input_mask"][inst_index] = input_mask
    features["segment_ids"][inst_index] = segment_ids
    features["masked_lm_positions"][inst_index] = masked_lm_positions
    features["masked_lm_ids"][inst_index] = masked_lm_ids
    features["next_sentence_labels"][inst_index] = next_sentence_label

    total_written += 1

    # if inst_index < 20:
    #   tf.logging.info("*** Example ***")
    #   tf.logging.info("tokens: %s" % " ".join(
    #       [tokenization.printable_text(x) for x in instance.tokens]))

    #   for feature_name in features.keys():
    #     feature = features[feature_name]
    #     values = []
    #     if feature.int64_list.value:
    #       values = feature.int64_list.value
    #     elif feature.float_list.value:
    #       values = feature.float_list.value
    #     tf.logging.info(
    #         "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

 
  print("saving data")
  f= h5py.File(output_file, 'w')
  f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
  f.create_dataset("input_mask", data=features["input_mask"], dtype='i1', compression='gzip')
  f.create_dataset("segment_ids", data=features["segment_ids"], dtype='i1', compression='gzip')
  f.create_dataset("masked_lm_positions", data=features["masked_lm_positions"], dtype='i4', compression='gzip')
  f.create_dataset("masked_lm_ids", data=features["masked_lm_ids"], dtype='i4', compression='gzip')
  f.create_dataset("next_sentence_labels", data=features["next_sentence_labels"], dtype='i1', compression='gzip')
  f.flush()
  f.close()


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  all_documents = []
  for file in input_files:
    with open(file) as f:
      for line in f:
        parts = line.rstrip('\r\n').split('\t')
        x, y = parts
        all_documents.append((x, y))

  instances = []
  for document in all_documents:
    for i in range(dupe_factor):
      x = tokenizer.encode_as_ids(document[0])
      y = tokenizer.encode_as_ids(document[1])
      random_document = all_documents[random.randint(0, len(all_documents) - 1)]
      z = tokenizer.encode_as_ids(random_document[1])

      # True
      tokens = [CLS_ID] + x + [SEP_ID] + y + [SEP_ID]
      segment_ids = [0] * (len(x) + 2) + [1] * (len(y) + 1)
      is_random_next = False
      if len(tokens) < max_seq_length:
        (tokens, masked_lm_positions,
            masked_lm_labels) = create_masked_lm_predictions(
                tokens, masked_lm_prob, max_predictions_per_seq, 32000, rng)
        instance = TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=is_random_next,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
        instances.append(instance)

      # Swap
      tokens = [CLS_ID] + y + [SEP_ID] + x + [SEP_ID]
      segment_ids = [0] * (len(y) + 2) + [1] * (len(x) + 1)
      is_random_next = True
      if len(tokens) < max_seq_length:
        (tokens, masked_lm_positions,
            masked_lm_labels) = create_masked_lm_predictions(
                tokens, masked_lm_prob, max_predictions_per_seq, 32000, rng)
        instance = TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=is_random_next,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
        instances.append(instance)

      # Random
      tokens = [CLS_ID] + x + [SEP_ID] + z + [SEP_ID]
      segment_ids = [0] * (len(x) + 2) + [1] * (len(z) + 1)
      is_random_next = True
      if len(tokens) < max_seq_length:
        (tokens, masked_lm_positions,
            masked_lm_labels) = create_masked_lm_predictions(
                tokens, masked_lm_prob, max_predictions_per_seq, 32000, rng)
        instance = TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=is_random_next,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
        instances.append(instance)

  return instances
  

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

CLS_ID = 32000
SEP_ID = 32001
MASK_ID = 32002

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == CLS_ID or token == SEP_ID:
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = MASK_ID
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = rng.randint(0, vocab_words - 1)

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary the BERT model will train on.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus. can be directory with .txt files or a path to a single file")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model checkpoints will be written.")

    ## Other parameters

    # str
    parser.add_argument("--bert_model", default="bert-large-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                              "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    #int 
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--dupe_factor",
                        default=10,
                        type=int,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--max_predictions_per_seq",
                        default=20,
                        type=int,
                        help="Maximum sequence length.")
                             

    # floats

    parser.add_argument("--masked_lm_prob",
                        default=0.15,
                        type=float,
                        help="Masked LM probability.")

    parser.add_argument("--short_seq_prob",
                        default=0.1,
                        type=float,
                        help="Probability to create a sequence shorter than maximum sequence length")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--random_seed',
                        type=int,
                        default=12345,
                        help="random seed for initialization")

    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('/tmp/patona_nlp/sentencepiece/spm_model-20200718.model')

    input_files = []
    if os.path.isfile(args.input_file):
      input_files.append(args.input_file)
    elif os.path.isdir(args.input_file):
      input_files = [os.path.join(args.input_file, f) for f in os.listdir(args.input_file) if (os.path.isfile(os.path.join(args.input_file, f)) and f.endswith('.txt') )]
    else:
      raise ValueError("{} is not a valid path".format(args.input_file))

    rng = random.Random(args.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, args.max_seq_length, args.dupe_factor,
        args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq,
        rng)

    output_file = args.output_file


    write_instance_to_example_file(instances, tokenizer, args.max_seq_length,
                                    args.max_predictions_per_seq, output_file)


if __name__ == "__main__":
    main()
