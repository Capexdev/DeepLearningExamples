import sys
#sys.path.append('/home/azureuser/workspace/DeepLearningExamples/PyTorch/LanguageModeling/BERT')
sys.path.append('/home/azureuser/workspace/dummy_apex')

from modeling import BertBiEncoder
from run_biencoder import create_pretraining_dataset, pretraining_dataset
import numpy as np
from argparse import Namespace

input_file = '/tmp/patona_nlp/bert/hdf5_lower_case_0_seq_len_64_max_pred_15_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/chat_corpus/chat_corpus_test_0.hdf5'
max_pred_length = 100
shared_list = [0]
worker_init = lambda : 0
args = Namespace()
args.train_batch_size = 3
args.n_gpu = 1
dataset = pretraining_dataset(input_file, max_pred_length)
from patona_nlp.tokenization import PatonaSentencePieceTokenizer
tokenizer = PatonaSentencePieceTokenizer.from_pretrained('/tmp/patona_nlp/pretrained/patona-base-spiece-20200815')

for input1_ids, input1_mask, input2_ids, input2_mask in dataset:
    input1_text = tokenizer.decode(input1_ids[np.where(input1_mask)])
    input2_text = tokenizer.decode(input2_ids[np.where(input2_mask)])
    print(input1_text, '=>', input2_text)