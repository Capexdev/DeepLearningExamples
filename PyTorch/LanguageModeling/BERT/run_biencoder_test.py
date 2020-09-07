import sys
#sys.path.append('/home/azureuser/workspace/DeepLearningExamples/PyTorch/LanguageModeling/BERT')
sys.path.append('/home/azureuser/workspace/dummy_apex')

import modeling
from run_biencoder import create_pretraining_dataset, pretraining_dataset
from run_biencoder import BertBiEncoderCriterion
import numpy as np
from argparse import Namespace
import torch

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
config = modeling.BertConfig.from_json_file('bert_config-base.json')
model = modeling.BertBiEncoder(config)
criterion = BertBiEncoderCriterion()
i = 0
input1_ids, input1_mask, input2_ids, input2_mask = [], [], [], []
for input1_ids_, input1_mask_, input2_ids_, input2_mask_ in dataset:
    if i % 10 == 0:
        input1_text = tokenizer.decode(input1_ids_[np.where(input1_mask_)])
        input2_text = tokenizer.decode(input2_ids_[np.where(input2_mask_)])
        print(input1_text, '=>', input2_text)
        input1_ids.append(torch.tensor(input1_ids_))
        input1_mask.append(torch.tensor(input1_mask_))
        input2_ids.append(torch.tensor(input2_ids_))
        input2_mask.append(torch.tensor(input2_mask_))
    i += 1
    if i >= 100:
        break

input1_ids = torch.stack(input1_ids) 
input1_mask = torch.stack(input1_mask)
input2_ids = torch.stack(input2_ids)
input2_mask = torch.stack(input2_mask)
input1_embedding = model(input1_ids, torch.zeros_like(input1_ids), input1_mask)
input2_embedding = model(input2_ids, torch.ones_like(input2_ids), input2_mask)
loss = criterion(input1_embedding, input2_embedding)
print(loss)
