import logging
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from args import args
from torch.utils.data import TensorDataset
import pytreebank

logger = logging.getLogger('preprocess.py')
import numpy as np
import os



def read_files(file_path):
	#read the dataset and find the labels
	train_text = []
	train_labels = []
	dev_text = []
	dev_labels = []
	test_text = []
	test_labels = []
	dataset = pytreebank.load_sst()

	return train_text, dev_text, test_text, train_labels, dev_labels, test_labels 






def read_samples_bert(file_path):
	train_text, dev_text, test_text, train_labels, dev_labels, test_labels = read_files(file_path)

	#tokenize for bert
	logging.info("Tokenizing the Dataset for BERT...")
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

	train_ids = []
	train_att_mask = []
	dev_ids = []
	dev_att_mask = []
	test_ids = []
	test_att_mask = []
	for article in train_text:
		encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
		train_ids.append(encoded_article['input_ids'])
		train_att_mask.append(encoded_article['attention_mask'])
	for article in test_text:
		encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
		test_ids.append(encoded_article['input_ids'])
		test_att_mask.append(encoded_article['attention_mask'])
	for article in dev_text:
		encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
		dev_ids.append(encoded_article['input_ids'])
		dev_att_mask.append(encoded_article['attention_mask'])

	#prepare torch tensors from the data
	train_ids = torch.cat(train_ids, dim=0)
	test_ids = torch.cat(test_ids, dim=0)
	train_att_mask = torch.cat(train_att_mask, dim=0)
	test_att_mask = torch.cat(test_att_mask, dim=0)
	train_labels = torch.tensor(train_labels)
	test_labels = torch.tensor(test_labels)
	dev_ids = torch.cat(dev_ids, dim=0)
	dev_att_mask = torch.cat(dev_att_mask, dim=0)
	dev_labels = torch.tensor(dev_labels)

	train_dataset = TensorDataset(train_ids, train_att_mask, train_labels)
	test_dataset = TensorDataset(test_ids, test_att_mask, test_labels)
	dev_dataset = TensorDataset(dev_ids, dev_att_mask, dev_labels)

	return train_dataset, dev_dataset, test_dataset












