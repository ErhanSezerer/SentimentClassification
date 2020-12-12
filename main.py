import logging
import coloredlogs
import torch
import csv
import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers import BertForSequenceClassification, AdamW
from args import args
from bert import train_bert, test_bert, load_model
from preprocess import read_samples_bert, read_samples_ML, read_samples_doc2vec

from gensim.models.doc2vec import Doc2Vec
from doc2vec import train_doc2vec, test_doc2vec
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from ML import train_ML, test_ML

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)





def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)





def run_bert(device, results_file):

	set_seed(args.seed)
	torch.cuda.empty_cache()

	#get the data
	logging.info('Constructing datasets...')
	train_data, dev_data, test_data = read_samples_bert(args.data_dir)

	#prepare the model and data
	model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_label,
                                                          output_attentions=False, output_hidden_states=False)
	optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
	epoch = args.epochs

	train_iter = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batch_size)
	dev_iter = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.batch_size)
	test_iter = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.batch_size)


	#create model save directory
	checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)


	#run the tests
	logging.info(
        "Number of training samples {train}, number of dev samples {dev}, number of test samples {test}".format(
            train=len(train_data),
            dev=len(dev_data),
            test=len(test_data)))

	train_bert(epoch, model, train_iter, dev_iter, optimizer, device, checkpoint_dir, results_file)

	model = load_model(checkpoint_dir)
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = test_bert(test_iter, model, device)
	del model
	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave






def run_randomforest(results_file):
	#get the data
	train_features, dev_features, test_features, train_labels, dev_labels, test_labels = read_samples_ML(args.data_dir)

	#prepare the model
	model = RandomForestClassifier(n_estimators=args.estimators)

	#create model save directory
	checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	#run tests
	logging.info(
        "Number of training samples {train}, number of dev samples {dev}, number of test samples {test}".format(
            train=train_features.shape[0],
            dev=dev_features.shape[0],
            test=test_features.shape[0]))

	train_ML(model, train_features, dev_features, train_labels, dev_labels, checkpoint_dir, results_file)
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = test_ML(model, test_features, test_labels)

	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave





def run_SVM(results_file):
	#get the data
	train_features, dev_features, test_features, train_labels, dev_labels, test_labels = read_samples_ML(args.data_dir)

	#prepare the model
	model = OneVsRestClassifier(SVC())

	#create model save directory
	checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	#run tests
	logging.info(
        "Number of training samples {train}, number of dev samples {dev}, number of test samples {test}".format(
            train=train_features.shape[0],
            dev=dev_features.shape[0],
            test=test_features.shape[0]))

	train_ML(model, train_features, dev_features, train_labels, dev_labels, checkpoint_dir, results_file)
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = test_ML(model, test_features, test_labels)

	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave






def run_doc2vec(results_file):
	#get the data
	train_samples, dev_samples, test_samples, dev_labels, test_labels = read_samples_doc2vec(args.data_dir)

	#prepare the model
	model = Doc2Vec(size=args.emb_dimension, alpha=args.lr, min_alpha=0.00025, min_count=1, dm=1)
	logreg = LogisticRegression()
	
	#create model save directory
	checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	#run tests
	logging.info(
        "Number of training samples {train}, number of dev samples {dev}, number of test samples {test}".format(
            train=len(train_samples),
            dev=len(dev_samples),
            test=len(test_samples)))

	train_doc2vec(args.epochs, model, logreg, train_samples, dev_samples, dev_labels, checkpoint_dir, results_file)
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = test_doc2vec(test_samples, test_labels, model, logreg)

	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave







if __name__ == '__main__':

	logger.info('===========Training============')

	#resolve file names and directories
	if args.classifier == "bert":
		device = 'cuda' if args.use_gpu and torch.cuda.is_available else 'cpu'
		file_name = '{model_name}_epochs_{epoch}_lr_{lr}.csv'.format(model_name='bert', epoch=args.epochs, lr=args.lr)
		args.model_name = '{model_name}_epochs_{epoch}_lr_{lr}'.format(model_name='bert', epoch=args.epochs, lr=args.lr)
	elif args.classifier == "randomforest":
		file_name = '{model_name}_estimators_{estimators}.csv'.format(model_name='randomforest', estimators=args.estimators)
		args.model_name = '{model_name}_estimators_{estimators}'.format(model_name='randomforest', estimators=args.estimators)
	elif args.classifier == "svm":
		file_name = '{model_name}.csv'.format(model_name='svm')
		args.model_name = '{model_name}'.format(model_name='svm')
	elif args.classifier == "doc2vec":
		file_name = '{model_name}_dim_{dim}_epochs_{epoch}_lr_{lr}.csv'.format(model_name='doc2vec', epoch=args.epochs, lr=args.lr, dim=args.emb_dimension)
		args.model_name = '{model_name}_dim_{dim}_epochs_{epoch}_lr_{lr}'.format(model_name='doc2vec', epoch=args.epochs, lr=args.lr, dim=args.emb_dimension)
		

	results_file = os.path.join(args.checkpoint_dir, file_name)
	with open(results_file, 'w') as output_file:
		cw = csv.writer(output_file, delimiter='\t')
		cw.writerow(['Epoch', 'Acc', 'Precision', 'Recall', 'F1', 
			'F1-description', 'F1-quality', 'F1-planning', 'F1-currentinfo', 'F1-gameinfo',
			'P-description', 'P-quality', 'P-planning', 'P-currentinfo', 'P-gameinfo',
			'R-description', 'R-quality', 'R-planning', 'R-currentinfo', 'R-gameinfo'])

	#run tests
	if args.classifier == "bert":
		acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = run_bert(device, results_file)
	elif args.classifier == "randomforest":
		acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = run_randomforest(results_file)
	elif args.classifier == "svm":
		acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = run_SVM(results_file)
	elif args.classifier == "doc2vec":
		acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = run_doc2vec(results_file)

	#print and log the results
	stats_template = '\nAccuracy: {acc}\n' \
                 'F1: {f1}\n' \
                 'ave. F1: {f1_ave}\n' \
                 'Recall: {recall}\n' \
                 'ave. Recall: {recall_ave}\n' \
                 'Precision: {prec}\n' \
                 'ave. Precision: {prec_ave}\n'
	logger.info(stats_template.format(acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave))

	#write results into csv
	with open(results_file, 'a') as output_file:
		cw = csv.writer(output_file, delimiter='\t')
		cw.writerow(['test',
				'%0.4f' % acc,
				'%0.4f' % prec_ave,
				'%0.4f' % recall_ave,
				'%0.4f' % f1_ave,
				'%0.4f' % f1[0],
				'%0.4f' % f1[1],
				'%0.4f' % f1[2],
				'%0.4f' % f1[3],
				'%0.4f' % f1[4],
				'%0.4f' % prec[0],
				'%0.4f' % prec[1],
				'%0.4f' % prec[2],
				'%0.4f' % prec[3],
				'%0.4f' % prec[4],
				'%0.4f' % recall[0],
				'%0.4f' % recall[1],
				'%0.4f' % recall[2],
				'%0.4f' % recall[3],
				'%0.4f' % recall[4]])






