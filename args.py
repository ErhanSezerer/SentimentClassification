import argparse

RANDOM_STATE = 42


# ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




parser = argparse.ArgumentParser(description='Experiments for Plausible Detection Models')
#BERT arguments
parser.add_argument('--MAX_LEN', type=int, default=512)
parser.add_argument('--num_label', type=int, default=5)
parser.add_argument('--bert_sep', type=str, default="<-SEP->")
parser.add_argument('--target_class', default=1, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--momentum', default=0.9)

#Random Forest arguments
parser.add_argument('--estimators', default=10, type=int)

#doc2vec arguments (also uses epochs and lr)
parser.add_argument('--emb_dimension', default=300, type=int)

#arguments for significance test
parser.add_argument('--test', default="anova", type=str) #options: anova, spearman, mcnemar, friedman


#other arguments
parser.add_argument('--checkpoint_dir', default='./model')
parser.add_argument('--data_dir', default='/media/darg1/Data/Projects/chess/CommentClassification/data')
parser.add_argument('--use_gpu', type=str2bool)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--model_name', type=str, default="bert_chess_classifier_1")
parser.add_argument('--classifier', type=str, default="bert") #options: bert, randomforest, svm, doc2vec
parser.add_argument('--print_preds', type=str2bool)



args = parser.parse_args()
