"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0
"""

import argparse, json, logging, numpy, os, random, sys, torch
from turtle import end_fill

from data import ConllCorpus,Dictionary
from train import train, final_test
from models import TransformerEncoder, LinearCRF, MLP
from utils import get_category2mentions, get_label2tokens
from transformers import BertTokenizer,XLMRobertaTokenizerFast


logger = logging.getLogger(__name__)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    # input
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--task_name", default="development", type=str)
    parser.add_argument("--train_filepath", default="train.txt", type=str)
    parser.add_argument("--dev_filepath", default="dev.txt", type=str)
    parser.add_argument("--test_filepath", default="test.txt", type=str)
    parser.add_argument("--source_folder", default='/data1/gl/project/process_CDR_CHR/PURE_DATA/CDR/final/txt', type=str)
    parser.add_argument('--task', type=str, default='pubmed', required=True, choices=['pubmed-met','pubmed','met','cail','cdr','chr','ade','ncbi','bc2gm','jnlpba','bc5cdr-disease','bc5cdr-chem','NCBI-disease','biored','official_biored','jnlpba_hf','gda','drugprot','multi_cdr'])

    parser.add_argument("--reset_weights", action='store_true', help='whether reset model weights with our model.')
    parser.add_argument("--type", default="bert", type=str)
    
    # output
    parser.add_argument("--output_dir", default="development", type=str)
    parser.add_argument("--result_filepath", default="development.json", type=str)
    parser.add_argument("--log_filepath", default="development.log")

    # train
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--min_lr", default=1e-8, type=float)
    parser.add_argument("--train_bs", default=16, type=int)
    parser.add_argument("--eval_bs", default=16, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--anneal_factor", default=0.5, type=float)
    parser.add_argument("--anneal_patience", default=20, type=int)
    parser.add_argument("--early_stop_patience", default=10, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--use_our_pretrain", action="store_true")
    parser.add_argument("--do_lower_case", action='store_true', help='whether use multi-task model.')

    parser.add_argument("--p_power", default=1.0, type=float,
                        help="the exponent in p^x, used to smooth the distribution, "
                             "if it is 1, the original distribution is used; "
                             "if it is 0, it becomes uniform distribution")
    # environment
    parser.add_argument("--seed", default=52, type=int)
    parser.add_argument("--device", default=0, type=int)

    # embeddings
    parser.add_argument("--embedding_type", default="bert", type=str)
    parser.add_argument("--pretrained_dir", default=None, type=str)
    parser.add_argument("--max_position_embeddings", default=512, type=int)
    
    # dropout
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--word_dropout", default=0.05, type=float)
    parser.add_argument("--variational_dropout", default=0.5, type=float)

    parser.add_argument("--do_train", action="store_true")

    parser.add_argument("--debug", action="store_true")

    args, _ = parser.parse_known_args()

    args.train_filepath = os.path.join(args.data_folder, args.train_filepath)
    args.dev_filepath = os.path.join(args.data_folder, args.dev_filepath)
    args.test_filepath = os.path.join(args.data_folder, args.test_filepath)

    return args


def random_seed(seed=52):
    random.seed(seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

task_ner_labels = {
    'ade':['ADE','Drug'],
    'cdr':['CHEM','DIS']
}

if __name__ == "__main__":
    args = parse_parameters()
    device = torch.device("cuda:%d" % args.device)
    args.result = {}

    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    random_seed(args.seed)
    logger.info(f'CONFIG: "{args}"')
    if args.type=='bert':
        tokenizer=BertTokenizer.from_pretrained(args.pretrained_dir,do_lower_case=args.do_lower_case)
    else:
        tokenizer=XLMRobertaTokenizerFast.from_pretrained(args.pretrained_dir,do_lower_case=args.do_lower_case)

    args.tokenizer=tokenizer
    corpus = ConllCorpus(args,False)
    
    ner_label_list=['O']
    
    for tag in task_ner_labels[args.task]:
        ner_label_list.append('B-'+tag)
        ner_label_list.append('I-'+tag)
    

    tag_dict = Dictionary(unk_value=None)
    for tag in ner_label_list:
        tag_dict.add_item(tag)
   
    category2mentions = get_category2mentions(corpus.train,args)
    label2tokens = get_label2tokens(corpus.train, args.p_power,args)

    args.num_labels=len(tag_dict)
    encoder = TransformerEncoder(args, device)
    mlp = MLP(encoder.output_dim, len(tag_dict), encoder.output_dim, 1).to(device)
    crf = LinearCRF(tag_dict, device)

    if args.do_train:
        dev_scores = train(args, encoder, mlp, crf, corpus.train, corpus.dev, category2mentions, label2tokens)

    args.result["dev_before_result"] = final_test(args, encoder, mlp, crf, corpus.dev, "dev",'dev')
    args.result["test_before_result"] = final_test(args, encoder, mlp, crf, corpus.test, "test",'dev')
    args.result["train_result"] = final_test(args, encoder, mlp, crf, corpus.train, "train",'dev')
    
    with open(args.result_filepath, "w") as f:
        args.tokenizer={}
        json.dump(vars(args), f, indent=4)