from datasets import DatasetDict,load_from_disk,concatenate_datasets
from tqdm import tqdm
import math
import json
from transformers import XLMRobertaTokenizerFast,XLMRobertaForMaskedLM,XLMRobertaConfig,XLMRobertaModel,XLMRobertaForTokenClassification
import torch,os
import copy
from collections import OrderedDict

def scale_model(save_path):
    # 获取词表大小
    with open('./utils_data/small_vocab_idxs.json','r',encoding='utf-8') as f:
        vocabs=json.load(f)

    tokenizer=XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    
    roberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')
    print(
            f"  Number of trainable parameters = {sum(p.numel() for p in roberta_model.parameters())}"
        )
    vocabs=set(vocabs)
    vocabs.update(tokenizer.convert_tokens_to_ids(['<s>','</s>','<unk>','<pad>','<mask>']))
    vocabs=list(vocabs)
    vocabs.sort()
    vocabs_dict={word:i for i,word in enumerate(vocabs)}
    config = roberta_model.config
    config.num_labels = len(vocabs)
    model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base',config=config)
    for k,v in vocabs_dict.items():
            with torch.no_grad():
                model.classifier.weight[v]=roberta_model.lm_head.decoder.weight[k]
                model.classifier.bias[v]=roberta_model.lm_head.decoder.bias[k]


    model.save_pretrained(save_path)

    print(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters())}"
        )
    
    num=0
    for p in model.parameters():
        num+=p.numel()
    print(num)
    

save_path='../pretrain/scale_xlmr'
scale_model(save_path)
