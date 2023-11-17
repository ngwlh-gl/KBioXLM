
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import re,json
from transformers import AutoTokenizer,XLMRobertaTokenizerFast
import numpy as np
import random
random.seed(42)

def contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fa5':
            return True
    return False

def init():
    ...


def pre_caption(origin_caption):
    if origin_caption is None or origin_caption == '':
        return origin_caption
    # caption = caption.encode('raw_unicode_escape').decode('utf8','ignore')
    caption = re.sub("\u2009|\xa0", '', origin_caption)
    for c in ",'!\"#:;~”“’‘":
        caption = re.sub(r"{}".format(c), ' '+c+' ', caption)
    # caption = re.sub(r"[\[\]]", ' ', caption)
    caption = re.sub(r"\[",' [ ',caption)
    caption = re.sub(r"]",' ] ',caption)
    # caption = re.sub(r"\.(?!\d)", ' . ', caption)
    caption = re.sub(r",", ' , ', caption)
    caption = re.sub(r"=", ' = ', caption)
    caption = re.sub(r"%", ' % ', caption)
    caption = re.sub(r"-", ' - ', caption)
    caption = re.sub(r"±", ' ± ', caption)
    caption = re.sub(r"\+", ' + ', caption)
    # caption = re.sub(r"\/", ' / ', caption)
    caption = re.sub(r";", ' ; ', caption)
    caption = re.sub(r"\?", ' ? ', caption)
    caption = re.sub(r"\*", ' * ', caption)
    caption = re.sub(r"\.", ' . ', caption)

    caption = re.sub(r"\(", ' ( ', caption)
    caption = re.sub(r"\)", ' ) ', caption)
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    return caption

    
def get_id_to_labels():
    with open('./utils_data/cui2sty.json','r',encoding='utf-8') as f:
        cons=f.readlines()
    bar=tqdm(cons,desc='Load id_to_label data...')
    id_to_label={}
    for con in bar:
        line=json.loads(con)
        for k,v in line.items():
            id_to_label[k]=v
    return id_to_label


def small_vocab():
    with open('./utils_data/small_vocab_idxs.json','r',encoding='utf-8') as f:
        vocabs=json.load(f)
    vocabs=set(vocabs)
    vocabs.update([0,1,2,3,250001])
    vocabs=list(vocabs)
    vocabs.sort()
    vocab_lst={word:i for i,word in enumerate(vocabs)}
    return vocab_lst


def medical_entity():
    dic={}
    with open('./utils_data/zh_to_en.json') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    for line in bar:
        line=json.loads(line)
        for k,v in line.items():
            dic[k]=v['label']
    stop_words=[]
    with open('./utils_data/stopwords_zh.txt','r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            line=line.strip()
            stop_words.append(line)
    stop_lst=[]
    for k,v in dic.items():
        if k in stop_words:
            stop_lst.append(k)
    for word in stop_lst:
        del dic[word]

    with open('./utils_data/en_to_zh.json') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    for line in bar:
        line=json.loads(line)
        for k,v in line.items():
            dic[k]=v['label']
    stop_words=[]
    with open('./utils_data/stopwords_en.txt','r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            line=line.strip()
            stop_words.append(line)
    stop_lst=[]
    for k,v in dic.items():
        if k in stop_words or k.lower() in stop_words:
            stop_lst.append(k)
    for word in stop_lst:
        del dic[word] 

    return dic

def get_rels():
    with open('./utils_data/eng_zh_rels.json','r',encoding='utf-8') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    en_to_zh={}
    zh_to_en={}
    for line in cons:
        line=json.loads(line)
        en=line['en']
        zh=line['zh']
        en_to_zh[en]=zh
        zh_to_en[zh]=en
    print(len(en_to_zh))
    print(len(zh_to_en))
    medical_rels={}
    medical_rels.update(en_to_zh)
    medical_rels.update(zh_to_en)
    return medical_rels



def package_data(in_file,out_file,rels=False,lang='en'):
    zh_corpus = load_dataset('json', data_files=in_file)
    medical_rels=get_rels()
    en_tokenizer=XLMRobertaTokenizerFast.from_pretrained('/data1/gl/project/program/ChinesePLMs/CBLUE-main/data/model_data/xlm-r')
    medical_dict=medical_entity()
    id_to_label=get_id_to_labels()
    label_to_id={v:k for k,v in id_to_label.items()}
    sm_vocabs=small_vocab()

    def get_entity_candidates(entities,sep_idx):
        # 假设实体表示不超过3个token
        candidates={}
        spans=list(entities.keys())
        length=len(spans)
        for i,span in enumerate(spans):
            start=i
            span_len=0
            # entity=entities[span]
            can=[]
            i=start
            while span_len<4:
                span_len+=spans[i][-1]-spans[i][0]+1
                if span_len>3:
                    break
                can.append(entities[spans[i]]['entity'])
                if can:
                    try:  #找实体
                        try:
                            label=medical_dict[''.join(can).strip()]
                            candidates[(spans[start][0],spans[i][1])]={'entity':''.join(can).strip(),'id':'','label':label}
                        except:
                                label=medical_dict[' '.join(can).strip()]
                                candidates[(spans[start][0],spans[i][1])]={'entity':' '.join(can).strip(),'id':'','label':label}
                    except:   #找关系
                        if rels:
                            if spans[start][0]>sep_idx:
                                try:
                                    label=medical_rels[''.join(can).strip()]
                                    candidates[(spans[start][0],spans[i][1])]={'entity':''.join(can).strip(),'id':'','label':label}
                                except:
                                    try:
                                        label=medical_rels[' '.join(can).strip()]
                                        candidates[(spans[start][0],spans[i][1])]={'entity':' '.join(can).strip(),'id':'','label':label}
                                    except:
                                        pass
                        else:
                            pass
                i+=1
                if i>len(spans)-1:
                    break
        # return entities
        return candidates
    
    def max_match_entity(entities):
        # 对span进行最大匹配
        # 对相同开头的span，保留结果最大的span
        spans=list(entities.keys())
        # 遍历spans的开头
        starts={}
        for span in spans:
            start=span[0]
            end=span[1]
            try:
                end_=starts[start]
                if end_<end:
                    starts[start]=end
            except:
                starts[start]=end
        spans=[(k,v) for k,v in starts.items()]
        ends={}
        for span in spans:
            start=span[0]
            end=span[1]
            try:
                start_=ends[end]
                if start_>start:
                    ends[end]=start
            except:
                ends[end]=start
        spans=[(v,k) for k,v in ends.items()]
        new_spans=[]
        for i,span1 in enumerate(spans):
            flag=True
            for j,span2 in enumerate(spans):
                if i!=j:
                    if span1[0]>=span2[0] and span1[1]<=span2[1]:
                        flag=False
                        break
            if flag:
                new_spans.append(span1)
            
        del_spans=[]
        for span,candidate in entities.items():
            if span not in new_spans:
                del_spans.append(span)
        for span in del_spans:
            try:
                del entities[span]
            except:
                pass
        return entities

    def filter_medical_entity(tokens,sep_idx):
        candidates=get_whole_en_entity(tokens)
        entities={}
        for i,(span,candidate) in enumerate(candidates.items()):
            try:
                label=medical_dict[candidate]
                # id_=label_to_id[label]
                entities[span]={'entity':candidate.strip(),'id':'','label':label}
            except:
                entities[span]={'entity':candidate.strip(),'id':'','label':''}
        entities=get_entity_candidates(entities,sep_idx)
        entities=max_match_entity(entities)
        return entities
    
    def get_whole_en_entity(input_tokens):
        cand_indexes = []
        for i, token in enumerate(input_tokens):
        
            if len(cand_indexes) >= 1 and not token.startswith("▁"):
                if token in ['<unk>','<s>','</s>']:
                    cand_indexes.append([i])
                else:
                    cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
            
        candidates={}
        for span in cand_indexes:
            sub_text=''.join(input_tokens[span[0]:span[-1]+1])
            if contain_chinese(sub_text) or len(sub_text)==1 or sub_text[1:].isdigit() or len(sub_text)==2 or sub_text[0]!='▁':
                for idx in span:
                    candidates[(idx,idx)]=input_tokens[idx]
            else:
                candidates[(span[0],span[-1])]=sub_text[1:]
        return candidates
    
        
    def tokenize(tokenizer,text,max_length=512):
        # print(text_lst)
        if not text:
            token_ids=[]
        else:
            token_ids=tokenizer(text)['input_ids']
        tokens=tokenizer.convert_ids_to_tokens(token_ids)
        sep_idx=tokens.index('</s>')
        entities=filter_medical_entity(tokens,sep_idx)
        if len(token_ids)>max_length:
            token_ids=token_ids[:max_length-1]+[token_ids[-1]]
        input_ids=token_ids
        small_input_ids=get_small_vocab_idx(input_ids)
        assert len(input_ids)==len(small_input_ids)
        chinese_entity_ids=[]
        english_entity_ids=[]
        entity_ids=[]
        for span,info in entities.items():
            entity=info['entity']
            ner_label=info['label']
            id_=info['id']
            if span[1]>=len(input_ids):
                break
            entity_ids.append({'ner_label':ner_label,'span':np.array(list(range(span[0],span[1]+1)),dtype=np.int16)})
        attention_mask=[1]*len(input_ids)
        tokens=tokenizer.convert_ids_to_tokens(input_ids)
        return {'input_ids':input_ids,'attention_mask':attention_mask,'entities':entity_ids,'lang':lang,'small_vocab_labels':np.array(small_input_ids,dtype=np.int32),'tokens':tokens}

    def get_small_vocab_idx(input_ids):
        new_input_ids=[]
        for id_ in input_ids:
            try:
                new_id=sm_vocabs[id_]
            except:
                new_id=sm_vocabs[3]
            new_input_ids.append(new_id)
            # new_token_ids.append(sub_tok_id)
        return new_input_ids

    def preprocess_function(examples):
        input_ids=[]
        attention_mask=[]
        entities_lst=[]
        lang=[]
        small_vocab_labels=[]
        tokens=[]
        try:
            for text in examples['text']:
            
                text=pre_caption(text)
                result=tokenize(en_tokenizer,text)
                input_ids.append(result['input_ids'])
                attention_mask.append(result['attention_mask'])
                entities_lst.append(result['entities'])
                lang.append(result['lang'])
                small_vocab_labels.append(result['small_vocab_labels'])
                tokens.append(result['tokens'])

                # tokenize_results.append(tokenize(en_tokenizer,text_lst,entities))
            results={'input_ids':input_ids,'attention_mask':attention_mask,'entities':entities_lst,'lang':lang,'small_vocab_labels':small_vocab_labels,'tokens':tokens}
        except:  #对 passage-level的数据 进行tokenize
            labels_to_ids={'link':0,'con':1,'random':2}
            input_ids=[]
            lang=[]
            small_vocab_labels=[]
            labels=[]
            for sent1,sent2,label in zip(examples['sent1'],examples['sent2'],examples['label']):
                # text=getTextFromSample(sample)
                sent1=pre_caption(sent1)
                sent2=pre_caption(sent2)
                text=sent1+' </s> '+sent2
                result=tokenize(en_tokenizer,text)
                input_ids.append(result['input_ids'])
                attention_mask.append(result['attention_mask'])
                entities_lst.append(result['entities'])
                lang.append(result['lang'])
                small_vocab_labels.append(result['small_vocab_labels'])
                tokens.append(result['tokens'])
                labels.append(labels_to_ids[label])
                
                # tokenize_results.append(tokenize(en_tokenizer,text_lst,entities))
            results={'input_ids':input_ids,'attention_mask':attention_mask,'entities':entities_lst,'lang':lang,'small_vocab_labels':small_vocab_labels,'tokens':tokens,'labels':labels}
        return results
    
   
    datasetPath = out_file
    zh_corpus['train']=zh_corpus['train'].map(preprocess_function,batched=True)
    try:
        zh_corpus=zh_corpus.remove_columns(["text"])
    except:
        zh_corpus=zh_corpus.remove_columns(["sent1","sent2"])
    zh_corpus.save_to_disk(datasetPath)
    print(load_from_disk(datasetPath))

def statistics(examples):
    # count=0
    count_entities=0
    count_samples=0
    count_length=0
    bar=tqdm(examples['train'])

    for data in bar:
        count_entities+=len(data['entities'])
        entities=data['entities']
        for entity in entities:
            span=entity['span']
        count_length+=len(data['input_ids'])
        count_samples+=1
    print(count_entities/count_samples)
    print(count_length/count_samples)
    


    # print('loading...')
    # statistics(zh_corpus)
# zh_corpus = load_from_disk('/data1/gl/project/program/UHC/data/chinese_data_entity_knowledge_1w')
# print('loading...')
# statistics(zh_corpus)

if __name__=="__main__":
    # in_file='./data/chinese_data_entity_knowledge_1w.json'
    # out_file="./data/chinese_data_entity_knowledge_1w"
    
    # package_data(in_file,out_file,False,'cn')

    # in_file='./data/english_data_entity_knowledge_1w.json'
    # out_file="./data/english_data_entity_knowledge_1w"
    
    # package_data(in_file,out_file,False,'en')

    # in_file='./data/chinese_data_fact_knowledge_split_1w.json'
    # out_file="./data/chinese_data_fact_knowledge_split_1w"
    
    # package_data(in_file,out_file,True,'cn')

    # in_file='./data/english_data_fact_knowledge_split_1w.json'
    # out_file="./data/english_data_fact_knowledge_split_1w"
    
    # package_data(in_file,out_file,True,'en')

    # in_file='./data/pair_data_1w.json'
    # out_file="./data/pair_data_1w"
    
    # package_data(in_file,out_file,False,'cn')
    in_file='./data/examples.json'
    out_file="./data/examples"
    
    package_data(in_file,out_file,False,'en')
    
    