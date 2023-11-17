

import json,os
import re
from tqdm import tqdm
import random
from transformers import XLMRobertaTokenizerFast
from datasets import load_dataset, load_from_disk,DatasetDict,concatenate_datasets
random.seed(42)

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
    caption = re.sub(r"\/", ' / ', caption)
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

def get_medical_entity():
    dic={}
    with open('../../XLMR+Pretraining/process_data/utils_data/en_to_zh.json') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    for line in bar:
        line=json.loads(line)
        for k,v in line.items():
            dic[k]=v
    return dic

def get_entity_candidates(text_lst):
    # 假设实体表示不超过5个token
    candidates={}
    i=0
    length=len(text_lst)
    while i<length:
        for size in range(1,6):
            upper=i+size
            if i+size>length:
                upper=length
            candidates[(i,upper)]=' '.join(text_lst[i:i+size]).strip()
            # candidates.update([' '.join(text_lst[i:i+size]).strip()])
        i+=1
    return candidates
    
def max_match(entities):
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
    del_spans=[]
    for span,candidate in entities.items():
        if len(candidate['zh'])==1 or candidate['zh'].lower()==candidate['en'].lower():
            del_spans.append(span)
    for span in del_spans:
        try:
            del entities[span]
        except:
            pass

    return entities

def filter_medical_article(text_lst,medical_dict):
    entities={}
    candidates=get_entity_candidates(text_lst)
    # entities=[]
    for span,candidate in candidates.items():
        try:
            v=medical_dict[candidate]
            # label=id_to_label[id_]
            entities[span]={'en':candidate.strip(),'zh':v['zh'],'label':v['label'],'id':v['id']}
        except:
            pass
    entities=max_match(entities)
    return entities

def random_replace_entity(text_lst,entities,num):
    length=len(entities)
    lst=list(range(length))
    random_choice=random.sample(lst,num)
    res=''
    last_idx=0
    for i,(k,v) in enumerate(entities.items()):
        if i in random_choice:
            span=k
            en_entity=v['en']
            zh_entity=v['zh']
            assert en_entity==' '.join(text_lst[span[0]:span[1]]).strip()
            # res=res.strip()
            res+=' '.join(text_lst[last_idx:span[0]])
            res+=' '+zh_entity+' '
            last_idx=span[1]
    res+=' '.join(text_lst[span[1]:])
    res=' '.join(res.split())
    return res.strip()


def entity_knowledge(in_file,out_file):
    medical_entity=get_medical_entity()
    w_f=open(out_file,'w',encoding='utf-8')
    tokenizer=XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    with open(in_file,'r',encoding='utf-8') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    count=0
    for example in bar:
        line=json.loads(example)
        text=line['text']
        text=pre_caption(text)
        text_lst=text.split()
        # tokens=tokenizer.tokenize(text)
        entities=filter_medical_article(text_lst,medical_entity)
        switchs_sents=set()
        if text.strip():
            if len(entities)>0:
                # num=random.choice(list(range(1,len(entities)+1)))
                num=min(10,len(entities))
                sent=random_replace_entity(text_lst,entities,num)
                switchs_sents.update([sent])
                switchs_sents=list(switchs_sents)
                for sent in switchs_sents:
                    json.dump({'text':sent.strip()},w_f,ensure_ascii=False)
                    w_f.write('\n')
                json.dump({'text':text.strip()},w_f,ensure_ascii=False)
                w_f.write('\n')
                count+=1
                if count==100000:
                    break


def get_pair_entity(entities,entity_to_rel):

    en_rich_con=set()
    zh_rich_con=set()
    mix_rich_con=set()
    for k1,v1 in entities.items():
        ent1_id=v1['id']
        for k2,v2 in entities.items():
            ent2_id=v2['id']
            try:
                rel=entity_to_rel[ent1_id+'-'+ent2_id]
                en_rich_con.update([' </s> '+v1['en']+' '+rel['en_rel']+' '+v2['en']])
                zh_rich_con.update([' </s> '+v1['zh']+' '+rel['zh_rel']+' '+v2['zh']])
                num=random.random()
                if num<0.5:
                    mix_rich_con.update([' </s> '+v1['en']+' '+rel['en_rel']+' '+v2['en']])
                else:
                    mix_rich_con.update([' </s> '+v1['zh']+' '+rel['zh_rel']+' '+v2['zh']])
                # print(rel)
            except:
                pass
    en_rich_con=list(en_rich_con)
    en_rich_con=' '.join(en_rich_con)
    zh_rich_con=list(zh_rich_con)
    zh_rich_con=' '.join(zh_rich_con)
    mix_rich_con=list(mix_rich_con)
    mix_rich_con=' '.join(mix_rich_con)
    return en_rich_con.strip(),zh_rich_con.strip(),mix_rich_con.strip()

def get_relations():
    eng_zh_rels={}
    with open('../../XLMR+Pretraining/process_data/utils_data/eng_zh_rels.json','r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            line=json.loads(line)
            eng_zh_rels[line['en']]=line['zh']
    with open('../../XLMR+Pretraining/process_data/utils_data/knowledege.json','r',encoding='utf-8') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    entity_to_rel={}

    for line in bar:
        line=json.loads(line)
        for k,v in line.items():
            entity_to_rel[k]={'zh_rel':eng_zh_rels[v],'en_rel':v}
    return entity_to_rel

def fact_knowledge_split(in_file,out_file):
    medical_entity=get_medical_entity()
    entity_to_rel=get_relations()
    tokenizer=XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    w_f=open(out_file,'w',encoding='utf-8')
    with open(in_file,'r',encoding='utf-8') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    count=0
    for example in bar:
        line=json.loads(example)
        text=line['text']
        text=pre_caption(text)
        texts=text.split('.')
        entity_num=0
        flag=False
        if text.strip():
            while texts[-1]=='':
                texts.pop(-1)
            i=0
            sub_text=''
            while i<len(texts):
                texts[i]=texts[i].strip()+'。 '
                sub_text+=texts[i]
                if len(sub_text)>=256:
                    text_lst=sub_text.split()
                    entities=filter_medical_article(text_lst,medical_entity)
                    # 随机选择要生成的code switch的句子数量
                    # sentence_num=random.choice(list(range(6)))
                    # switchs_sents=set()
                    entity_num+=len(entities)
                    if len(entities)>0:
                        en_rich_con,zh_rich_con,mix_rich_con=get_pair_entity(entities,entity_to_rel)
                        if en_rich_con:
                            text=sub_text+' '+en_rich_con.strip()
                            json.dump({'text':text},w_f,ensure_ascii=False)
                            w_f.write('\n')
                            text=sub_text+' '+zh_rich_con.strip()
                            json.dump({'text':text},w_f,ensure_ascii=False)
                            w_f.write('\n')
                            text=sub_text+' '+mix_rich_con.strip()
                            json.dump({'text':text},w_f,ensure_ascii=False)
                            w_f.write('\n')
                            count+=1
                            if count==100000:
                                flag=True
                                break
                    sub_text=''
                i+=1
            else:
                if len(sub_text)>=128:
                    text_lst=sub_text.split()
                    entities=filter_medical_article(text_lst,medical_entity)
                    # 随机选择要生成的code switch的句子数量
                    # sentence_num=random.choice(list(range(6)))
                    # switchs_sents=set()
                    entity_num+=len(entities)
                    if len(entities)>0:
                        en_rich_con,zh_rich_con,mix_rich_con=get_pair_entity(entities,entity_to_rel)
                        if en_rich_con:
                            text=sub_text+' '+en_rich_con.strip()
                            json.dump({'text':text},w_f,ensure_ascii=False)
                            w_f.write('\n')
                            text=sub_text+' '+zh_rich_con.strip()
                            json.dump({'text':text},w_f,ensure_ascii=False)
                            w_f.write('\n')
                            text=sub_text+' '+mix_rich_con.strip()
                            json.dump({'text':text},w_f,ensure_ascii=False)
                            w_f.write('\n')
                            count+=1
                            if count==100000:
                                flag=True
                                break
                    sub_text=''
        if flag:
            break

if __name__=="__main__":
    in_file='./data/english_examples.json'
    out_file='./data/english_entity_text.json'
    entity_knowledge(in_file,out_file)
    out_file='./data/english_fact_text.json'
    fact_knowledge_split(in_file,out_file)