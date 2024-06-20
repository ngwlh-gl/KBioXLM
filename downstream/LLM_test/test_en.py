import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = '/data1/gl/project/ner-relation/kbio-xlm/downstream/LLM_test/models/huatuoGPT2-7b'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config.max_new_tokens = 2048
def get_response(message):
    messages = []
    
    # messages.append({"role": "user", "content": "肚子疼怎么办？"})
    messages.append(message)
    response = model.HuatuoChat(tokenizer, messages)
    return response

class Metric:
    def __init__(self):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        return sum(self._tps.values()) if class_name is None else self._tps[class_name]

    def get_fp(self, class_name=None):
        return sum(self._fps.values()) if class_name is None else self._fps[class_name]

    def get_tn(self, class_name=None):
        return sum(self._tns.values()) if class_name is None else self._tns[class_name]

    def get_fn(self, class_name=None):
        return sum(self._fns.values()) if class_name is None else self._fns[class_name]

    def f_score(self, class_name=None):
        tp = self.get_tp(class_name)
        fp = self.get_fp(class_name)
        fn = self.get_fn(class_name)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1

    def accuracy(self, class_name=None):
        tp = self.get_tp(class_name)
        fp = self.get_fp(class_name)
        fn = self.get_fn(class_name)
        return tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0

    def micro_avg_f_score(self):
        return self.f_score()[-1]

    def macro_avg_f_score(self):
        scores = [self.f_score(c)[-1] for c in self.get_classes()]
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def micro_avg_accuracy(self):
        return self.accuracy()

    def macro_avg_accuracy(self):
        accuracies = [self.accuracy(c) for c in self.get_classes()]
        return sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0.0

    def get_classes(self):
        all_classes = set(list(self._tps.keys()) + list(self._fps.keys()) + list(self._tns.keys()) + list(self._fns.keys()))
        return sorted([c for c in all_classes if c is not None])

    def to_dict(self):
        result = {}
        for n in self.get_classes():
            result[n] = {"tp": self.get_tp(n), "fp": self.get_fp(n), "fn": self.get_fn(n), "tn": self.get_tn(n)}
            result[n]["p"], result[n]["r"], result[n]["f"] = self.f_score(n)
        result["overall"] = {"tp": self.get_tp(), "fp": self.get_fp(), "fn": self.get_fn(), "tn": self.get_tn()}
        result["overall"]["p"], result["overall"]["r"], result["overall"]["f"] = self.f_score()
        return result

def update_metrics(metric, entities, tags, res):
    golds = []
    for ent, tag in zip(entities, tags):
        if (ent, tag) not in golds:
            golds.append((ent, tag))
    preds = res
    
    for pred in preds:
        # total_pre+=1
        if pred in golds:
            metric.add_tp(pred[1])
            # rec+=1
        else:
            metric.add_fp(pred[1])

    for gold in golds:
        # total_rec+=1
        if gold not in preds:
            metric.add_fn(gold[1])
            # pre+=1
        else:
            metric.add_tn(gold[1])
    

def test_ner(file, ent_info):
    sentences = []
    sents = []
    sent = []
    tags = []
    entities = []
    now_entity = []
    now_tag = []
    ent1, ent2 = list(ent_info.keys())
    last_tag = 'O'
    metric = Metric()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    bar = tqdm(lines)
    for line in bar:
        line = line.strip()
        if not line:
            if sent:
                text = ' '.join(sent)
                sent = []
                # message = {
                    # "system": "你是个很有用的助手。", 
                    # "user": '''
                    # 假设你是一个命名实体识别模型，你需要根据我给你的输入返回结果。 要求：您需要识别的实体类型如下：｛“{}”，“{}”｝。
                    # 如果存在具有上述实体类型的实体，则应返回下表：
                    # |实体类型|实体名称|
                    # |[实体类型1]|[名称1]| 
                    # |[实体类型2]|[名称2]|
                    # ...
                    # |[实体类型n]|[名称n]|
                    # 请将表中的[实体类型]和[名称]替换为您标识的特定实体类型和名称。'''.format(ent1, ent2),
                    # "assistant": "我明白了。作为一个命名实体识别模型，我将根据您提供的输入序列对其进行信息抽取，并以表格形式返回结果。",
                    # "user": "Input: {} 请输出结果:".format(text)
                # }
                message = {
                    "role": "user", 
                    "content": '''
                    Imagine you are a Named Entity Recognition model, and you need to return the results according to the input I give you as required. 
                    Requirement: The entity types you need to identify are as follows:｛“{}”，“{}”｝。
                    If there are entities with entity types mentioned above, the following table should be returned:
                    | Entity Type | Entity Name | 
                    | [Entity Type 1] | [Name1] | 
                    | [Entity Type 2] | [Name2] |
                    ...
                    | [Entity Type n] | [Namen] |
                    Please replace [Entity Type] and [Name] in the table with the specific entity type and name that you have identified.
                    Input: {} Please output the results:'''.format(ent1, ent2, text)
                }
                res = get_response(message)
                results = res.split('\n')[2:]
                res_list = []
                for res in results:
                    lst = res[1:-1].split('|')
                    try:
                        ent = lst[1].strip()
                        tp = ent_info[lst[0].strip()]
                        res_list.append((ent, tp))
                    except:
                        pass
                update_metrics(metric, entities, tags, res_list)
                entities = []
                tags = []
        else:
            word, tag = line.split('\t')
            if word == '-DOCSTART-':
                sentences.append(' '.join(sents))
                sents = []
                sent = []
            else:
                sent.append(word)
                if tag != 'O':
                    if tag[0] == 'B':
                        if now_entity:
                            entities.append(' '.join(now_entity))
                            now_entity = []
                            tags.append(''.join(now_tag))
                            now_tag = []

                        now_tag.append(tag[2:])
                        now_entity.append(word)
                    elif tag[0] == 'I':
                        now_entity.append(word)
                else:
                    if now_entity:
                        entities.append(' '.join(now_entity))
                        now_entity = []
                    if now_tag:
                        tags.append(''.join(now_tag))
                        now_tag = []
    f1 = metric.micro_avg_f_score()
    return f1

def test_gad(file, ent1, ent2):
    # metric = Metric()
    golds = []
    preds = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    bar = tqdm(lines)
    for line in bar:

        line = json.loads(line)
        text = line['sentence']
        message = {
            "role": "user", 
            "content": '''
            Assuming you are a relationship judgment model, you need to return the result based on the input I give you.
            Requirement: I will provide you with a sentence containing two entities ({}, {}) that may have the following types of relationships: 1 represents "related", 0 represents "not related".
            You need to determine whether there is a relationship between entities based on the content of the article.
            Please return 1 or 0 to indicate whether there is a relationship between two entities.
            Input: {} Please output the results:'''.format(ent1, ent2, text)
        }
        res = get_response(message)
        preds.append(int(res))
        golds.append(int(line['label']))
    preds = np.array(preds)
    golds = np.array(golds)
    TP = ((preds == golds) & (preds != 0)).astype(int).sum().item()
    P_total = (preds != 0).astype(int).sum().item()
    L_total = (golds != 0).astype(int).sum().item()
    P = TP / P_total if P_total else 0
    R = TP / L_total if L_total else 0
    F1 = 2 * P * R / (P + R) if (P + R) else 0
    # return {"precision": P, "recall": R, "F1": F1}
    return F1

def test_hoc(file, labels):
    from utils_hoc import eval_hoc
    golds = [] #[num_ex, num_class]
    preds = []  #[num_ex, num_class]
    ids = []
    # ids = eval_dataset["id"]
    # golds = []
    # preds = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    bar = tqdm(lines)
    for line in bar:

        line = json.loads(line)
        text = line['sentence']
        ids.append(line['id'])
        message = {
            "role": "user", 
            "content": '''
            Assuming you are a document classification model, you need to return results based on the input I give you.
            Requirement: I will provide you with a document that may belong to the following categories: {}。
            You need to determine the category of the document based on the content of the article, and there may be more than one category.
            Please return the index of the category to which the document belongs in the form of a list.
            Input: {} Please output the results:'''.format(labels, text)
        }
        res = get_response(message)
        pos1 = res.find('[')
        pos2 = res.find(']')
        try:
            if pos1 != -1 and pos2 != -1:
                results = eval(res[pos1:pos2+1])
                res = [0]*len(labels)
                for idx in results:
                    res[idx] = 1
            else:
                res = [0]*len(labels)
        except:
            res = [0]*len(labels)
        preds.append(res)
        golds.append(line['label'])
    preds = np.array(preds)
    golds = np.array(golds)
    return eval_hoc(golds.tolist(), preds.tolist(), ids)
    # return {"precision": P, "recall": R, "F1": F1}
    # return F1


if __name__ == "__main__":       
    file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/NER/datasets/ade/zh_to_eng/test.txt'
    f1 = test_ner(file, {'Drug':'Drug', 'Adverse Drug Event':'ADE'})
    print('ADE', f1)
    file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/NER/datasets/cdr/zh_to_eng/test.txt'
    f1 = test_ner(file, {'Chemical':'CHEM', 'Disease':'DIS'})
    print('CDR', f1)
    file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/RE/data/seqcls/GAD_hf_en_zh/test_en.json'
    f1 = test_gad(file, '@DISEASE$', '@GENE$')
    print('GAD', f1)
    file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/RE/data/seqcls/HoC_hf_en_zh/test_en.json'
    # labels = ['Sustaining proliferative signaling (PS)', 'Evading growth suppressors (GS)', 'Resisting cell death (CD)', 'Enabling replicative immortality (RI)', 'Inducing angiogenesis (A)', 'Activating invasion & metastasis (IM)', 'Genome instability & mutation (GI)', 'Tumor-promoting inflammation (TPI)', 'Deregulating cellular energetics (CE)', 'Avoiding immune destruction (ID)']
    labels = ['activating invasion and metastasis', 'avoiding immune destruction',
          'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
          'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
          'sustaining proliferative signaling', 'tumor promoting inflammation']
    labels_infos = {}
    for i, label in enumerate(labels):
        labels_infos[i] = label
    f1 = test_hoc(file, labels_infos)
    print('HoC', f1)