import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
    

def test_ner(file, ent_info, few_shot = False, few_shot_file = None):
    if few_shot:
        with open(few_shot_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        random.shuffle(lines)
        samples = lines[:5]

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
                text = ''.join(sent)
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
                if not few_shot:
                    message = {
                        "role": "user", 
                        "content": '''
                        假设你是一个命名实体识别模型，你需要根据我给你的输入返回结果。 要求：您需要识别的实体类型如下：｛“{}”，“{}”｝。
                        如果存在具有上述实体类型的实体，则应返回下表：|实体类型|实体名称|
                        |[实体类型1]|[名称1]| 
                        |[实体类型2]|[名称2]|
                        ...
                        |[实体类型n]|[名称n]|
                        请将表中的[实体类型]和[名称]替换为您标识的特定实体类型和名称。输入: {} 请输出结果:'''.format(ent1, ent2, text)
                    }
                else:
                    examples = ''
                    reverse_ent_info = {v:k for k, v in ent_info.items()}
                    for sample in samples:
                        sample = json.loads(sample)
                        format_style = '输入: {} 请输出结果:{}'
                        result_format = '| 实体类型 | 实体名称 |\n| --- | --- |\n'
                        concate_str = '| {} | {} |\n'
                        sen = sample['sentence']
                        golds = sample['golds']
                        inp = sen
                        for gold in golds:
                            ent, tag = gold
                            tag = reverse_ent_info[tag]
                            result_format += concate_str.format(tag, ent)
                        # result_format = result_format
                        format_style = format_style.format(inp, result_format)
                        examples += format_style
                    examples = examples.rstrip()

                    message = {
                        "role": "user", 
                        "content": '''
                        假设你是一个命名实体识别模型，你需要根据我给你的输入返回结果。 要求：您需要识别的实体类型如下：｛“{}”，“{}”｝。
                        如果存在具有上述实体类型的实体，则应返回下表：|实体类型|实体名称|
                        |[实体类型1]|[名称1]| 
                        |[实体类型2]|[名称2]|
                        ...
                        |[实体类型n]|[名称n]|
                        请将表中的[实体类型]和[名称]替换为您标识的特定实体类型和名称。
                        举例如下：
                        {}
                        输入: {} 请输出结果:'''.format(ent1, ent2, examples, text)
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
                            entities.append(''.join(now_entity))
                            now_entity = []
                            tags.append(''.join(now_tag))
                            now_tag = []

                        now_tag.append(tag[2:])
                        now_entity.append(word)
                    elif tag[0] == 'I':
                        now_entity.append(word)
                else:
                    if now_entity:
                        entities.append(''.join(now_entity))
                        now_entity = []
                    if now_tag:
                        tags.append(''.join(now_tag))
                        now_tag = []
    f1 = metric.micro_avg_f_score()
    return f1

def test_gad(file, ent1, ent2, few_shot=False, few_shot_file = None):
    if few_shot:
        with open(few_shot_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        random.shuffle(lines)
        samples = lines[:5]
    # metric = Metric()
    golds = []
    preds = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    bar = tqdm(lines)
    for line in bar:

        line = json.loads(line)
        text = line['sentence']
        if not few_shot:
            message = {
                "role": "user", 
                "content": '''
                假设你是一个关系判断模型，你需要根据我给你的输入返回结果。
                要求：我将为您提供一段包含两个实体({},{})的句子，两个实体之间可能具有以下关系类型：1表示“有关系”，0表示“没关系”。
                您需要根据文章的内容来确定实体之间是否存在关系。
                请返回1或者0来表示两个实体之间是否存在关系。
                输入: {} 请输出结果:'''.format(ent1, ent2, text)
            }
        else:
            examples = ''
            for sample in samples:
                sample = json.loads(sample)
                format_style = '输入: {} 请输出结果:{}'
                examples += format_style.format(sample['sentence'], sample['label']) + '\n'
            examples = examples.rstrip()
            message = {
            "role": "user", 
            "content": '''
            假设你是一个关系判断模型，你需要根据我给你的输入返回结果。
            要求：我将为您提供一段包含两个实体({},{})的句子，两个实体之间可能具有以下关系类型：1表示“有关系”，0表示“没关系”。
            您需要根据文章的内容来确定实体之间是否存在关系。
            请返回1或者0来表示两个实体之间是否存在关系。
            举例如下：
            {}
            输入: {} 请输出结果:'''.format(ent1, ent2, examples, text)
        }
        res = get_response(message).strip()
        try:
            preds.append(int(res[0]))
        except:
            preds.append(int(res[-1]))
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

def test_hoc(file, labels, few_shot = False, few_shot_file = None):
    if few_shot:
        with open(few_shot_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        random.shuffle(lines)
        samples = lines[:2]
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
        if not few_shot:
            message = {
                "role": "user", 
                "content": '''
                假设你是一个文档分类模型，你需要根据我给你的输入返回结果。
                要求：我将为您提供一段文档，这个文档可能属于以下的类别：{}。
                您需要根据文章的内容来确定文档所属类别，类别可能不止一个。
                请以列表形式返回文档所属类别的索引。
                输入: {} 请输出结果:'''.format(labels, text)
            }
        else:
            examples = ''
            for sample in samples:
                sample = json.loads(sample)

                format_style = '输入: {} 请输出结果:{}'
                idxs = []
                for i, l in enumerate(sample['label']):
                    if l == 1:
                        idxs.append(i)
                examples += format_style.format(sample['sentence'], idxs) + '\n'
            examples = examples.rstrip()
            message = {
                "role": "user", 
                "content": '''
                假设你是一个文档分类模型，你需要根据我给你的输入返回结果。
                要求：我将为您提供一段文档，这个文档可能属于以下的类别：{}。
                您需要根据文章的内容来确定文档所属类别，类别可能不止一个。
                请以列表形式返回文档所属类别的索引。
                举例如下：
                {}
                输入: {} 请输出结果:'''.format(labels, examples, text)
            }
        res = get_response(message)
        pos1 = res.rfind('[')
        pos2 = res.rfind(']')
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
    # file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/NER/datasets/ade/eng_to_zh/test.txt'
    # f1 = test_ner(file, {'药物':'Drug', '不良反应':'ADE'})
    # print('ADE', 'zero-shot', f1)
    # few_shot_file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/NER/datasets/ade/zh_to_eng/train.json'
    # f1 = test_ner(file, {'药物':'Drug', '不良反应':'ADE'}, True, few_shot_file)
    # print('ADE', '5-shot', f1)
    # file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/NER/datasets/cdr/eng_to_zh/test.txt'
    # f1 = test_ner(file, {'化学药品':'CHEM', '疾病':'DIS'})
    # print('CDR', 'zero-shot', f1)
    # few_shot_file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/NER/datasets/cdr/zh_to_eng/train.json'
    # f1 = test_ner(file, {'化学药品':'CHEM', '疾病':'DIS'}, True, few_shot_file)
    # print('CDR', '5-shot', f1)
    file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/RE/data/seqcls/GAD_hf_en_zh/test_zh.json'
    # f1 = test_gad(file, '@DISEASE$', '@GENE$')
    # print('GAD', 'zero-shot', f1)
    few_shot_file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/RE/data/seqcls/GAD_hf_zh_en/train.json'
    f1 = test_gad(file, '@DISEASE$', '@GENE$', True, few_shot_file)
    print('GAD', 'few-shot', f1)
    file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/RE/data/seqcls/HoC_hf_en_zh/test_zh.json'
    # labels = ['activating invasion and metastasis', 'avoiding immune destruction',
    #       'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
    #       'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
    #       'sustaining proliferative signaling', 'tumor promoting inflammation']
    labels = ['activating invasion and metastasis', 'avoiding immune destruction',
          'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
          'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
          'sustaining proliferative signaling', 'tumor promoting inflammation']
    labels_infos = {}
    for i, label in enumerate(labels):
        labels_infos[i] = label
    # f1 = test_hoc(file, labels_infos)
    # print('HoC', 'zero-shot', f1)
    few_shot_file = '/data1/gl/project/ner-relation/kbio-xlm/downstream/RE/data/seqcls/HoC_hf_zh_en/train.json'
    f1 = test_hoc(file, labels_infos, True, few_shot_file)
    print('HoC', 'few-shot', f1)
