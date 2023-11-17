import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification,AutoModelForMaskedLM,RobertaForMaskedLM,RobertaForSequenceClassification,XLMRobertaForMaskedLM,XLMRobertaModel,XLMRobertaForTokenClassification
from torch.nn import CrossEntropyLoss, MSELoss
# from initialize import get_en_zh_dict
import json


def scale_vocab():
    with open('/data1/gl/project/program/UHC/small_vocab/small_vocab_idxs.json','r',encoding='utf-8') as f:
        vocabs=json.load(f)
    vocabs=set(vocabs)
    vocabs.update([0,1,2,3,250001])
    vocabs=list(vocabs)
    vocabs.sort()
    return vocabs
vocabs=scale_vocab()
id_to_labels={k:i for i,k in enumerate(vocabs)}

class UHCBert(nn.Module):
    def __init__(self, model_pth,config,model_type='bert'):
        # super().__init__(config)
        super(UHCBert, self).__init__()
        self.num_labels = config.num_labels
        self.config=config
        if model_type=='bert':
            self.model=AutoModelForMaskedLM.from_pretrained(model_pth)
        else:
            self.model=XLMRobertaForTokenClassification.from_pretrained(model_pth)
            # self.classifier=torch.nn.Linear(768,self.num_labels)
            checkpoint = torch.load(model_pth+'/pytorch_model.bin', map_location='cpu')
            msg = self.model.load_state_dict(checkpoint, strict=False)
            print(msg)
            
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                cls_labels=None,
                # student_input_ids=None,
                # small_vocab_labels=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None):

        student_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        student_logits = student_outputs['logits']
        loss = 0.0
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(student_logits.view(-1, self.config.num_labels), labels.view(-1))
        output = (student_logits,)  # + student_outputs[2:]
        # output = (sampled_logits,)
        return ((loss,) + output) if loss is not None else output
