from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
# from initialize import get_en_zh_dict
import json
from collections import OrderedDict
class seq_model(nn.Module):
    def __init__(self, model,num_labels,config):
        # super().__init__(config)
        super(seq_model, self).__init__()
        self.model=model
        self.num_labels=num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1) # self.classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None):

        outputs = self.model(
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
        pooled_output = outputs[1]        # use [CLS] pooled output
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        logits = self.sigmoid(logits)
        outputs = (logits,) + outputs[2:] 
        if labels is not None:
            if self.num_labels == 2:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.to(torch.float32).view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        output = (logits,)
        # output = (sampled_logits,)
        return ((loss,) + output) if loss is not None else output
