from transformers import (
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertModel,
    Trainer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    TrainerCallback,
    TrainingArguments,
    ProgressCallback
    )
import torch
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from transformers.data.data_collator import *
from transformers.data.data_collator import _torch_collate_batch
import numpy as np
import numpy.core.defchararray as nchar
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import json

def small_vocab():
    with open('/data1/gl/project/program/UHC/small_vocab/small_vocab_idxs.json','r',encoding='utf-8') as f:
        vocabs=json.load(f)

    vocabs=set(vocabs)
    vocabs.update([0,1,2,3,250001])
    vocabs=list(vocabs)
    vocabs.sort()
    vocabs_dict={word:i for i,word in enumerate(vocabs)}
    return vocabs_dict

vocabs=small_vocab()


class DataCollatorForWWM_mlm_entity(DataCollatorForWholeWordMask):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
            attention_mask = [e['attention_mask'] for e in examples]
            labels=[]
            for e in examples:
                labels.append(e['small_vocab_labels'])
                assert len(e['input_ids'])==len(e['small_vocab_labels'])
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]
        # student_input_ids=[[en_dict[id_] for id_ in ids] for ids in input_ids]
        # examples['student_input_ids']=input_ids
        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        batch_attention_mask = _torch_collate_batch(attention_mask, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        batch_labels = _torch_collate_batch(labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        mask_labels = []
        for e in examples:
            ref_tokens=e['tokens']
            lang=e['lang']
            entities=e['entities']
            if lang=='en':
                mask_labels.append(self._whole_word_mask_en(ref_tokens,entities))
            else:
                mask_labels.append(self._whole_word_mask_zh(ref_tokens,entities))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask,batch_labels)
        return {"input_ids": inputs, "attention_mask":batch_attention_mask, "labels": labels}
    
    def _whole_word_mask_en(self, input_tokens: List[str], entities,max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
        #     warnings.warn(
        #         "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
        #         "Please refer to the documentation for more information."
        #     )

        ent_indexes=[]
        exclude_indexes=set()
        for entity in entities:
            if entity['span'][0]<max_predictions:
                ent_indexes.append(entity['span'])
                exclude_indexes.update(list(range(entity['span'][0],entity['span'][-1]+1)))
            
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "<s>" or token == "</s>":
                continue
            if i in exclude_indexes:
                continue
            if len(cand_indexes) >= 1 and not token.startswith("▁"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        random.shuffle(ent_indexes)

        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict//2:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict//2:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)
        
        for index_set in ent_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels
    
    def _whole_word_mask_zh(self, input_tokens: List[str],entities, max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
        #     warnings.warn(
        #         "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
        #         "Please refer to the documentation for more information."
        #     )

        ent_indexes=[]
        exclude_indexes=set()
        for entity in entities:
            if entity['span'][0]<max_predictions:
                ent_indexes.append(entity['span'])
                exclude_indexes.update(list(range(entity['span'][0],entity['span'][-1]+1)))
            
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "<s>" or token == "</s>":
                continue
            if i in exclude_indexes:
                continue
            cand_indexes.append([i])

        random.shuffle(cand_indexes)
        random.shuffle(ent_indexes)

        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict//2:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict//2:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)
        
        for index_set in ent_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels
    
    def torch_mask_tokens(self, inputs: Any, mask_labels: Any,labels=None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = labels.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            # padding_mask = labels.eq(vocabs[self.tokenizer.pad_token_id])
            padding_mask = labels.eq(vocabs[self.tokenizer.pad_token_id])

            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = vocabs[self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)]
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)


        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        random_words = torch.randint(len(vocabs), labels.shape, dtype=torch.long)

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
