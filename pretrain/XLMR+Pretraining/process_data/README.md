# 1.utils_data 

## [data link](https://pan.baidu.com/s/1RKgoDwW35hlWHQCn_arxDA?pwd=z65t)

## folder description
### cui2sty.json：Medical entities and their labels
### en_to_zh.json：English-Chinese entity pair
### eng_zh_rels.json：Correspondence between Chinese and English relationships
### small_vocab_idxs.json：XLM-R only contains the ID of tokens in both Chinese and English
### stopwords_en.txt：English stop words
### stopwords_zh.txt：Chinese stop words
### zh_to_en.json：Chinese-English entity pair

# 2.Process the JSON file into the form required for model input:
```

python tokenize_abstracts.py

```

### in_file：pre-training data, JSON format: {'text ':'... '}, out_file: The processed data is fed into the model, which includes：input_ids, attention_mask, entities, lang, small_vocab_labels, tokens, labels
### Explanation of each field in the out_file：
```
    input_ids：sequence token id, 
    attention_mask, 
    entities：span of entities in the sequence, 
    lang：The language of the input text, en or zh,, 
    small_vocab_labels：Because our training model mainly targets Chinese and English tokens, we have added a smaller classification layer than the xlm-r vocabulary after the output layer, with a size of 37030. Therefore, we need to map the ID to these 37030 dimensions as the label of the model,
    tokens： tokens of a sequence,
    labels： If there is this field, it represents the relationship between two messages

```

# 3.scale_model
## Reduce the dimension of the output layer of XLM-R from 250002 * 768 to 37030 * 768
```

python scale_model.py

```
