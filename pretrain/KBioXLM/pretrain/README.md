# The initialization model for this stage can be XLMR+Pretraining, or other models with the same structure, such as scale_ Xlmr

# Pre-trained Run Script
```

GPU="2,3,4,5"
MODEL=./XLMR+Pretraining
TOKENIZER_NAME=xlm-roberta-base
LR=5e-5

for seed in 1 # 2 3 4 5 
do
 

OUTPUT_DIR=result_uhc_plm/xlmr-$seed-$LR-KBioXLM

CUDA_VISIBLE_DEVICES=$GPU deepspeed --master_port 29546 --num_gpus=4 run_uhc_mlm.py --fp16 --deepspeed ./scripts/config.json \
  --student_model_name_or_path $MODEL --task_name $TASK  \
  --seed $seed  --en_train_file "../process_data/data/en_examples" \
  --zh_train_file "../process_data/data/zh_examples" \
  --do_eval --do_train \
  --data_ratio 1 \
  --max_seq_length 512 \
  --model_type roberta \
  --tokenizer_name $TOKENIZER_NAME \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --save_total_limit 1 \
  --metric_for_best_model accuracy \
  --save_strategy steps \
  --save_steps 200 \
  --evaluation_strategy  steps  \
  --logging_steps 200  \
  --learning_rate $LR \
  --num_train_epochs 30 \
  --warmup_steps 5000 \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir 
done 
done 


```

## en_train_file：Pre-trained corpus for entity and fact alignment in English, constructed through process_data
## zh_train_file：Pre-trained corpus for entity and fact alignment in Chinese, constructed through process_data
## pair_data：This is the pre-trained corpus for paragraph alignment
### Note: All three corpora have been post-processed by the tokenize_abstracts.py code in XLMR+Pretraining
