GPU="2,3,4,5"
MODEL=./scale_xlmr
TOKENIZER_NAME=xlm-roberta-base
LR=1e-4

for seed in 1 # 2 3 4 5 
do
 

OUTPUT_DIR=result_uhc_plm/xlmr-$seed-$LR-mlm-entity-mask-total

CUDA_VISIBLE_DEVICES=$GPU deepspeed --master_port 29547 --num_gpus=4 run_uhc_mlm.py --fp16 --deepspeed ./scripts/config.json \
  --student_model_name_or_path $MODEL  \
  --seed $seed  --en_train_file ../process_data/data/examples \
  --zh_train_file ../process_data/data/examples \
  --do_eval --do_train \
  --data_ratio 1 \
  --max_seq_length 512 \
  --model_type roberta \
  --tokenizer_name $TOKENIZER_NAME \
  --per_device_train_batch_size 20 \
  --per_device_eval_batch_size 20 \
  --gradient_accumulation_steps 16 \
  --save_total_limit 1 \
  --metric_for_best_model accuracy \
  --load_best_model_at_end \
  --save_strategy steps \
  --save_steps 500 \
  --evaluation_strategy  steps  \
  --logging_steps 500  \
  --learning_rate $LR \
  --max_steps 150000 \
  --warmup_steps 10000 \
  --output_dir $OUTPUT_DIR 
done  

