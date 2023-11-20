
export CUDA_VISIBLE_DEVICES=1

export LR=3e-5
export EPOCH=20
export SEED=42
for MODEL in KBioXLM
do
for BS in 16
do

export MODEL_PATH=ngwlh/KBioXLM

# task=GAD_hf_zh_en
# datadir=../data/seqcls/$task
# outdir=runs/$task/$MODEL-$LR-$EPOCH-$SEED-$BS
# mkdir -p $outdir
# python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
#   --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test_en.json \
#   --do_train --do_eval --do_predict --metric_name PRF1 \
#   --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
#   --learning_rate $LR --num_train_epochs $EPOCH --max_seq_length 256 \
#   --save_strategy steps --evaluation_strategy steps --eval_steps 50 --save_steps 50 \
#   --metric_for_best_model F1 --output_dir $outdir --overwrite_output_dir \
#   --load_best_model_at_end --save_total_limit 1 --logging_steps 50 --overwrite_cache --seed $SEED

task=GAD_hf_en_zh
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL-$LR-$EPOCH-$SEED-$BS
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test_zh.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate $LR --num_train_epochs $EPOCH --max_seq_length 256 \
  --save_strategy steps --evaluation_strategy steps --eval_steps 50 --save_steps 50 \
  --metric_for_best_model F1 --output_dir $outdir --overwrite_output_dir \
  --load_best_model_at_end --save_total_limit 1 --logging_steps 50 --overwrite_cache --seed $SEED

done
done
