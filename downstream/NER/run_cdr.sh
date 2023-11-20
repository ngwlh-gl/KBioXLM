
export CUDA_VISIBLE_DEVICES=7

export LR=3e-5
export SEED=42

export MODEL=ngwlh/KBioXLM

for TASK in cdr
do

for DIRECTION in eng_to_zh
do

python main.py \
    --do_train \
    --data_folder ./datasets/$TASK/$DIRECTION \
    --pretrained_dir ../models/$MODEL \
    --result_filepath ./results/$DIRECTION/$MODEL-$SEED-$LR-$TASK.json \
    --max_position_embeddings 512 \
    --output_dir ./ckpts/$DIRECTION/$MODEL-$SEED-$LR-$TASK \
    --train_bs 16 \
    --type roberta \
    --do_lower_case \
    --lr $LR \
    --max_epochs 100 \
    --task $TASK \
    --seed $SEED \

done
done


