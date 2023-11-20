# KBioXLM is based on XLM-R, so type needs to be set 'roberta' and load our model requires setting hyperparameters 'reset_weights'
```
export LR=2e-5
for SEED in 42
do

python main.py \
    --do_train \
    --data_folder ./datasets/ADE/eng_to_zh \
    --pretrained_dir ngwlh/KBioXLM \
    --result_filepath ./results/KBioXLM_model-$SEED-$LR.json \
    --max_position_embeddings 512 \
    --output_dir ./ckpts/KBioXLM-$SEED-$LR \
    --train_bs 16 \
    --type roberta \
    --do_lower_case \
    --lr $LR \
    --max_epochs 100 \
    --task ade \
    --seed $SEED \

done

```

### pretrained_dir: Biomedical Model storage location
### output_dir: Fine tuned model storage path
### result_filepath: The path to the storage file for the F1 value and other evaluations
### data_folder: Dataset Path
