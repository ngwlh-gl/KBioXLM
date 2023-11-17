import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import random
import sys
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
import torch
import numpy as np
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_dataset, load_metric,DatasetDict,load_from_disk,concatenate_datasets
from torch.utils.data import DataLoader
from run_uhc_data import DataCollatorForWWM_en,DataCollatorForWWM_zh,DataCollatorForWWM,DataCollatorForWWM_entity,DataCollatorForWWM_mlm_entity
# from initialize import get_extra_vocab_from_file
import transformers
import warnings
warnings.filterwarnings("ignore")
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    # Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    XLMRobertaTokenizerFast,
    RobertaConfig,RobertaModel,
    DataCollatorForLanguageModeling
)
# from initialize import get_en_zh_dict,get_extra_vocab_from_file,get_extend_tokenizer

from transformers.trainer_utils import get_last_checkpoint, is_main_process

from models.uhc_mlm import UHCBert
from mlm_trainer import mlm_trainer

import math as mt

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    en_train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    zh_train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    pair_data: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    data_ratio: Optional[float] = field(
        default=1, metadata={"help": "data ration of english and chinese."}
    )
    
    # validation_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the validation data."}
    # )
    # test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    model_type: Optional[str] = field(
        default='bert', metadata={"help": "bert or roberta."}
    )
    few_shot_k: Optional[int] = field(default=-1,
                                      metadata={"help": "Number of instance for training of each class"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    student_model_name_or_path: str = field(default='bert-base-uncased',
                                            metadata={
                                                "help": "Path to the pre-trained lm"}
                                            )
    t1_model_name_or_path: Optional[str] = field(default=None,
                                                 metadata={
                                                     "help": "Path to pretrained bert model or model identifier from huggingface.co/models"}
                                                 )
    t2_model_name_or_path: Optional[str] = field(default=None,
                                                 metadata={
                                                     "help": "Path to pretrained bert model or model identifier from huggingface.co/models"}
                                                 )
    rec_alpha: Optional[float] = field(default=1.0, metadata={"help": "original rec loss on the labeled data"})
    almal_alpha: Optional[float] = field(default=1.0, metadata={"help": "original almal loss on the labeled data"})
    align_number: Optional[int] = field(default=-1, metadata={"help": "How many layers to align for almal block"})

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resume_checkpoint: str = field(default=None,
                                            metadata={
                                                "help": "resume from checkpoint"}
                                            )

    temperature: Optional[float] = field(default=1.0, metadata={"help": "KD distillation kl temperature"})
    teacher_paths: Optional[str] = field(default=None, metadata={"help": "teacher paths, split by ; "})
    teacher_number: Optional[int] = field(default=2, metadata={"help": "teacher number"})
    eval_strategy: Optional[str] = field(default='student', metadata={"help": "eval strategy of model"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.zh_train_file:
        zh_files=data_args.zh_train_file.split(';')
        zh_corpus = DatasetDict() 
        zh_corpus_lst=[]
        for file in zh_files:
            ds_zh= load_from_disk(file)
            zh_corpus_lst.append(ds_zh['train'])
        zh_corpus['train'] = concatenate_datasets(zh_corpus_lst).shuffle(seed=42)
        train_dataset_zh, validation_dataset_zh= zh_corpus['train'].train_test_split(test_size=0.01,seed=42).values()
        ds_zh = DatasetDict({"train":train_dataset_zh,"validation":validation_dataset_zh})
        print(ds_zh['train'][0])
        print(ds_zh['train'][1])
        print(ds_zh['validation'][0])
        print(ds_zh['validation'][1])
    def statistics(examples):
        # count=0
        count_entities=0
        count_samples=0
        count_length=0
        bar=tqdm(examples)
        for data in bar:
            count_entities+=len(data['entities'])
            count_length+=len(data['input_ids'])
            count_samples+=1
        print(count_entities/count_samples)
        print(count_length/count_samples)
    if data_args.en_train_file:
        # ds_en = load_from_disk(data_args.en_train_file) 
        en_files=data_args.en_train_file.split(';')
        en_corpus = DatasetDict() 
        en_corpus_lst=[]
        for file in en_files:
            ds_en= load_from_disk(file)
            en_corpus_lst.append(ds_en['train'])
        en_corpus['train'] = concatenate_datasets(zh_corpus_lst).shuffle(seed=42)
        # train_dataset, validation_dataset=ds_en['train'].train_test_split(test_size=100000,seed=42).values()
        print('english statistics')
        # statistics(ds_en['train'])
        train_dataset, validation_dataset=en_corpus['train'].train_test_split(test_size=0.01,seed=42).values()
        
        ds_en = DatasetDict({"train":train_dataset,"validation":validation_dataset})
        print(ds_en['train'][0])
        print(ds_en['train'][1])
        print(ds_en['validation'][0])
        print(ds_en['validation'][1])
    # 中文数据集划分
    # train_dataset_zh, validation_dataset_zh= zh_corpus['train'].train_test_split(test_size=100000,seed=42).values()
        print('english statistics')
        # statistics(zh_corpus['train'])
        print('chinese train data num:{}, validation data num:{}'.format(len(ds_zh['train']),len(ds_zh['validation'])))
        print('english data num:{}, validation data num:{}'.format(len(ds_en['train']),len(ds_en['validation'])))
        total_val_num=len(ds_zh['validation'])+len(ds_en['validation'])
        last_batch_rest_val_num=total_val_num-(mt.floor(total_val_num//training_args.per_device_eval_batch_size))*training_args.per_device_eval_batch_size

        total_train_num=len(ds_zh['train'])+len(ds_en['train'])
        last_batch_rest_train_num=total_train_num-(mt.floor(total_train_num//training_args.per_device_train_batch_size))*training_args.per_device_train_batch_size
        print('------------------------------')
        print('last batch train data num:{}'.format(last_batch_rest_train_num))
        print('last batch validation data num:{}'.format(last_batch_rest_val_num))

    if data_args.pair_data:
        ds_pair = load_from_disk(data_args.pair_data) 
        # train_dataset, validation_dataset=ds_en['train'].train_test_split(test_size=100000,seed=42).values()
        print('pair data statistics')
        # statistics(ds_en['train'])
        train_dataset, validation_dataset=ds_pair['train'].train_test_split(test_size=0.01,seed=42).values()
        
        ds_pair = DatasetDict({"train":train_dataset,"validation":validation_dataset})
        if data_args.en_train_file:
            multi_corpus = DatasetDict() 
            multi_corpus['train'] = concatenate_datasets([ds_en['train'],ds_zh['train'],ds_pair['train']]).shuffle(seed=42)
            multi_corpus['validation'] = concatenate_datasets([ds_en['validation'],ds_zh['validation'],ds_pair['validation']]).shuffle(seed=42)
        else:
            multi_corpus=ds_pair
    else:
        multi_corpus = DatasetDict() 
        multi_corpus['train'] = concatenate_datasets([ds_en['train'],ds_zh['train']]).shuffle(seed=42)
        multi_corpus['validation'] = concatenate_datasets([ds_en['validation'],ds_zh['validation']]).shuffle(seed=42)

    if data_args.model_type=='bert':
        tokenizer=AutoTokenizer.from_pretrained(model_args.student_model_name_or_path)
        student_config = AutoConfig.from_pretrained(model_args.student_model_name_or_path)
        student_config.vocab_size = len(tokenizer)
    else:
        tokenizer=XLMRobertaTokenizerFast.from_pretrained(model_args.tokenizer_name)
        student_config = RobertaConfig.from_pretrained(model_args.student_model_name_or_path)
        student_config.vocab_size = len(tokenizer)

    # trained teacher model
    model = UHCBert(
        model_args.student_model_name_or_path,
        config=student_config,
        model_type=data_args.model_type,
        pair=True if data_args.pair_data else False
    )

    data_collator = DataCollatorForWWM_entity(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")

    metric = load_metric('../../XLMR+Pretraining/pretrain/accuracy')
    def compute_metrics(p: EvalPrediction):
        preds = p[0].reshape(-1)
        label_ids = p[1].reshape(-1)
        predictions = []
        labels = []
        for pred, label in zip(preds, label_ids):
            if label.item() == -100:
                continue
            predictions.append(pred)
            labels.append(label)
        result = metric.compute(predictions=predictions, references=labels)
        return result
    # Initialize our Trainer
    trainer = mlm_trainer(
        model=model,
        args=training_args,
        train_dataset=multi_corpus['train'],
        eval_dataset=multi_corpus['validation'] if training_args.do_eval else None,
        # train_dataset=ds_en['train'],
        # eval_dataset=ds_en['validation'] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=20, early_stopping_threshold=0.05)],
        # deepspeed="/data1/gl/project/gl/UHC-small/scripts/config.json"
        # resume_from_checkpoint=os.path.join(training_args.output_dir,'checkpoint-'+model_args.resume_checkpoint)
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Training
    if training_args.do_train:
        if training_args.resume_from_checkpoint:
            checkpoint = os.path.join(training_args.output_dir,'checkpoint-'+model_args.resume_checkpoint)
        else:
            checkpoint=None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        # max_eval_samples = len(ds_en['validation'])+len(ds_zh['validation'])
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            import math
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
