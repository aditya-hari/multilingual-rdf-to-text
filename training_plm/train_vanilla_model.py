from glob import glob 
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from datasets import load_dataset
from sacrebleu import BLEU 
import regex as re
import numpy as np
import random 
from tqdm import tqdm
import evaluate 
import argparse 
import sys
import wandb 

lang_codes = {
    "cy": "Welsh",
    "br": "Breton",
    "ga": "Irish",
    "mt": "Maltese",
    "ru": "Russian",
    "de": "German",
    "en": "English"
}

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--model_checkpoint", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--lang", type=str)

args = parser.parse_args()
data = args.data_path
epochs = args.epochs
actual_model_old = args.model_checkpoint
save_path = args.save_path
lang = args.lang

pattern = re.compile(r'(?<!^)(?=[A-Z])')
wandb.init()

config_dict = {"TRAIN_BATCH_SIZE": 2, 
            "VALID_BATCH_SIZE": 2, 
            "LEARNING_RATE":1e-4, 
            "CLASS_WEIGHTS": 0, 
            "EPOCHS": epochs, 
            "WT_DECAY":0}

print(data, epochs, actual_model_old, save_name)

if(lang == 'en'):
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(actual_model_old)
else: 
    model_name = "google/mt5-small"
    model = MT5ForConditionalGeneration.from_pretrained(actual_model_old)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'additional_special_tokens': ['<TSP>']})
model.resize_token_embeddings(len(tokenizer))

lang_files = glob(f"{data}/*")

train_srcs = [] 
train_tgts = []

eval_srcs = [] 
eval_tgts = []

test_srcs = []
test_tgts = []

for lang_file in lang_files:
    lang = lang_file.split("/")[-1]
    if(lang not in [lang]):
        continue
    train_src = open(f'{lang_file}/train_src', 'r').readlines()
    train_tgt = open(f'{lang_file}/train_tgt', 'r').readlines()
    train_srcs.extend([re.sub(r"[ ]{2,}", " " , f"{line}").strip() for line in train_src])
    print(train_srcs[:5])
    train_tgts.extend([line.strip() for line in train_tgt])
    
    eval_src = open(f'{lang_file}/eval_src', 'r').readlines()        
    eval_tgt = open(f'{lang_file}/eval_tgt', 'r').readlines()
    eval_srcs.extend([re.sub(r"[ ]{2,}", " " , f"{line}").strip() for line in eval_src])
    eval_tgts.extend([line.strip() for line in eval_tgt])

class Text2TextDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_text = self.inputs[index]
        target_text = self.targets[index]

        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


train_dataset = Text2TextDataset(train_srcs, train_tgts, tokenizer)
eval_dataset = Text2TextDataset(eval_srcs, eval_tgts, tokenizer)

training_args = TrainingArguments(
    output_dir=save_path,  
    num_train_epochs=config_dict['EPOCHS'],              
    per_device_train_batch_size=config_dict["TRAIN_BATCH_SIZE"],  
    per_device_eval_batch_size=config_dict["VALID_BATCH_SIZE"],   
    warmup_steps=0,                
    weight_decay=config_dict["WT_DECAY"],              
    logging_dir=f'{save_path}/logs',            
    logging_steps=1000   ,
    save_strategy='steps',
    save_total_limit=25,
    evaluation_strategy="steps", 
    eval_steps=10000,
    save_steps=10000,
    learning_rate = config_dict["LEARNING_RATE"],
    metric_for_best_model = 'eval_loss',
    load_best_model_at_end = True,
    gradient_accumulation_steps=2,
    optim='adafactor',
    lr_scheduler_type="linear",
    fp16=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop)],
)

trainer.train()