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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
parser.add_argument("--synthetic_data_path", type=str)
parser.add_argument("--trusted_data_path", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--model_checkpoint", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--lang", type=str)
parser.add_argument("--synthetic_scores_path", type=str)
parser.add_argument("--denoised_scores_path", type=str)

args = parser.parse_args()
synthetic_data_path = args.synthetic_data_path
trusted_data_path = args.trusted_data_path
epochs = args.epochs
actual_model_old = args.model_checkpoint
save_path = args.save_path
lang = args.lang
synthetic_scores_path = args.synthetic_scores_path
denoised_scores_path = args.denoised_scores_path

print(data, lang)
syn_scores = [float(i) for i in open(synthetic_scores_path).readlines()]
denoised_scores = [float(i) for i in open(denoised_scores_path).readlines()]
score_diff = np.array(denoised_scores) - np.array(syn_scores)

wandb.init()
pattern = re.compile(r'(?<!^)(?=[A-Z])')
config_dict = {"TRAIN_BATCH_SIZE": 2, 
            "VALID_BATCH_SIZE": 2, 
            "LEARNING_RATE":1e-5, 
            "CLASS_WEIGHTS": 0, 
            "EPOCHS": epochs, 
            "WT_DECAY":0,
            "GRADIENT_ACCUMULATION_STEPS": 2,
            "EVAL_STEPS": 10000,
            "SAVE_STEPS": 10000,
            }

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
model.to(device)

lang_files = glob(f"{data}/*")

syn_src = open(f'{synthetic_data_path}/{lang}/train_src', 'r').readlines()
syn_tgt = open(f'{synthetic_data_path}/{lang}/train_tgt', 'r').readlines()
trusted_src = open(f'{trusted_data_path}/{lang}/train_src', 'r').readlines()
trusted_tgt = open(f'{trusted_data_path}/{lang}/train_tgt', 'r').readlines()

triples = list(zip(syn_src, syn_tgt, score_diff))
triples = sorted(triples, key=lambda x: x[2], reverse=True)

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

eval_srcs = []
eval_tgts = []
eval_src = open(f'{lang_file}/eval_src', 'r').readlines()        
eval_tgt = open(f'{lang_file}/eval_tgt', 'r').readlines()
eval_srcs.extend([re.sub(r"[ ]{2,}", " " , f"{line}").strip() for line in eval_src])
eval_tgts.extend([line.strip() for line in eval_tgt])
eval_dataset = Text2TextDataset(eval_srcs, eval_tgts, tokenizer)

for epoch in range(epochs):
    total_loss = 0
    print(f"Epoch {epoch+1}")
    if((epoch+1)/3 < 0.33):
        split_end = len(triples)//2
    elif((epoch+1)/3 < 0.66):
        split_end = len(triples)//4
    else:
        split_end = -1 

    srcs = []
    tgts = []

    srcs.extend(trusted_src)
    tgts.extend(trusted_tgt)
    for src, tgt, score in triples:
        if(score > 0):
            srcs.append(src)
            tgts.append(tgt)

    if(split_end != -1):
        non_positives = [i for i in triples if i[2] <= 0][:split_end]
        if(len(non_positives) > 50000):
            sample_indices = np.random.choice(len(non_positives), 50000)
        else:
            sample_indices = np.random.choice(len(non_positives), len(non_positives))
        for i in sample_indices:
            srcs.append(non_positives[i][0])
            tgts.append(non_positives[i][1])
    srcs = [i.strip() for i in srcs]
    tgts = [i.strip() for i in tgts]
    train_dataset = Text2TextDataset(train_srcs, train_tgts, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config_dict["TRAIN_BATCH_SIZE"], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config_dict["VALID_BATCH_SIZE"], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config_dict["LEARNING_RATE"], weight_decay=config_dict["WT_DECAY"])
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss = loss / config_dict["GRADIENT_ACCUMULATION_STEPS"]
        loss.backward()
        if (step % config_dict["GRADIENT_ACCUMULATION_STEPS"] == 0) or (step == len(train_dataloader) - 1):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {total_loss/(step+1)}")

        if(step % config_dict["EVAL_STEPS"] == 0):
            model.eval()
            total_eval_loss = 0
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_eval_loss += loss.detach().float()
            print(f"Eval loss: {total_eval_loss/len(eval_dataloader)}")
            model.train()
        
        if(step % config_dict["SAVE_STEPS"] == 0):
            model.save_pretrained(f"{model_path}/{lang}/epoch_{epoch}_step_{step}")

