from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

#ru_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='eng_Latn', tgt_lang='rus_Cyrl')
br_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='eng_Latn', tgt_lang='bre_Latn')
ga_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='eng_Latn', tgt_lang='gle_Latn')
cy_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='eng_Latn', tgt_lang='cym_Latn')
ml_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='eng_Latn', tgt_lang='mlt_Latn')

new_tokens = ["bre_Latn"]
br_tokenizer.add_tokens(list(new_tokens))

tokenizer_map = {
    'br': br_tokenizer,
    #'ru': ru_tokenizer,
    'ga': ga_tokenizer,
    'cy': cy_tokenizer,
    'mt': ml_tokenizer
}

src_txt = []
tgt_txt = []
langs = []

for lang in ['br', 'cy', 'ga', 'mt']:
  data_file = open(f"{lang}_pairs", 'r').readlines()
  data = [line.strip().split('\t') for line in data_file]
  lang_src_txt = [line[0] for line in data]
  lang_tgt_txt = [line[1] for line in data]
  lang_langs = [lang for _ in range(len(lang_tgt_txt))]

  src_txt.extend(lang_src_txt)
  tgt_txt.extend(lang_tgt_txt)
  langs.extend(lang_langs)

all_triples = list(zip(src_txt, tgt_txt, langs))
train_all, test_all = train_test_split(all_triples, test_size=0.2, random_state=42)

train_src = [i[0] for i in train_all]
train_tgt = [i[1] for i in train_all]
train_langs = [i[2] for i in train_all]
val_src = [i[0] for i in test_all]
val_tgt = [i[1] for i in test_all]
val_langs = [i[2] for i in test_all]

class Text2TextDataset(Dataset):
    def __init__(self, inputs, targets, langs, tokenizer_dict):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer_dict = tokenizer_dict
        self.langs = langs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_text = self.inputs[index]
        target_text = self.targets[index]
        lang = self.langs[index]
        input_encoding = self.tokenizer_dict[lang](input_text, text_target=target_text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = input_encoding["labels"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataset = Text2TextDataset(train_src, train_tgt, train_langs, tokenizer_map)
val_dataset = Text2TextDataset(val_src, val_tgt, val_langs, tokenizer_map)

config_dict = {"TRAIN_BATCH_SIZE": 1,
            "VALID_BATCH_SIZE": 1,
            "LEARNING_RATE":1e-5,
            "CLASS_WEIGHTS": 0,
            "EPOCHS": 4,
            "WT_DECAY":0}

training_args = TrainingArguments(
    output_dir='all_langs_new',
    num_train_epochs=config_dict['EPOCHS'],
    per_device_train_batch_size=config_dict["TRAIN_BATCH_SIZE"],
    per_device_eval_batch_size=config_dict["VALID_BATCH_SIZE"],
    warmup_steps=0,
    weight_decay=config_dict["WT_DECAY"],
    logging_dir='translations',
    logging_steps=250,
    save_strategy='epoch',
    save_total_limit=1,
    evaluation_strategy="epoch",
    learning_rate = config_dict["LEARNING_RATE"],
    metric_for_best_model = 'eval_loss',
    load_best_model_at_end = True,
    #lr_scheduler_type="constant",
)

trainer = Trainer(
    model=model,
    args=training_args,
    #tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()