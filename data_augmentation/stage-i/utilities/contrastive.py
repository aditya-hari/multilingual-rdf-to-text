from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
import random 
import tqdm 
import numpy as np 
import wandb 

wandb.init() 

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

sbert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
sbert_model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1").to('cuda')

for param in sbert_model.parameters():
    param.requires_grad = False

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.margin = 0.5
    
    def forward(self, x1, x2, y):
        cos_sim = self.cos(x1, x2)
        loss = torch.mean((1 - y) * torch.pow(cos_sim, 2) + y * torch.pow(torch.nn.functional.relu(self.margin - cos_sim), 2))
        return loss

rdf_tokenizer = sbert_tokenizer
rdf_model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1").to('cuda')

rdf_tokenizer.add_special_tokens({'additional_special_tokens': ['<TSP>']})
rdf_model.resize_token_embeddings(len(rdf_tokenizer))

positive_pairs = [p.split('\t') for p in open("positive_pairs.txt", 'r').readlines()]
negative_pairs = [p.split('\t') for p in open("negative_pairs.txt", 'r').readlines()]

negative_pointer = 0 
training_pairs = []
for positive in positive_pairs:
    training_pairs.append(positive) 
    training_pairs.extend(negative_pairs[negative_pointer: negative_pointer + 7])
    negative_pointer += 31

train_labels = []
for i in range(len(training_pairs)):
    if(i % 8 == 0):
        train_labels.append(1)
    else:
        train_labels.append(0)

class CustomDataset(Dataset):
    def __init__(self, pairs, labels, rdf_tokenizer, sbert_tokenizer):
        self.pairs = pairs
        self.labels = labels
        self.rdf_tokenizer = rdf_tokenizer
        self.sbert_tokenizer = sbert_tokenizer
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rdf, text = self.pairs[idx]
        rdf_encodings = self.rdf_tokenizer(rdf, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        text_encodings = self.sbert_tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')

        return rdf_encodings, text_encodings, torch.tensor(self.labels[idx])

train_dataset = CustomDataset(training_pairs[:33352*8], train_labels[:33352*8], rdf_tokenizer, sbert_tokenizer)
eval_dataset = CustomDataset(training_pairs[33352*8:], train_labels[33352*8:], rdf_tokenizer, sbert_tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

optimizer = torch.optim.Adam(rdf_model.parameters(), lr=1e-6)
loss_fn = ContrastiveLoss()

train_losses = [] 
eval_losses = []

for epoch in range(5):
    pb_train = tqdm.tqdm(total=len(train_dataloader))
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        rdf_encodings, text_encodings, labels = batch
        rdf_encodings = {key: val.to('cuda').squeeze() for key, val in rdf_encodings.items()}
        text_encodings = {key: val.to('cuda').squeeze() for key, val in text_encodings.items()}
        rdf_outputs = rdf_model(**rdf_encodings)
        text_outputs = sbert_model(**text_encodings)

        rdf_embedding = mean_pooling(rdf_outputs, rdf_encodings['attention_mask'])
        text_embedding = mean_pooling(text_outputs, text_encodings['attention_mask'])

        loss = loss_fn(rdf_embedding, text_embedding, labels.to('cuda'))
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

        if(i % 100 == 0):
            wandb.log({"train_loss": np.mean(train_losses)})

        pb_train.set_description("Epoch: {}, Loss: {}".format(epoch, np.mean(train_losses)))
        pb_train.update(1)
    pb_train.close()
    print("Epoch: {}, Train Loss: {}".format(epoch, np.mean(train_losses)))

    pb_eval = tqdm.tqdm(total=len(eval_dataloader))
    for i, batch in enumerate(eval_dataloader):
        rdf_encodings, text_encodings, labels = batch
        rdf_encodings = {key: val.to('cuda').squeeze() for key, val in rdf_encodings.items()}
        text_encodings = {key: val.to('cuda').squeeze() for key, val in text_encodings.items()}
        rdf_outputs = rdf_model(**rdf_encodings)
        text_outputs = sbert_model(**text_encodings)

        rdf_embedding = mean_pooling(rdf_outputs, rdf_encodings['attention_mask'])
        text_embedding = mean_pooling(text_outputs, text_encodings['attention_mask'])

        loss = loss_fn(rdf_embedding, text_embedding, labels.to('cuda'))
        eval_losses.append(loss.item())

        if(i % 100 == 0):
            wandb.log({"eval_loss": np.mean(eval_losses)})
            
        pb_eval.set_description("Epoch: {}, Loss: {}".format(epoch, np.mean(eval_losses)))
        pb_eval.update(1)
    pb_eval.close()
    print("Epoch: {}, Eval Loss: {}".format(epoch, np.mean(eval_losses)))
    torch.save(rdf_model.state_dict(), f'contrastive_model_{epoch}.pt'.format(epoch))
