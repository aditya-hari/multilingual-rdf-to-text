from transformers import AutoTokenizer, AutoModel, pipeline 
import torch
from torch.utils.data import DataLoader, Dataset
import random 
import tqdm 
import sys
import numpy as np 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--contrastive_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--negative_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
args = parser.parse_args()

model_name = args.model_name
contrastive_path = args.contrastive_path
data_path = args.data_path
negative_path = args.negative_path
save_path = args.save_path

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

sbert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
sbert_model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1").to('cuda')

rdf_model = AutoModel.from_pretrained(model_name).to('cuda')
rdf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
rdf_tokenizer.add_special_tokens({'additional_special_tokens': ['<TSP>']})
rdf_model.resize_token_embeddings(len(rdf_tokenizer))
rdf_model.load_state_dict(torch.load(contrastive_path))

for param in sbert_model.parameters():
    param.requires_grad = False

batch_size = 8

eval_src = open(f'{data_path}/eval_src', 'r').readlines()
eval_tgt = open(f'{data_path}/eval_tgt', 'r').readlines()

eval_src_batched = [eval_src[i:i + batch_size] for i in range(0, len(eval_src), batch_size)]
eval_tgt_batched = [eval_tgt[i:i + batch_size] for i in range(0, len(eval_tgt), batch_size)]

negative_pairs = open(negative_path).readlines()
neg_src = [p.split('\t')[0] for p in negative_pairs]
neg_tgt = [p.split('\t')[1] for p in negative_pairs]

neg_src_batched = [neg_src[i:i + batch_size] for i in range(0, len(neg_src), batch_size)]
neg_tgt_batched = [neg_tgt[i:i + batch_size] for i in range(0, len(neg_tgt), batch_size)]

similarities_neg = [] 
pb = tqdm.tqdm(total=len(neg_src_batched))
for src, tgt in zip(neg_src_batched, neg_tgt_batched):
    pb.update(1)
    src_tokens = rdf_tokenizer(src, padding=True, truncation=True, max_length=256, return_tensors="pt").to('cuda')
    tgt_tokens = sbert_tokenizer(tgt, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')

    with(torch.no_grad()):
        src_output = rdf_model(**src_tokens)
        src_embedding = mean_pooling(src_output, src_tokens['attention_mask'])

        tgt_output = sbert_model(**tgt_tokens)
        tgt_embedding = mean_pooling(tgt_output, tgt_tokens['attention_mask'].to('cuda'))
    
    similarity = torch.cosine_similarity(src_embedding, tgt_embedding, dim=1)
    similarities_neg.extend(similarity.tolist())

with(open(f'{save_path}/negative_scores', 'w')) as f:
    f.write('\n'.join([str(s) for s in similarities_neg]))

similarities_pos = [] 
pb = tqdm.tqdm(total=len(eval_src_batched))
for src, tgt in zip(eval_src_batched, eval_tgt_batched):
    pb.update(1)
    src_tokens = rdf_tokenizer(src, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
    tgt_tokens = sbert_tokenizer(tgt, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')

    with(torch.no_grad()):
        src_output = rdf_model(**src_tokens)
        src_embedding = mean_pooling(src_output, src_tokens['attention_mask'])

        tgt_output = sbert_model(**tgt_tokens)
        tgt_embedding = mean_pooling(tgt_output, tgt_tokens['attention_mask'].to('cuda'))
    
    similarity = torch.cosine_similarity(src_embedding, tgt_embedding, dim=1)
    similarities_pos.extend(similarity.tolist())

with(open(f'{save_path}/positive_scores', 'w')) as f:
    f.write('\n'.join([str(s) for s in similarities_pos]))