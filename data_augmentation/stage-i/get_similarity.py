from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.util import ngrams 
from nltk.tokenize import sent_tokenize
from collections import Counter
import regex as re 
from numpy import dot
from numpy.linalg import norm
import numpy as np 
import json 
import random 
import spacy 
import glob 
import tqdm 
import sys
import argparse 
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch 

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings(sents, tokenizer, model, batch_size=8):
    sents_batched = [sents[i:i + batch_size] for i in range(0, len(sents), batch_size)]
    sent_embeddings = []
    for src in sents_batched:
        sent_tokens = tokenizer(src, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
        with(torch.no_grad()):
            sent_output = model(**sent_tokens)
            sent_embedding = mean_pooling(sent_output, sent_tokens['attention_mask'])
            sent_embeddings.append(sent_embedding)
    return torch.cat(sent_embeddings, dim=0)

# Function to compute average embedding similarity between every pair of sentences in two lists of sentences
def get_semantic_similarity(sent_list2, sent_list1):
    sent1_embeddings = get_embeddings(sent_list1, sbert_tokenizer, sbert_model)
    sent2_embeddings = get_embeddings(sent_list2, rdf_tokenizer, rdf_model)
    norm_sent1 = torch.norm(sent1_embeddings, dim=1, keepdim=True)
    norm_sent2 = torch.norm(sent2_embeddings, dim=1, keepdim=True)
    sim_mat = torch.mm(sent1_embeddings, sent2_embeddings.transpose(0, 1)) / torch.mm(norm_sent1, norm_sent2.transpose(0, 1))
    # This step ensures that the similarity values are between 0 and 1 (inclusive)
    sim_mat = torch.clip(sim_mat, 0.0, 1.0)
    return sim_mat

# Function to compute average TF-IDF similarity between every pair of sentences in two lists of sentences
def get_similarity(sent_list1, sent_list2):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    sent_list1 = [' '.join([i for i in pattern.split(sent)]) for sent in sent_list1]
    sent_list2 = [' '.join([i for i in pattern.split(sent)]) for sent in sent_list2]
    X1 = vectorizer.fit_transform(sent_list1)
    X2 = vectorizer.transform(sent_list2)
    sim_mat = np.zeros((len(sent_list1), len(sent_list2)))
    for i in range(len(sent_list1)):
        for j in range(len(sent_list2)):
            if(norm(X1[i].toarray()[0])*norm(X2[j].toarray()[0]) == 0):
                sim_mat[i][j] = 0
            else:
                sim_mat[i][j] = dot(X1[i].toarray()[0], X2[j].toarray()[0])/(norm(X1[i].toarray()[0])*norm(X2[j].toarray()[0]))
    return sim_mat 

parser = argparse.ArgumentParser()
parser.add_argument('--candidates_path', type=str)
parser.add_argument('--langs', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--sim_type', type=str)

args = parser.parse_args()
candidates_path = args.candidates_path
langs = args.langs.split(',')
save_dir = args.save_dir
sim_type = args.sim_type

sim_function = get_semantic_similarity if sim_type == 'semantic' else get_similarity

sbert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
sbert_model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1").to('cuda')

rdf_model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to('cuda')
rdf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
rdf_tokenizer.add_special_tokens({'additional_special_tokens': ['<TSP>']})
rdf_model.resize_token_embeddings(len(rdf_tokenizer))

for param in sbert_model.parameters():
    param.requires_grad = False
for param in rdf_model.parameters():
    param.requires_grad = False

candidates = json.load(open(candidates_path, 'r'))
# regex pattern to split at camel case
pattern = re.compile(r'(?<!^)(?=[A-Z])')



keys = sorted(list(candidates.keys()))
pb = tqdm.tqdm(total=len(keys))
filtered_candidates = {} 

# if(end != -1):
#     pb = tqdm.tqdm(total=len(keys[start:end]))
#     slice_ = keys[start:end]
# else:
#     pb = tqdm.tqdm(total=len(keys[start:]))
#     slice_ = keys[start:]

for key in keys:
    pb.update(1)
    try:
        all_prop_strs = [] 
        props_filtered = [prop for prop in candidates[key]['properties'] if('' not in prop)]
        for prop in props_filtered:
                all_prop_strs.append(' | '.join(prop))
        if('en_text' in candidates[key] and len(candidates[key]['en_text']) != 0):
            en_similarity_mat = sim_function(all_prop_strs, candidates[key]['en_text']).cpu()
            en_above_thresh = np.where(en_similarity_mat > 0)
            if(len(en_above_thresh[1]) != 0):
                # en_retained_props = [[] for _ in range(len(candidates[key]['en_text']))]
                # for sent_idx, prop_idx in zip(en_above_thresh[1], en_above_thresh[0]):
                #     en_retained_props[sent_idx].append(props_filtered[prop_idx])
                if(key not in filtered_candidates):
                    filtered_candidates[key] = candidates[key]
                # filtered_candidates[key]['en_text'] = candidates[key]['en_text']
                # filtered_candidates[key]['en_retained_props'] = en_retained_props
                filtered_candidates[key][f'en_{sim_type[:3]}_mat'] = en_similarity_mat.tolist()
                
        for lang in langs:
            if(lang in candidates[key] and len(candidates[key][lang]) != 0):
                similarity_mat = sim_function(all_prop_strs, candidates[key][lang])
                above_thresh = np.where(similarity_mat > 0)
                if(len(above_thresh[1]) != 0):
                    # retained_props = [[] for _ in range(len(candidates[key][lang]))]
                    # for sent_idx, prop_idx in zip(above_thresh[1], above_thresh[0]):
                    #     retained_props[sent_idx].append(props_filtered[prop_idx])
                    if(key not in filtered_candidates):
                        filtered_candidates[key] = candidates[key]
                    # filtered_candidates[key][lang] = candidates[key][lang]
                    # filtered_candidates[key][lang + '_retained_props'] = retained_props
                    filtered_candidates[key][lang + f'_{sim_type[:3]}_mat'] = similarity_mat.tolist()

    except Exception as e:
        print(e)
        continue
    
with(open(f'{save_dir}/prop_candidates_{sim_type}.json', 'w')) as f:
    json.dump(filtered_candidates, f)
