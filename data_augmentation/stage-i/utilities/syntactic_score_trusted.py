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
import argparse

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
parser.add_argument('--negative_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()
negative_path = args.negative_path
data_path = args.data_path
save_path = args.save_path

pattern = re.compile(r'(?<!^)(?=[A-Z])')

neg_pairs = [i.split('\t') for i in open(negative_path, 'r').readlines()]
pos_src = open(f'{data_path}/eval_src', 'r').readlines()
pos_tgt = open(f'{data_path}/eval_tgt', 'r').readlines()
pos_pairs = [[src, tgt] for src, tgt in zip(pos_src, pos_tgt)]

pb = tqdm.tqdm(total=len(pos_pairs))
pos_scores = []
for pair in pos_pairs:
    pb.update(1)
    rdf, sent = pair
    rdf_components = rdf.split('|')
    prop_str = ' '.join([' '.join(pattern.split(re.sub(r'[^\w]+', ' ', i))) for i in rdf_components[:]])
    sim_mat = get_similarity([sent], [prop_str])
    sim_score = sim_mat[0][0]
    pos_scores.append(sim_score)
    
with(open(f'{save_path}/positive_scores', 'w')) as f:
    f.write('\n'.join([str(i) for i in pos_scores]))

pb = tqdm.tqdm(total=len(neg_pairs))
neg_scores = []
for pair in neg_pairs:
    pb.update(1)
    rdf, sent = pair
    rdf_components = rdf.split('|')
    prop_str = ' '.join([' '.join(pattern.split(re.sub(r'[^\w]+', ' ', i))) for i in rdf_components[:]])
    sim_mat = get_similarity([sent], [prop_str])
    sim_score = sim_mat[0][0]
    neg_scores.append(sim_score)

with(open(f'{save_path}/negative_scores', 'w')) as f:
    f.write('\n'.join([str(i) for i in neg_scores]))