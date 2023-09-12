import json 
import glob
import tqdm 
import argparse
import numpy as np 
import regex as re 
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--langs', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--semantic_scores', type=str)
parser.add_argument('--syntactic_scores', type=str)
parser.add_argument('--semantic_dict_path', type=str)
parser.add_argument('--syntactic_dict_path', type=str)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()
langs = args.langs.split(',')
model_name = args.model_name
semantic_scores = args.semantic_scores
syntactic_scores = args.syntactic_scores
semantic_dict_path = args.semantic_dict_path
syntactic_dict_path = args.syntactic_dict_path
save_path = args.save_path

model_name = 'all-distilroberta-v1_sbert_not_pt_cont'
positives = [float(i.strip()) for i in open(f'{semantic_scores}/postive_scores', 'r').readlines()]
negatives = [float(i.strip()) for i in open(f'{semantic_scores}/negative_scores', 'r').readlines()]
positives_syn = [float(i.strip()) for i in open(f'{syntactic_scores}/postive_scores', 'r').readlines()]
negatives_syn = [float(i.strip()) for i in open(f'{syntactic_scores}/negative_scores', 'r').readlines()]

positives_2 = list(zip(positives, positive_syn))
negatives_2 = list(zip(negatives, negative_syn))
merged = positives_2 + negatives_2
positive_labels = [1 for i in positives]
negative_labels = [0 for i in negatives]
merged_labels = positive_labels + negative_labels
#X = np.array(merged).reshape(-1, 1)
y = np.array(merged_labels)
X = np.array(merged)
clf = KNeighborsClassifier(3)
# clf = SVC(kernel='rbf')
clf.fit(X, y)

merged_dict = json.load(open(syntactic_dict_path, 'r'))
semantic_dict = json.load(open(semantic_dict_path, 'r'))
for key in semantic_dict:
    if(key not in merged_dict):
        continue
    if('en_sem_mat' in current_dict[key]):
        merged_dict[key]['en_sem_mat'] = current_dict[key]['en_sem_mat']
    for lang in langs:
        if(f'{lang}_sem_mat' in current_dict[key]):
            merged_dict[key][f'{lang}_sem_mat'] = current_dict[key][f'{lang}_sem_mat']

keys = list(merged_dict.keys())

filtered_candidates = {}
pb = tqdm.tqdm(total=len(keys))
ga_count = 0
for key, value in merged_dict.items(): 
    pb.update(1)
    if('en_text' in value and 'en_sim_mat' in value and len(value['en_text']) != 0):
        if(key not in filtered_candidates):
            filtered_candidates[key] = {}
        filtered_candidates[key]['en_text'] = value['en_text']
        en_sim_mat = np.array(value['en_sim_mat'])
        en_sem_mat = np.array(value['en_sem_mat']).reshape(en_sim_mat.shape)
        features = np.stack((en_sim_mat, en_sem_mat), axis=2).reshape(-1, 2)
        predictions = clf.predict(features).reshape(en_sim_mat.shape)
        en_above_thresh = np.where(predictions > 0)
        if(len(en_above_thresh[1]) != 0):
            en_retained_props = [[] for _ in range(len(value['en_text']))]
            for sent_idx, prop_idx in zip(en_above_thresh[1], en_above_thresh[0]):
                if('' not in value['properties'] and value['properties'][prop_idx][0]!=value['properties'][prop_idx][2]):
                    en_retained_props[sent_idx].append(value['properties'][prop_idx])
            filtered_candidates[key]['en_retained_props'] = en_retained_props
    
    for lang in langs:
        if(f'{lang}_text' in value and f'{lang}_sim_mat' in value and len(value[f'{lang}_text']) != 0):
            if(key not in filtered_candidates):
                filtered_candidates[key] = {}
            filtered_candidates[key][f'{lang}_text'] = value[f'{lang}_text']
            lang_sim_mat = np.array(value[f'{lang}_sim_mat'])
            lang_sem_mat = np.array(value[f'{lang}_sem_mat']).reshape(lang_sim_mat.shape)
            features = np.stack((lang_sim_mat, lang_sem_mat), axis=2).reshape(-1, 2)
            predictions = clf.predict(features).reshape(lang_sim_mat.shape)
            lang_above_thresh = np.where(predictions > 0)
            if(len(lang_above_thresh[1]) != 0):
                lang_retained_props = [[] for _ in range(len(value[f'{lang}_text']))]
                for sent_idx, prop_idx in zip(lang_above_thresh[1], lang_above_thresh[0]):
                    if('' not in value['properties'] and value['properties'][prop_idx][0]!=value['properties'][prop_idx][2]):
                        lang_retained_props[sent_idx].append(value['properties'][prop_idx])
                filtered_candidates[key][f'{lang}_retained_props'] = lang_retained_props

keys = list(filtered_candidates.keys())
remove_spaces = lambda x: ' '.join(x.split())

sent_prop_pairs = [] 
sent_prop_src = [] 
for key in keys:
    if('en_retained_props' in filtered_candidates[key]):
        for sent_idx, props in enumerate(filtered_candidates[key]['en_retained_props']):
            for prop in props:
                sent_prop_pairs.append((filtered_candidates[key]['en_text'][sent_idx], ' | '.join(prop), filtered_candidates[key]['en_text'][sent_idx]))
                sent_prop_src.append((key, sent_idx, 'en'))
    
    for lang in langs:
        if(f'{lang}_retained_props' in filtered_candidates[key]):
            for sent_idx, props in enumerate(filtered_candidates[key][f'{lang}_retained_props']):
                for prop in props:
                    sent_prop_pairs.append((filtered_candidates[key][f'{lang}_text'][sent_idx], ' | '.join(prop), filtered_candidates[key][f'{lang}_text'][sent_idx]))
                    sent_prop_src.append((key, sent_idx, lang))

with(open(f'{save_name}/sent_prop_src_new.txt', 'w')) as f:
    for src in sent_prop_src:
        f.write('\t'.join([str(x) for x in src]) + '\n')

with(open(f'{save_name}/sent_prop_pairs.txt', 'w')) as f:
    for pair in sent_prop_pairs:
        f.write('\t'.join(pair) + '\n')