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
import tqdm 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--langs', type=str)
parser.add_argument('--abstract_path', type=str)
parser.add_argument('--subject_set_labels', type=str)
parser.add_argument('--ontology_props', type=str)
parser.add_argument('--instance_transitive_en', type=str)
parser.add_argument('--save_name', type=str)

args = parser.parse_args()
langs = args.langs.split(',')
abstract_path = args.abstract_path
subject_set_labels = args.subject_set_labels
ontology_props = args.ontology_props
instance_transitive_en = args.instance_transitive_en
save_name = args.save_name

label_map = {} 
candidates = {} 
wanted_types = ["Place", "Building", "Infrastructure", "Person", "Organization", "Organisation", "Work"]
langs = langs.split(',')

def get_prop_name(item):
    if("#literal" in item):
        return item.split('#literal')[0], True
    if(item in label_map):
        tail_name = label_map[item]
    else:
        tail_name = re.sub('_', ' ',item.split('/')[-1])
    return tail_name, False

def merge_sentences_with_open_brackets(sentences):
    merged_sentences = []
    current_sentence = ""

    for sentence in sentences:
        current_sentence += sentence
        open_count = current_sentence.count("(")
        close_count = current_sentence.count(")")

        if open_count == close_count:
            merged_sentences.append(current_sentence)
            current_sentence = ""

    return merged_sentences

with open(subject_set_labels, 'r') as f:
    for line in f:
        line = json.loads(line)
        label_map.update(line)

prop_dict = {} 
with open(ontology_props, 'r') as f:
    for line in f:
        line = json.loads(line)
        prop_dict.update({line['name']: line['properties']})

type_dict = {} 
with open(instance_transitive_en, 'r') as f:
    for line in f:
        line = line.split()
        if(line[0] not in type_dict):
            type_dict[line[0]] = set() 
        type_dict[line[0]].add(line[2].split('/')[-1])

for lang in langs:
    count = 0 
    print(lang)
    tokenizer = spacy.load('xx_sent_ud_sm')
    tokenizer.add_pipe('sentencizer')
    with(open(f'{abstract_path}/{lang}.jsonl', 'r', encoding='utf-8')) as f:
        pb = tqdm.tqdm(total=1000000)
        for i, line in enumerate(f):
            pb.update(1)
            # if(i%100000) == 0:
            #     print(i)
            item = json.loads(line)
            rsc = item['resource']
            txt = item['text']
            found = False 
            # all_types.update(type_dict[f'<{rsc}>'])
            for t in wanted_types:
                if(f'{t}>' in type_dict[f'<{rsc}>']):
                    found = True 
            
            if(not(found)):
                continue  

            name = get_prop_name(rsc)[0]
            if(name not in candidates):
                candidates[name] = {}

                properties = [] 
                fw_props = prop_dict[rsc]['properties']
                for prop in fw_props:
                    for item in fw_props[prop]:
                        item_name = get_prop_name(item)[0]
                        properties.append((name, prop, item_name))
                
                rv_props = prop_dict[rsc]['reverse_properties']
                for prop in rv_props:
                    for item in rv_props[prop][:3]:
                        item_name = get_prop_name(item)[0]
                        properties.append((item_name, prop, name))
                
                candidates[name]['properties'] = properties

            doc = tokenizer(txt)
            sents = [sent.text for sent in doc.sents]
            sents = merge_sentences_with_open_brackets(sents)
                
            filtered_sents = []

            if(lang!='en'):
                merged_sents = []
                merge_next =  False
                for sent in sents:
                    if(re.search(r'\d+\.$', sent)):
                        merge_next = True
                        merged_sents.append(sent)
                    else:
                        if(merge_next):
                            merged_sents[-1] = merged_sents[-1] + ' ' + sent
                            merge_next = False
                        else:
                            merged_sents.append(sent)
            else:
                merged_sents = sents
            
            for sent in merged_sents:
                if(len(sent.split()) > 5 and len(sent.split()) < 250):
                    filtered_sents.append(sent)
            candidates[name][f'{lang}_text'] = filtered_sents
    
with open(save_name, 'w') as f:
    json.dump(candidates, f)