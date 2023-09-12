import json 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--langs', type=str)
parser.add_argument('--abstract_path', type=str)
parser.add_argument('--global_ids', type=str)
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()
langs = args.langs.split(',')
abstract_path = args.abstract_path
global_ids = args.global_ids
save_dir = args.save_dir

abstracts = {lang: {} for lang in langs}
id_mapping = {} 

with open(global_ids, 'r') as f:
    for i, line in enumerate(f):
        if(i%10000000 == 0):
            print(i)
        tokens = line.split()
        url_domain = tokens[0].split('.')[0].split('/')[-1]
        obj_id = tokens[-1]
        if(url_domain in ['ru', 'cy', 'dbpedia']):
            if(obj_id not in id_mapping):
                id_mapping[obj_id] = {}
            id_mapping[obj_id][url_domain] = tokens[0]

for lang in langs:
    print(lang)
    lang_file = open(f'{save_dir}/abstracts_{lang}.jsonl', 'w')
    with(open(f'{abstract_path}/{lang}.ttl', 'r')) as f:
        for i, line in enumerate(f):
            resource = line.split('<http://dbpedia.org/ontology/abstract>')[0][1:-2]
            if(resource in ids_retained[lang]):
                text = line.split('<http://dbpedia.org/ontology/abstract>')[1].split(f'@{lang}')[0][1:-1]
                lang_file.write(json.dumps({'resource': id_lang_mapping[resource], 'text': text}, ensure_ascii=False) + "\n")
    lang_file.close()