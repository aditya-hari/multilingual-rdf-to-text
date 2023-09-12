from collections import Counter
import re
import json 
import random 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--instance_transitive_en', type=str)
parser.add_argument('--labels_en', type=str)
parser.add_argument('--subject_set_labels_save', type=str)
parser.add_argument('--mappingbased_en', type=str)
parser.add_argument('--mappingbased_literals_en', type=str)

args = parser.parse_args()
instance_transitive_en_path = args.instance_transitive_en
labels_en_path = args.labels_en
subject_set_labels_save_path = args.subject_set_labels_save
mappingbased_en_path = args.mappingbased_en
mappingbased_literals_en_path = args.mappingbased_literals_en

subject_set = set() 
labels_map = {} 
subject_set_labels = {} 
subject_properties = {} 

with open(instance_transitive_en_path , 'r') as f:
    for i, line in enumerate(f):
        if(i%10000000 == 0):
            print(i)
        if(len(re.findall(r'__\d+', line.split()[0]))!=0):
            continue 
        subject_set.add(line.split()[0])

with(open(labels_en_path, 'r')) as f:
    for i, line in enumerate(f):
        if(i == 0 or i == 16233168):
            continue 
        if(i%1000000 == 0):
            print(i)
        tokens = line.split("<http://www.w3.org/2000/01/rdf-schema#label>")
        labels_map[tokens[0].strip()] = tokens[1].strip().split('@')[0][1:-1]

for subject in subject_set:
    if(subject in labels_map):
        new_name = labels_map[subject]
    else:
        if(len(re.findall(r'__\d+', subject))!=0):
            continue 
        new_name = re.sub('_', ' ',subject.split('/')[-1])
    if("List of" in new_name or new_name[-6:] == 'squads'):
        continue 
    subject_set_labels[subject] = new_name

with open(subject_set_labels_save_path, 'w') as f:
    for key, value in subject_set_labels.items():
        f.write(json.dumps({key[1:-1]:value}) + '\n')

with open(mappingbased_en_path, 'r') as f:
    for i, line in enumerate(f):
        if(i%2500000 == 0): 
            print(i)
        if(line.split()[0] in subject_set_labels):
            #name = subject_set_labels[line.split()[0]]
            name = line.split()[0][1:-1]
            if(len(name) == 0 or name[0] == '.'):
                continue 
            if(name not in subject_properties):
                subject_properties[name] = {'properties': {}, 'reverse_properties': {}}
            property_name = line.split()[1].split('/')[-1][:-1]
            if("#" in property_name):
                property_name = property_name.split("#")[1]
            if(property_name == 'seeAlso'):
                continue 
            object_name_ = line.split()[2]
            if(object_name_ in subject_set_labels):
                object_name = subject_set_labels[object_name_]
                # if(len(re.findall(r'__\d+', object_name))!=0):
                #     stations.write(object_name+'\n')
                #     continue
            else:
                # if(len(re.findall(r'__\d+', object_name_))!=0):
                #     stations.write(object_name_+'\n')
                #     continue
                object_name = re.sub('_', ' ',object_name_.split('/')[-1])
            if(property_name not in subject_properties[name]['properties']):
                subject_properties[name]['properties'][property_name] = []
            #subject_properties[name]['properties'][property_name].append(re.sub('>', '', object_name_))
            subject_properties[name]['properties'][property_name].append(object_name_[1:-1])

with open(mappingbased_literals_en_path, 'r') as f:
    for i, line in enumerate(f):
        if(i%2500000 == 0): 
            print(i)
        if(line.split()[0] in subject_set_labels):
            # name = subject_set_labels[line.split()[0]]
            name = line.split()[0][1:-1]
            if(len(name) == 0 or name[0] == '.'):
                continue 
            if(name not in subject_properties):
                subject_properties[name] = {'properties': {}, 'reverse_properties': {}}
            property_name = line.split()[1].split('/')[-1][:-1]
            if("#" in property_name):
                property_name = property_name.split("#")[-1]
            if(property_name == 'seeAlso'):
                continue 
            object_name = line.split('"')[1]
            if(property_name not in subject_properties[name]['properties']):
                subject_properties[name]['properties'][property_name] = []
            subject_properties[name]['properties'][property_name].append(re.sub('>', '', f'{object_name}#literal'))

with open(mappingbased_en_path, 'r') as f:
    for i, line in enumerate(f):
        if(i%2500000 == 0): 
            print(i)
        if(line.split()[0] in subject_set_labels):
            #name = subject_set_labels[line.split()[0]]
            name = line.split()[0][1:-1]
            if(len(name) == 0 or name[0] == '.'):
                continue 
            property_name = line.split()[1].split('/')[-1][:-1]
            if("#" in property_name):
                property_name = property_name.split("#")[1]
            if(property_name == 'seeAlso'):
                continue 
            object_name_ = line.split()[2]
            if(object_name_ in subject_set_labels):
                object_name = subject_set_labels[object_name_]
            else:
                if(len(re.findall(r'__\d+', object_name_))!=0):
                    continue 
                object_name = re.sub('_', ' ',object_name_.split('/')[-1])
            object_name = re.sub('>', '', object_name)
            if(object_name_[1:-1] in subject_properties):
                if(property_name not in subject_properties[object_name_[1:-1]]['reverse_properties']):
                    subject_properties[object_name_[1:-1]]['reverse_properties'][property_name] = []
                subject_properties[object_name_[1:-1]]['reverse_properties'][property_name].append(name)

with(open("ontology_props.jsonl", "w")) as f:
    for key, value in subject_properties.items():
        dict_ = {'name': re.sub('>', '', key), 'properties': value}
        f.write(json.dumps(dict_, ensure_ascii=False) + "\n")

with(open('subjects_list.txt', 'w')) as f:
    for key, value in subject_properties.items():
        f.write(re.sub('>', '', key) + "\n")