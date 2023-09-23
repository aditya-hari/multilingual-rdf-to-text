import argparse
import random 

parser = argparse.ArgumentParser()
parser.add_argument('--webnlg_data', type=str)
parser.add_argument('--save_name', type=str)

train_src_lines = open(f"{webnlg_data}/train_src", 'r').readlines()
train_tgt_lines = open(f"{webnlg_data}/train_tgt", 'r').readlines()

train_rdfs = {}
for src, tgt in zip(train_src_lines, train_tgt_lines):
    if(src.strip() not in train_rdfs):
        train_rdfs[src.strip()] = []
    train_rdfs[src.strip()].append(tgt.strip())

keys = list(train_rdfs.keys())

positive_pairs = []
for src in train_rdfs:
    for tgt in train_rdfs[src]:
        positive_pairs.append((src, tgt))

negative_pairs = []
pb = tqdm.tqdm(total=len(train_rdfs))
for src in train_rdfs:
    pb.update(1)
    for tgt in train_rdfs[src]:
        non_srcs = random.sample(keys, 31)
        while src not in non_srcs:
            non_srcs = random.sample(keys, 31)
        for non_src in non_srcs:
            negative_pairs.append((src, random.choice(train_rdfs[non_src])))

with(open(f'{save_name}/positive_pairs.txt', 'w')) as f:
    for src, tgt in positive_pairs:
        f.write(src + '\t' + tgt + '\n')

with(open(f'{save_name}/negative_pairs.txt', 'w')) as f:
    for src, tgt in negative_pairs:
        f.write(src + '\t' + tgt + '\n')