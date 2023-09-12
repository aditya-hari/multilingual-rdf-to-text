import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sent_prop_src_path', type=str)
parser.add_argument('--entailments_path', type=str)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()
sent_prop_src_path = args.sent_prop_src_path
entailments_path = args.entailments_path
save_path = args.save_path

sent_prop_src = open('/home2/aditya_hari/gsoc/rdf-to-text/scraping/notebooks/sent_prop_src_new.txt', 'r').readlines()
sent_prop_src = [[i.strip() for i in line.split('\t')] for line in sent_prop_src]

with(open(entailments_path, 'r')) as f:
    for line in f:
        num, sent, prop = line.strip().split('\t')
        sent = sent.strip('"')
        lang = sent_prop_src[int(num)][2]
        if(sent not in entailments[lang]):
            entailments[lang][sent] = set()
        entailments[lang][sent].add(prop)

for lang in entailments:
    sampled = entailments[lang].keys()
    with open(f'{save_path}/{lang}/train_src', 'w') as f:
        for sent in sampled:
            f.write(f'{" <TSP> ".join(entailments[lang][sent])}\n')
    with open(f'{save_path}/{lang}/train_tgt', 'w') as f:
        for sent in sampled:
            f.write(f'{sent}\n')