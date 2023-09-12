from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
import tqdm 
import time 
import argparse  

parser = argparse.ArgumentParser()
parser.add_argument('--langs', type=str)
parser.add_argument('--translation_save_dir', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--candidates_path', type=str)

args = parser.parse_args()
langs = args.lang.split(',')
translation_save_dir = args.translation_save_dir
save_name = args.save_name
candidates_path = args.candidates_path

model_name = "facebook/nllb-200-distilled-600M"
lang_map = {
    'br': 'bre_Latn',
    'cy': 'cym_Latn',
    'ga': 'gle_Latn',
    'mt': 'mlt_Latn',
    'ru': 'rus_Cyrl'
}

tokenizers = {lang: AutoTokenizer.from_pretrained(model_name, src_lang=lang_map[lang]) for lang in langs}
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
nllb_model.eval()

candidates = json.load(open(candidates_path, 'r'))

all_sents = {lang: [] for lang in langs}
all_sents_src = {lang: [] for lang in langs}

all_props = {lang: [] for lang in langs}
all_props_src = {lang: [] for lang in langs}

word_length = lambda x: len(x.split())  

for key, value in list(candidates.items()):
    props = [' | '.join(prop) for prop in value['properties']]
    lang in langs:
        if(f'{lang}_text' in value):
            all_sents[lang].extend(value[f'{lang}_text'])
            all_props[lang].extend(props)

sents_out = {lang: [] for lang in langs}
props_out = {lang: [] for lang in langs}

for lang in all_sents:
    tok = tokenizers[lang]
    out_file = open(f'{translation_save_dir}/translated_sents_{lang}.txt', 'w')

    sents_batched = [all_sents[lang][i:i+32] for i in range(0, len(all_sents[lang]), 32)]
    pb = tqdm.tqdm(range(len(sents_batched)))
    for i, batch in enumerate(sents_batched):
    pb.update(1)
    inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
    translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=tok.lang_code_to_id['eng_Latn'], max_length=256)
    out = tok.batch_decode(translated_tokens, skip_special_tokens=True)
    pairs = list(zip(batch, out))
    for pair in pairs:
        out_file.write(f'{pair[0]} @@@ {pair[1]}\n')
    sents_out[lang].extend(out)
    out_file.close()

for entity in candidates:
    for lang in langs:
        ptr = 0 
        if(f'{lang}_text' in candidates[entity]):
            candidates[entity][f'{lang}_translated'] = sents_out[lang][ptr:ptr+len(candidates[entity][f'{lang}_text'])]
            ptr += len(candidates[entity][f'{lang}_text'])

with open(save_name, 'w') as f:
    json.dump(candidates, f, ensure_ascii=False)