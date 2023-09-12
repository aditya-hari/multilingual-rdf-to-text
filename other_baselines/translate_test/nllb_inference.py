from transformers import Text2TextGenerationPipeline, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, AutoTokenizer
from sacrebleu import BLEU
import torch 

#model_name = "google/mt5-small"

nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='eng_Latn', )
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("checkpoint")
nllb_model.eval()

lang_map = {
    'br': 'bre_Latn',
    'cy': 'cym_Latn',
    'ga': 'gle_Latn',
    'ml': 'mlt_Latn',
}

data_file = [i.strip() for i in open(f'dev_src.txt').readlines()]
ref_file = [i.strip() for i in open(f'dev_ref.txt').readlines()]
gens_actual = [i.strip() for i in open(f'eng_all_gen.txt').readlines()]

with torch.no_grad():
    for lang in ['cy', 'ga', 'mt']:
        inputs = nllb_tokenizer(gens_actual, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=nllb_tokenizer.lang_code_to_id[lang_map[lang]], max_length=256)
        out = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)    
        with open(f'data/results/{lang}_nllb_gen.txt', 'w') as f:
            f.write('\n'.join(out))
        print(f'BLEU score for {lang}: {BLEU().corpus_score(out, [ref_file]).score}')

