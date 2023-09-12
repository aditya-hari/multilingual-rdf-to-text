from torch.nn.functional import softmax
from transformers import MT5ForConditionalGeneration, AutoTokenizer
import torch
import pickle
import tqdm
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--pairs_path', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()
pairs_path = args.pairs_path
save_path = args.save_path

model_name = "alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name, device_map=0, load_in_4bit=True)
model.eval()

pairs = open(pairs_path).readlines()

pairs_ = [p.split('\t') for p in pairs]
premises = [p[0].strip() for p in pairs_]
hypos = [p[1].strip() for p in pairs_]

ENTAILS_LABEL = "▁0"
NEUTRAL_LABEL = "▁1"
CONTRADICTS_LABEL = "▁2"

label_inds = tokenizer.convert_tokens_to_ids(
    [ENTAILS_LABEL, NEUTRAL_LABEL, CONTRADICTS_LABEL])

def process_nli(premise: str, hypothesis: str):
    """ process to required xnli format with task prefix """
    return "".join(['xnli: premise: ', premise, ' hypothesis: ', hypothesis])

pairs = list(zip(premises, hypos))
seqs = [process_nli(premise=premise, hypothesis=hypothesis) for premise, hypothesis in pairs]

batched_seqs = [seqs[i:i+32] for i in range(0, len(seqs), 32)]

entailment_ind = 0
contradiction_ind = 2

all_outputs = []
pb = tqdm.tqdm(range(len(batched_seqs)))
for bno, batch in enumerate(batched_seqs):
  pb.update(1)
  inputs = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True).to('cuda')
  out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, num_beams=1)
  scores = out.scores[0]
  scores = scores[:, label_inds]
  entail_vs_contra_scores = scores[:, [entailment_ind, contradiction_ind]]
  entail_vs_contra_probas = softmax(entail_vs_contra_scores, dim=1)
  batch_scores = torch.argmax(entail_vs_contra_probas, axis=1).cpu().numpy().tolist()
  # for i, val in enumerate(all_outputs):
  #   if(val == 0):
  #     outputs.write(pairs[(bno*32)+i])
  all_outputs.extend(batch_scores)

entailments = []
for i, val in enumerate(all_outputs):
  if(val == 0):
    entailments.append((i, pairs[i]))

with open(f'{save_path}/entailments.tsv', 'w', encoding='utf-8') as f:
  for entailment in entailments:
    f.write(f'{entailment[0]}\t{entailment[1][0]}\t{entailment[1][1]}\n')