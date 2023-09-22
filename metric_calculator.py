from sacrebleu.metrics import BLEU, CHRF
from nltk.translate import meteor_score, chrf_score 
from rouge_score import rouge_scorer
import evaluate
import sys 
import glob 

bleu = BLEU()
chrf = CHRF()

gen_dir = sys.argv[1]
tgt_dir = sys.argv[2]

gen_files = sorted(glob.glob(f'{gen_dir}/*'))
test_tgt = [] 
tgts = sorted(glob(f'{tgt_dir}/*'))
print(tgts)

for tgt in tgts:
    test_tgt.append(open(tgt, 'r', encoding='utf-8').readlines())

def compute_bleu(self, ref_lines, gen_lines):
    corpus_score = bleu.corpus_score(gen_lines, ref_lines).score
    return corpus_score

def compute_meteor(self, ref_lines, gen_lines, tokenizer):
    gen_lines = [tokenizer(i) for i in gen_lines]
    ref_lines = [[tokenizer(i) for i in ref if i!=''] for ref in ref_lines]
    score = meteor_score.meteor_score(ref_lines, gen_lines)
    return score  

def compute_rouge(self, ref_lines, gen_lines):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    r1_sum = 0
    r2_sum = 0
    for i in range(len(ref_lines)):
        best_r1 = 0 
        best_r2 = 0
        for ref in ref_lines[i]:
            if ref == '':
                continue 
            scores = scorer.score(ref, gen_lines[i])
            best_r1 = max(best_r1, scores['rouge1'].fmeasure)
            best_r2 = max(best_r2, scores['rougeL'].fmeasure)
        r1_sum += best_r1
        r2_sum += best_r2
    return r1_sum/len(ref_lines), r2_sum/len(ref_lines)

def compute_chrf(self, ref_lines, gen_lines):
    corpus_score = chrf.corpus_score(gen_lines, ref_lines).score
    return corpus_score