import torch
import torch.nn.functional as F


'''
metric for note generation
'''
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu_all(references, hypothesis):
    if len(references) == 0:
        raise ValueError("references size 0")
    if len(hypothesis) == 0:
        raise ValueError("hypothesis size 0")
    bleu1 = sentence_bleu(references, hypothesis, weights=(1.0, 0.0, 0.0, 0.0))
    bleu2 = sentence_bleu(references, hypothesis, weights=(0.0, 1.0, 0.0, 0.0))
    bleu3 = sentence_bleu(references, hypothesis, weights=(0.0, 0.0, 1.0, 0.0))
    bleu4 = sentence_bleu(references, hypothesis, weights=(0.0, 0.0, 0.0, 1.0))
    bleua = sentence_bleu(references, hypothesis)
    chencherry = SmoothingFunction()
    bleus = sentence_bleu(references, hypothesis, smoothing_function=chencherry.method2)
    keys = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu-a', 'bleu-s']
    vals = [bleu1, bleu2, bleu3, bleu4, bleua, bleus]
    return [{keys[i]:100*vals[i] for i in range(len(keys))}]

# from rouge import Rouge 
# def rouge_all(references, hypothesis):
#     rouge = Rouge()
#     return rouge.get_scores(hypothesis, references)

def precision_at_k(labels, scores, k=10):
    sample_precision = list()
    for label, score in zip(labels, scores):
        label = [i for i, e in enumerate(label) if e != 0]
        top_k_pred = sorted(range(len(score)), key=lambda k: score[k], reverse=True)[:k]
        hit = sum([1 if pred in label else 0 for pred in top_k_pred])
        sample_precision.append(hit/k)

    return sum(sample_precision)/len(sample_precision)

def recall_at_k(labels, scores, k=10):
    sample_recall = list()
    for label, score in zip(labels, scores):
        label = [i for i, e in enumerate(label) if e != 0]
        if len(label)>0:
            top_k_pred = sorted(range(len(score)), key=lambda k: score[k], reverse=True)[:k]
            hit = sum([1 if pred in label else 0 for pred in top_k_pred])
            sample_recall.append(hit/len(label))

    return sum(sample_recall)/len(sample_recall)
