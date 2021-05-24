import torch
import torch.nn.functional as F
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
def bleu_all(list_of_references, hypotheses):
    # calculate bleu scores (batch-wise)
    assert len(list_of_references) == len(hypotheses)
    if len(list_of_references) == 0:
        raise ValueError("references size 0")
    if len(hypotheses) == 0:
        raise ValueError("hypotheses size 0")
    
    # # instance-level measure
    # bleu1 = sentence_bleu(references=references, hypothesis=hypothesis, weights=(1.0, 0.0, 0.0, 0.0))
    # bleu2 = sentence_bleu(references=references, hypothesis=hypothesis, weights=(0.0, 1.0, 0.0, 0.0))
    # bleu3 = sentence_bleu(references=references, hypothesis=hypothesis, weights=(0.0, 0.0, 1.0, 0.0))
    # bleu4 = sentence_bleu(references=references, hypothesis=hypothesis, weights=(0.0, 0.0, 0.0, 1.0))
    # bleua = sentence_bleu(references=references, hypothesis=hypothesis)
    # chencherry = SmoothingFunction()
    # bleus = sentence_bleu(references=references, hypothesis=hypothesis, smoothing_function=chencherry.method2)
    
    # batch-level measrue
    bleu1 = corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses, weights=(1.0, 0.0, 0.0, 0.0))
    bleu2 = corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses, weights=(0.0, 1.0, 0.0, 0.0))
    bleu3 = corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses, weights=(0.0, 0.0, 1.0, 0.0))
    bleu4 = corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses, weights=(0.0, 0.0, 0.0, 1.0))
    bleua = corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses)
    chencherry = SmoothingFunction()
    bleus = corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses, smoothing_function=chencherry.method2)
    
    keys = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu-a', 'bleu-s']
    vals = [bleu1, bleu2, bleu3, bleu4, bleua, bleus]
    bleu_scores = {k:100*vals[i] for i, k in enumerate(keys)}
    return bleu_scores

from rouge import Rouge
def rouge_all(hypotheses, references):
    # calculate rouge scores (batch-wise)
    assert len(hypotheses) == len(references)
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(hyps=hypotheses, refs=references, avg=True)
    except:
        hypotheses = [h if h != '' else ' ' for h in hypotheses]
        references = [r if r != '' else ' ' for r in references]
        rouge_scores = rouge.get_scores(hyps=hypotheses, refs=references, avg=True)
    return rouge_scores


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

def metrics_for_tasks(task,
                      stage,
                      batch=None,
                      outputs=None,
                      scores=None,
                      model=None,
                      tokenizer=None,
                      current_epoch=None,                  
    ):
    metrics = {f"{stage}_{k}":v for k,v in outputs.loss_dict.items()}
    if task == "Pre":
        if batch['lm_label'] is not None:
            lm_pred = torch.max(outputs.lang_prediction_logits, dim=2)[-1][:batch['lm_label'].size(0)][~batch['lm_label'].eq(-100)].view(-1).long()
            lm_gt = batch['lm_label'][~batch['lm_label'].eq(-100)].view(-1).long()
            metrics[f"{stage}_lm_acc"] = lm_pred==lm_gt
        if batch['kg_label'] is not None:
            kg_pred = torch.max(outputs.kg_prediction_logits, dim=2)[-1][:batch['lm_label'].size(0)][~batch['kg_label'].eq(-100)].view(-1).long()
            kg_gt = batch['kg_label'][~batch['kg_label'].eq(-100)].view(-1).long().detach()
            metrics[f"{stage}_kg_acc"] = kg_pred==kg_gt
        
    elif task == "Re":
        if stage == "valid":
            score = torch.max(outputs.pooled_logits,dim=1)[-1]
            gt = batch['label']
            metrics[f"{stage}_acc"] = score==gt
        else:
            raise ValueError("Test criterion for Retrieval is in pl_data.py. It should not be called here.")
 
    elif task == "AdmPred":
        pred = F.sigmoid(outputs.pooled_logits)
        gt = batch['label']
        metrics[f"{stage}_P@1"] = precision_at_k(gt, pred, k=1)
        metrics[f"{stage}_P@3"] = precision_at_k(gt, pred, k=3)
        metrics[f"{stage}_P@5"] = precision_at_k(gt, pred, k=5)
        metrics[f"{stage}_P@10"] = precision_at_k(gt, pred, k=10)

    elif task == "ErrDetect":
        pred = F.sigmoid(outputs.pooled_logits)
        if batch['lm_label'] is not None:
            gt = batch['lm_label']
        elif batch['kg_label'] is not None:
            gt = batch['kg_label']
        else:
            raise ValueError("Label for one of the domain needed to exist")

        metrics[f"{stage}_R@1"] = recall_at_k(gt, pred, k=1)
        metrics[f"{stage}_R@3"] = recall_at_k(gt, pred, k=3)
        metrics[f"{stage}_R@5"] = recall_at_k(gt, pred, k=5)
        metrics[f"{stage}_R@10"] = recall_at_k(gt, pred, k=10)

    elif task == "Gen":
        if stage == "valid":
            # measure the token-level accuracy (for masked language modeling)
            pred = torch.max(outputs.lang_prediction_logits, dim=2)[-1][~batch['lm_label'].eq(-100)].view(-1).long()
            gt = batch['lm_label'][~batch['lm_label'].eq(-100)].view(-1).long()
            metrics[f"{stage}_lm_acc"] = pred==gt
            
            # IGNORE_EARLY_EPOCHS = 30
            # VAL_INTERVAL_EPOCHS = 5
            # if current_epoch >= IGNORE_EARLY_EPOCHS and \
            #     (current_epoch+1) % VAL_INTERVAL_EPOCHS == 0:  # ignore ealry trials (\because time issue matters)
                
            #     # measure the ppl score
            #     batch_mean_ppl, org_lang_input_ids = model.decode_for_ppl(**batch)  # (teacher-forcing like) autrogressive
            #     metrics[f"{stage}_ppl"] = batch_mean_ppl
                
            #     # measure the rouge score
            #     pred, _, _ = model.decode(**batch)  # fully autorgressive
                
            #     batch_gt_strings = tokenizer.batch_decode(org_lang_input_ids, skip_special_tokens=True)
            #     batch_pred_strings = tokenizer.batch_decode(pred, skip_special_tokens=True)
                
            #     rouge_tot_scores = rouge_all(references=batch_gt_strings, hypotheses=batch_pred_strings)
            #     for rouge_metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            #         for rouge_sub_metric in ['f', 'p', 'r']:
            #             metrics[f"{stage}_{rouge_metric}_{rouge_sub_metric}"] = torch.tensor(rouge_tot_scores[rouge_metric][rouge_sub_metric])
                        
            #     # measure the bleu score
            #     batch_gt_strings_list = [[b] for b in batch_gt_strings]
            #     bleu_tot_scores = bleu_all(list_of_references=batch_gt_strings_list, hypotheses=batch_pred_strings)
            #     for bleu_metric in bleu_tot_scores.keys():
            #         metrics[f"{stage}_{bleu_metric}"] = torch.tensor(bleu_tot_scores[bleu_metric])
                
                
        else:
            # measure the ppl score
            batch_mean_ppl, org_lang_input_ids = model.decode_for_ppl(**batch)
            metrics[f"{stage}_ppl"] = batch_mean_ppl
            
            # measure the rouge score
            pred, _, _ = model.decode(**batch)
            
            batch_gt_strings = tokenizer.batch_decode(org_lang_input_ids, skip_special_tokens=True)
            batch_pred_strings = tokenizer.batch_decode(pred, skip_special_tokens=True)
            rouge_tot_scores = rouge_all(references=batch_gt_strings, hypotheses=batch_pred_strings)
            for rouge_metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                for rouge_sub_metric in ['f', 'p', 'r']:
                    metrics[f"{stage}_{rouge_metric}_{rouge_sub_metric}"] = torch.tensor(rouge_tot_scores[rouge_metric][rouge_sub_metric])
                    
            # measure the bleu score
            batch_gt_strings_list = [[b] for b in batch_gt_strings]
            bleu_tot_scores = bleu_all(list_of_references=batch_gt_strings_list, hypotheses=batch_pred_strings)
            for bleu_metric in bleu_tot_scores.keys():
                metrics[f"{stage}_{bleu_metric}"] = torch.tensor(bleu_tot_scores[bleu_metric])
                        
            # save eval_output files
            batch_kg_input_ids = batch['kg_input_ids'].cpu()
            org_lang_input_ids = org_lang_input_ids.cpu()
            pred_output_ids = [tensor_ids.cpu() for tensor_ids in pred]
            
            decode_outputs = (batch_kg_input_ids, org_lang_input_ids, pred_output_ids)
            return metrics, decode_outputs
    else:
        raise ValueError("Task not exist")

    return metrics