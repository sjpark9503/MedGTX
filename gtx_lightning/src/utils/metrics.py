import torch
import torch.nn.functional as F
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
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


def precision_at_k(labels, scores, ks=None):
    if ks is None:
        sample_precision = {"Precision":list()}
    else:
        sample_precision = {k:list() for k in ks}
        sample_precision["Precision"] = list()
    for label, score in zip(labels, scores):
        label_pos = (label==1).nonzero()
        if ks is not None:
            for k in ks:
                top_k_pred = score.topk(k).indices
                hit = sum([1 if pred in label_pos else 0 for pred in top_k_pred])
                sample_precision[k].append(hit/k)
        pred = (score>0.5).float()
        hit = pred[label==1].sum()
        if pred.sum() == 0:
            sample_precision["Precision"].append(0.0)
        else:
            sample_precision["Precision"].append(hit/pred.sum())

    return {k:torch.tensor(v) for k,v in sample_precision.items()}

def recall_at_k(labels, scores, ks=None):
    sample_recall = {k:list() for k in ks}
    sample_recall["Recall"] = list()
    for label, score in zip(labels, scores):
        label_pos = (label==1).nonzero()
        if len(label_pos)>0:
            for k in ks:
                top_k_pred = score.topk(k).indices
                hit = sum([1 if pred in label_pos else 0 for pred in top_k_pred])
                sample_recall[k].append(hit/len(label_pos))
            pred = (score>0.5).float()
            hit = pred[label==1].sum()
            sample_recall["Recall"].append(hit/label.sum())

    return {k:torch.tensor(v) for k,v in sample_recall.items()}

def metrics_for_tasks(task,
                      stage,
                      batch=None,
                      gt=None,
                      outputs=None,
                      loss_only=False,
                      scores=None,
                      model=None,
                      tokenizer=None,
                      current_epoch=None,                  
    ):
    if task not in ["ReAdm","NextDx","Death30","Death180","Death365"] or loss_only:
        metrics = {f"{stage}_{k}":v for k,v in outputs.loss_dict.items()}
        if loss_only:
            return metrics
    else:
        metrics = {}
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
        task_metrics = precision_at_k(gt, pred, ks=[1,3,5,10])
        for k,v in task_metrics.items():
            metrics[f"{stage}_P@{k}"] = v

    elif task == "ErrDetect":
        pred = F.sigmoid(outputs.pooled_logits)
        if 'lm_label' in batch:
            gt = batch['lm_label']
        elif 'kg_label' in batch:
            gt = batch['kg_label']
        else:
            raise ValueError("Label for one of the domain needed to exist")

        task_metrics = recall_at_k(gt, pred, ks=[1,5,10,20,50])
        precision_metrics = precision_at_k(gt, pred)
        task_metrics['Precision'] = precision_metrics['Precision']
        for k,v in task_metrics.items():
            if isinstance(k,int):
                metrics[f"{stage}_R@{k}"] = v
            else:
                metrics[f"{stage}_{k}"] = v

    elif task in ["ReAdm","Death30","Death180","Death365"]:
        pred = F.sigmoid(outputs).cpu()
        pred_ind = pred>0.5
        gt = gt.cpu()
        metrics[f"{stage}_acc"] = accuracy_score(gt, pred_ind)
        metrics[f"{stage}_f1"] = f1_score(gt, pred_ind)
        # if stage == "test":
        metrics[f"{stage}_AUPRC"] = average_precision_score(gt, pred)
        metrics[f"{stage}_AUROC"] = roc_auc_score(gt, pred)

    elif task == "NextDx":
        living_gt_idx = gt.sum(0)!=0
        gt = gt[:,living_gt_idx].cpu()
        pred = F.sigmoid(outputs[:,living_gt_idx]).cpu()
        pred_ind = pred>0.5
        metrics[f"{stage}_acc"] = accuracy_score(gt, pred_ind)
        metrics[f"{stage}_macro_f1"] = f1_score(gt, pred_ind, average="macro")
        metrics[f"{stage}_micro_f1"] = f1_score(gt, pred_ind, average="micro")
        # if stage == "test":
        metrics[f"{stage}_macro_AUPRC"] = average_precision_score(gt, pred, average="macro")
        metrics[f"{stage}_micro_AUPRC"] = average_precision_score(gt, pred, average="micro")
        metrics[f"{stage}_macro_AUROC"] = roc_auc_score(gt, pred, average="macro")
        metrics[f"{stage}_micro_AUROC"] = roc_auc_score(gt, pred, average="micro")


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