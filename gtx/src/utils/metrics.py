import torch
import torch.nn.functional as F
from tqdm import tqdm

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

def precision_at_k(labels, scores, ks=None):
    sample_precision = {k:list() for k in ks}
    for label, score in zip(labels, scores):
        label = (label==1).nonzero()
        for k in ks:
            top_k_pred = score.topk(k).indices
            hit = sum([1 if pred in label else 0 for pred in top_k_pred])
            sample_precision[k].append(hit/k)

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

def metrics_for_tasks(task, stage, batch=None, outputs=None, scores=None):
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
        for k,v in task_metrics.items():
            if isinstance(k,int):
                metrics[f"{stage}_R@{k}"] = v
            else:
                metrics[f"{stage}_{k}"] = v

    elif task == "Gen":
        if stage == "valid":  # measure the token-level accuracy
            pred = torch.max(outputs.lang_prediction_logits, dim=2)[-1][~batch['lm_label'].eq(-100)].view(-1).long()
            gt = batch['lm_label'][~batch['lm_label'].eq(-100)].view(-1).long()
            metrics[f"{stage}_lm_acc"] = pred==gt
        else:  # finally, decoding
            test_outputs = evaluate_for_generation(model=model,
                                                   tokenizer=tokenizer,
                                                   dataset=test_dataset,
                                                   data_loader=test_dataloader,
                                                   training_args=training_args,
                                                   decode_option=decode_option,
                                                   mode='test')
            
            # summarize metrics
            _ = summarize_bleu_score(results=test_outputs, return_results=False)
            _ = summarize_ppl(results=test_outputs, return_results=False)
            
            # summarize metrics (for now, px)
            if '/px' in data_args.test_data_file:
                infos = graph_label_info(data_file=data_args.test_data_file, mimic_dir=MIMIC_TB_PATH, mode='test')
                _ = compute_and_summarize_refer_ratio(results=test_outputs,
                                                    tokenizer=tokenizer,
                                                    id2node=infos['id2node'],
                                                    db_words_pool=infos['db_words_pool'],
                                                    num_kg_relations=config.num_relations,
                                                    return_results=False)
    else:
        raise ValueError("Task not exist")

    return metrics