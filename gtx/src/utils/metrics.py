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
            pred = torch.max(outputs.lang_prediction_logits, dim=2)[-1][~batch['lm_label'].eq(-100)].view(-1).long()
            gt = batch['lm_label'][~batch['lm_label'].eq(-100)].view(-1).long()
            # metrics[f"{stage}_lm_acc"] = pred==kg
            metrics[f"{stage}_lm_acc"] = pred==gt
        else:
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