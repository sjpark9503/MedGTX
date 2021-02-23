import torch

def mean_rank(self):
    """
    Returns
    -------
    mean_rank: float
        Mean rank of the true entity when replacing alternatively head
        and tail in any fact of the dataset.
    filt_mean_rank: float
        Filtered mean rank of the true entity when replacing
        alternatively head and tail in any fact of the dataset.
    """
    if not self.evaluated:
        raise NotYetEvaluatedError('Evaluator not evaluated call '
                                   'LinkPredictionEvaluator.evaluate')
    sum_ = (self.rank_true_heads.float().mean() +
            self.rank_true_tails.float().mean()).item()
    filt_sum = (self.filt_rank_true_heads.float().mean() +
                self.filt_rank_true_tails.float().mean()).item()
    return sum_ / 2, filt_sum / 2


def hit_at_k_heads(self, k=10):
    if not self.evaluated:
        raise NotYetEvaluatedError('Evaluator not evaluated call '
                                   'LinkPredictionEvaluator.evaluate')
    head_hit = (self.rank_true_heads <= k).float().mean()
    filt_head_hit = (self.filt_rank_true_heads <= k).float().mean()

    return head_hit.item(), filt_head_hit.item()


def hit_at_k_tails(self, k=10):
    if not self.evaluated:
        raise NotYetEvaluatedError('Evaluator not evaluated call '
                                   'LinkPredictionEvaluator.evaluate')
    tail_hit = (self.rank_true_tails <= k).float().mean()
    filt_tail_hit = (self.filt_rank_true_tails <= k).float().mean()

    return tail_hit.item(), filt_tail_hit.item()


def hit_at_k(self, k=10):
    """
    Parameters
    ----------
    k: int
        Hit@k is the number of entities that show up in the top k that
        give facts present in the dataset.
    Returns
    -------
    avg_hitatk: float
        Average of hit@k for head and tail replacement.
    filt_avg_hitatk: float
        Filtered average of hit@k for head and tail replacement.
    """
    if not self.evaluated:
        raise NotYetEvaluatedError('Evaluator not evaluated call '
                                   'LinkPredictionEvaluator.evaluate')

    head_hit, filt_head_hit = self.hit_at_k_heads(k=k)
    tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)

    return (head_hit + tail_hit) / 2, (filt_head_hit + filt_tail_hit) / 2


def mrr(self):
    """
    Returns
    -------
    avg_mrr: float
        Average of mean recovery rank for head and tail replacement.
    filt_avg_mrr: float
        Filtered average of mean recovery rank for head and tail
        replacement.
    """
    if not self.evaluated:
        raise NotYetEvaluatedError('Evaluator not evaluated call '
                                   'LinkPredictionEvaluator.evaluate')
    head_mrr = (self.rank_true_heads.float() ** (-1)).mean()
    tail_mrr = (self.rank_true_tails.float() ** (-1)).mean()
    filt_head_mrr = (self.filt_rank_true_heads.float() ** (-1)).mean()
    filt_tail_mrr = (self.filt_rank_true_tails.float() ** (-1)).mean()

    return ((head_mrr + tail_mrr).item() / 2,
            (filt_head_mrr + filt_tail_mrr).item() / 2)
