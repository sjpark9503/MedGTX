import torch

def get_accuracy(logit, label):
    try:
        _, predicted = torch.max(logit, dim=2)
        active_correct = (predicted == label)[~label.eq(-100)]
        accuracy = active_correct.sum().item()/active_correct.size(0)
    except:
        accuracy = 0
    return accuracy
