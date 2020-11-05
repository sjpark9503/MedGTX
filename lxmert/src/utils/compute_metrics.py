import torch

def get_accuracy(logit, label):
    _, predicted = torch.max(logit, 1)
    total = label.size(0)
    correct = (predicted == label).sum().item()
    
    return correct / total