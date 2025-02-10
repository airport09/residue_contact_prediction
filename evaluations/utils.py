import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(gt, pred, thresh=0.5):
    pred = torch.where(pred > thresh, 1, 0)

    gt.diagonal().fill_(1)
    pred.diagonal().fill_(1)
    
    gt = gt.flatten().numpy()
    pred = pred.flatten().numpy() 

    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)

    return {"accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1}
