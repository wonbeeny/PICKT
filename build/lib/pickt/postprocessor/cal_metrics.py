# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-04-30


import torch
import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from .postprocessor_outputs import Universal_Output, Specific_Output

def universal_metric(
    config,
    labels: torch.Tensor, 
    preds: torch.Tensor,
) -> Universal_Output:
    y_true = labels.cpu()
    y_hat = preds.cpu()
    print("\n" + f"y_true shape: {y_true.shape}")
    print(f"y_hat shape: {y_hat.shape}\n")

    # confusion matrix 계산
    y_pred = torch.where(y_hat >= config.threshold, config.response2id["O"], config.response2id["X"])
    confusion_mat = confusion_matrix(y_true, y_pred, labels=[config.response2id["X"], config.response2id["O"]])
    acc_per_class = confusion_mat.diagonal() / confusion_mat.sum(axis=1)
    
    # 각 metric 계산
    acc_wrong = acc_per_class[0]      # 학생이 틀린 문제에 대한 정확도
    acc_correct = acc_per_class[1]    # 학생이 맞춘 문제에 대한 정확도
    acc_macro = np.mean(acc_per_class)
    acc_micro = accuracy_score(y_true, y_pred)
    # auc_macro = roc_auc_score(y_true, y_hat, average="macro", multi_class="ovr")
    auc_micro = roc_auc_score(y_true, y_hat, average="micro", multi_class="ovr")
    
    return Universal_Output(
        acc_wrong=round(float(acc_wrong), 4),
        acc_correct=round(float(acc_correct), 4),
        acc_macro=round(float(acc_macro), 4),
        acc_micro=round(float(acc_micro), 4),
        # auc_macro=round(float(auc_macro), 4),
        auc_micro=round(float(auc_micro), 4)
    )
    

def specific_metric(
    config,
) -> Specific_Output:

    return Specific_Output
