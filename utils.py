# utils.py
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score, confusion_matrix
from config import Config

def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.DEVICE.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def bootstrap_evaluate(ourmodel, test_data, n_bootstrap=50, save_pred_path=None):
    test_data_cw = test_data[0]
    test_data_tf = test_data[1]
    test_data_gpt = test_data[2]
    test_label = test_data[3]

    n_samples = test_label.size(0)
    all_metrics = []
    all_preds_records = []

    for i in range(n_bootstrap):
        indices = torch.randint(0, n_samples, (n_samples,), dtype=torch.long)
        x_cw_sample = test_data_cw[indices]
        x_tf_sample = test_data_tf[indices]
        x_gpt_sample = test_data_gpt[indices]
        y_sample = test_label[indices]

        x_data = [x_cw_sample, x_tf_sample, x_gpt_sample]

        with torch.no_grad():
            pred, total_loss_valid, total_valid = ourmodel(x_data)
            pred = pred.cpu().numpy().reshape(-1)
            y_true = y_sample.cpu().numpy().reshape(-1)

        auc = roc_auc_score(y_true, pred)
        fpr, tpr, thresholds = roc_curve(y_true, pred)
        ks = np.max(tpr - fpr)

        best_f1 = -1
        optimal_threshold = 0.5
        for th in thresholds:
            preds_temp = (pred >= th).astype(int)
            f1_temp = f1_score(y_true, preds_temp, pos_label=1)
            if f1_temp > best_f1:
                best_f1 = f1_temp
                optimal_threshold = th

        preds = (pred >= optimal_threshold).astype(int)

        f1_1 = f1_score(y_true, preds, pos_label=1)

        all_metrics.append({
            'AUC': auc, 'KS': ks, 'F1_1': f1_1
        })

        for idx in range(len(y_true)):
            all_preds_records.append({
                'bootstrap': i+1, 'sample_idx': idx,
                'label': int(y_true[idx]), 'pred_prob': float(pred[idx]),
                'pred_label': int(preds[idx])
            })

        print(f"[Bootstrap {i+1}] AUC: {auc:.3f}, KS: {ks:.3f}, Risk F1: {f1_1:.3f}")

    if save_pred_path is not None:
        pd.DataFrame(all_preds_records).to_csv(save_pred_path, index=False)

    return all_metrics