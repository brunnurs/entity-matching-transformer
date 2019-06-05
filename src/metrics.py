import numpy as np
from torch import Tensor

#
# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)
#
#
# def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = True):
#     "Compute accuracy when `y_pred` and `y_true` are the same size."
#     if sigmoid: y_pred = y_pred.sigmoid()
#     #     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
#     return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()
#
#
# def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9,
#           sigmoid: bool = True):
#     "Computes the f_beta between `preds` and `targets`"
#     beta2 = beta ** 2
#     if sigmoid: y_pred = y_pred.sigmoid()
#     y_pred = (y_pred > thresh).float()
#     y_true = y_true.float()
#     TP = (y_pred * y_true).sum(dim=1)
#     prec = TP / (y_pred.sum(dim=1) + eps)
#     rec = TP / (y_true.sum(dim=1) + eps)
#     res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
#     return res.mean().item()
