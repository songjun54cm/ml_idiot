__author__ = 'JunSong'
import numpy as np
def distance_loss(pred_feas, target_feas):
    """
    :param pred_feas: (N, D)
    :param target_feas: (N, D)
    :return:
    """
    loss = 0.5 * np.sum(np.square((pred_feas - target_feas)), axis=1).mean()
    return loss

def grad_distance_loss(pred_feas, target_feas):
    grad_pred = (pred_feas-target_feas) / pred_feas.shape[0]
    return grad_pred