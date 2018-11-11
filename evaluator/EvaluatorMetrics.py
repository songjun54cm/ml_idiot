__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/2
import argparse


def classify_accuracy(gth_labels, pred_labels):
    # assert type(gth_labels) is list and type(pred_labels) is list, "type of gth_labels and pred_labels should be list"
    correct_num = 0
    total_num = len(gth_labels)
    for(gl, pl) in zip(gth_labels, pred_labels):
        if gl == pl:
            correct_num += 1
    return correct_num / total_num


def auc(gth_labels, pred_vals):
    # assert type(gth_labels) is list and type(pred_vals) is list, "type of gth_labels and pred_labels should be list"
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(gth_labels, pred_vals)
