__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/7
config = {
    'data_set_name': 'iris',
    'model_name': 'LgbmGbdt',
    'model_config': {
        "num_leaves": 20,
        "num_trees": 100,
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "num_round": 10
    },
}