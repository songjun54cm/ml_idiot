__author__ = 'JunSong<songjun54cm@gmail.com>'


config = {
    'data_set_name': 'iris',
    'model_name': 'SJLR',
    'test_interval': 1,
    "max_epoch": 5,
    "optimizer": "sgd",
    "learning_rate": 0.002,
    "batch_size": 10,
    "valid_batch_size": 10,
    'model_config': {
        "fea_size": 4,
    },
}
