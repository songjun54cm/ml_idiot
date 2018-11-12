__author__ = 'JunSong<songjun54cm@gmail.com>'


config = {
    "data_set_name": "iris",
    "model_name": "SKLearnLRModel",
    "data_provider": "SJLRDataProvider",
    "trainer": "SJLRTrainer",
    "tester": "SJLRTester",
    "evaluator": "SJLREvaluator",
    "optimizer": "null",
    "batch_iter_n": 1,
    "max_epoch": 10,
    "model_config": {
        # "learning_rate": "optimal",
        "learning_rate": "constant",
        "eta0": 0.0005
    }
}