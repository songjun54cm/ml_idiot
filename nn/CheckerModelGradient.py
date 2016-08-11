__author__ = 'SongJun-Dell'
from CheckerGradient import CheckerGraident

class CheckerModelGradient(CheckerGraident):
    def __init__(self, model, gc_data):
        super(CheckerModelGradient, self).__init__()
        self.model = model
        self.gc_data = gc_data

    def get_loss(self, mode='train'):
        return self.model.get_loss(self.gc_data, mode=mode)

    def get_gc_params(self):
        return self.model.get_params()

