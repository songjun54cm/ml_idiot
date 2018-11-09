__author__ = 'SongJun-Dell'
from gradient_check.CheckerGradient import CheckerGradient

class CheckerModelGradient(CheckerGradient):
    def __init__(self, model, gc_data):
        super(CheckerModelGradient, self).__init__()
        self.model = model
        self.gc_data = gc_data

    def get_loss(self, mode='gc'):
        """

        :param mode:
        :return: loss, grads
        """
        return self.model.get_loss(self.gc_data, mode=mode)

    def get_gc_params(self):
        return self.model.get_params()

