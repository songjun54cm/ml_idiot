__author__ = 'SongJun-Dell'
from solver.NeuralNetworkSolver import NeuralNetworkSolver

class TheanoSolver(NeuralNetworkSolver):
    def __init__(self, config):
        super(TheanoSolver, self).__init__(config)


    def update_model_one_batch(self, model, batch_data):
        loss = model.train_one_batch(batch_data)
        return loss
