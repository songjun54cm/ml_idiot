__author__ = 'SongJun-Dell'
import json, time
from NeuralNetworkSolver import NeuralNetworkSolver

class TheanoSolver(NeuralNetworkSolver):
    def __init__(self, state):
        super(TheanoSolver, self).__init__(state)


    def update_model_one_batch(self, model, batch_data):
        loss = model.train_one_batch(batch_data)
        return loss
