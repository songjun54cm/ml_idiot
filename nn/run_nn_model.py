__author__ = 'SongJun-Dell'
from models.msae import MSAE


def fullfill_default_state(state):

    return state

def init_state(state):

    return state

def create_dp(state):
    data_provider = None
    return data_provider, state

def create_model(state):
    if state['model_name'] == 'msae':
        model = MSAE(state)
    else:
        raise(StandardError('models name error.'))
    return model

def main(state):
    state = fullfill_default_state(state)
    state = init_state(state)
    data_provider, state = create_dp(state)
    print 'state initialized!'

    # create models
    model = create_model(state)

    # create trainers
    from solver.NeuralNetworkSolver import NeuralNetworkSolver
    solver = NeuralNetworkSolver(state)

    print 'start training...'
    check_point_path = solver.train(model, data_provider, state)
    print 'training finish. start testing...'
    solver.test(model, data_provider, state)
    print 'Finish running...'