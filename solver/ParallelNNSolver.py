__author__ = 'SongJun-Dell'
from multiprocessing import Pool
import time
import sys

from solver.NeuralNetworkSolver import NeuralNetworkSolver

class ParallelNNSolver(NeuralNetworkSolver):
    def __init__(self, config):
        super(ParallelNNSolver, self).__init__(config)
        self.pool = None
        if config['parallel']:
            print 'create parallel pool with %d processes.' % self.config['num_processes']
            self.pool = Pool(processes=self.config['num_processes'])

    def update_model_one_batch(self, model, batch_data):
        if self.config['parallel']:
            # loss_cost = self.parallel_train_process_one_batch(models, batch_data)
            loss_cost, grad_params = model.train_one_batch(batch_data, self.pool, self.config['num_processes'])
        else:
            loss_cost, grad_params = model.train_one_batch(batch_data)
            # loss_cost = self.train_process_one_batch(models, batch_data)
        if self.iter_count == 0:
            model.print_params()
            print grad_params

        self.update_model(model, grad_params)
        return loss_cost

    def train_one_batch(self, model, batch_data, epoch_i):
        self.iter_count += 1
        t0 = time.time()
        # print 'batch size: %d' % len(batch_data)
        loss_cost = self.update_model_one_batch(model, batch_data)

        loss_cost *= self.loss_scale
        batch_size = self.get_batch_size(batch_data)
        self.valid_sample_count += batch_size
        # calculate smooth cost
        if self.iter_count == 1:
            self.smooth_train_cost = loss_cost
        else:
            self.smooth_train_cost = 0.99 * self.smooth_train_cost + 0.01 * loss_cost
        # print message
        time_eclipse = time.time() - t0
        self.sample_count += batch_size

        epoch_rate = epoch_i + 1.0 * self.sample_count / self.train_size
        message = 'samples %d/%d done in %.3fs. epoch %.3f/%d. loss_cost= %f, (smooth %f)' \
            % (self.sample_count, self.train_size, time_eclipse, epoch_rate, self.max_epoch, loss_cost, self.smooth_train_cost)
        self.log_train_message(message)
        # detect loss exploding
        if not self.detect_loss_explosion(loss_cost):
            sys.exit()

