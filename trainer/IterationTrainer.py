__author__ = 'SongJun-Dell'
import importlib
import logging
import time
import numpy as np
from ml_idiot.trainer.NormalTrainer import NormalTrainer


def create_optimizer(config):
    opt = config.get("optimizer", "sgd")
    if opt=="null":
        print("optimizer=null, do not need optimizer.")
        return None
    logging.info("create optimizer: %s ..." % opt)
    opt_cls = getattr(importlib.import_module("ml_idiot.optimizer"), opt)
    opter = opt_cls(config)
    return opter


class IterationTrainer(NormalTrainer):
    def __init__(self, config):
        super(IterationTrainer, self).__init__(config)
        self.optimizer = create_optimizer(config)
        self.smooth_train_loss = float('inf')
        self.top_metric = 0.0
        self.valid_sample_count = 0
        self.valid_iter = config.get("valid_iter", None)
        self.valid_sample_num = None
        self.iter_count = 0
        self.max_epoch = config['max_epoch']
        self.train_size = None
        self.sample_count = 0
        self.smooth_rate = None
        self.to_valid_mode = config.get('to_valid_mode', 'num_epoch') # mode: num_epoch/num_iter/num_sample
        self.valid_epoch_stride = config.get('valid_epoch_stride', 1)
        self.tester = None

    def prepare_trainer(self, solver):
        super(IterationTrainer, self).prepare_trainer(solver)
        data_provider = solver.data_provider
        self.train_size = data_provider.split_size('train')
        self.batch_size = self.config.get("batch_size", self.train_size)
        self.valid_size = data_provider.split_size("valid")
        self.valid_batch_size = self.config.get("valid_batch_size", self.valid_size)
        v_num1 = int(self.train_size * self.valid_epoch_stride) if self.valid_epoch_stride is not None else self.train_size
        v_num2 = int(self.batch_size * self.valid_iter) if self.valid_iter is not None else self.train_size
        self.valid_sample_num = min(v_num1, v_num2)
        self.log_train_message('valid sample number: %d' % self.valid_sample_num)
        self.tester = solver.tester

    def train_model(self, model, data_provider, tester):
        self.valid_model(model, data_provider, 0)
        for epoch_i in range(self.max_epoch):
            self.sample_count = 0
            for batch_data in data_provider.iter_train_batches(self.batch_size):
                self.train_one_batch(model, batch_data, epoch_i)
                # validation
                if self.to_validate(epoch_i):
                    # self.valid_sample_count = 0
                    train_res, valid_res, test_res, valid_csv_message = self.valid_model(model, data_provider, epoch_i)
                    validate_res = {'train': train_res, 'valid': valid_res, 'test': test_res}

                    self.save_or_not(valid_res, model, valid_csv_message, validate_res)

    def to_validate(self, epoch_i):
        flag = (self.valid_sample_count >= self.valid_sample_num) or \
           (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)
        if flag:
            self.valid_sample_count = 0
        return flag

    def valid_prepare(self, model, data_provider):
        # over driver if needed.
        pass

    def train_one_batch(self, model, batch_data, epoch_i):
        batch_size = batch_data['batch_size']
        self.iter_count += 1
        t0 = time.time()

        train_res = model.train_batch(batch_data, self.optimizer)
        batch_loss = train_res["batch_loss"]
        score_loss = train_res["score_loss"]
        regu_loss = train_res["regu_loss"]
        # detect loss exploding
        if not self.detect_loss_explosion(batch_loss):
            raise BaseException('Loss Explosion !!! ......')

        batch_loss *= self.loss_scale
        self.valid_sample_count += batch_size
        self.update_smooth_train_loss(batch_loss)
        time_eclipse = time.time() - t0
        self.sample_count += batch_size

        epoch_rate = epoch_i + 1.0 * self.sample_count / self.train_size
        message = 'samples %d/%d done in %.3fs. epoch %.3f/%d. score_loss= %f, loss= %f, (smooth %f)' \
            % (self.sample_count, self.train_size, time_eclipse, epoch_rate, self.max_epoch, score_loss, batch_loss,
               self.smooth_train_loss)
        self.log_train_message(message)
        csv_message = 'epoch,%.3f,/,%d,samples,%d,/,%d,done_in,%.3f,seconds,score_loss,%f,loss,%f,smooth,%f' \
            % (epoch_rate, self.max_epoch, self.sample_count, self.train_size, time_eclipse, score_loss, batch_loss,
               self.smooth_train_loss)
        self.log_train_csv(csv_message)


    def update_smooth_train_loss(self, new_loss):
        # calculate smooth loss
        if self.iter_count == 1:
            self.smooth_train_loss = new_loss
            self.smooth_rate = 1.0 - min(0.01, (self.batch_size*1.0)/(self.train_size*1.0))
            logging.info('loss smooth rate: %f' % self.smooth_rate)
        else:
            self.smooth_train_loss = self.smooth_rate * self.smooth_train_loss + (1.0 - self.smooth_rate) * new_loss

    def valid_model(self, model, data_provider, epoch_i):
        valid_csv_message = 'epoch_num,%d\n' % epoch_i
        # validate on the train valid data set
        train_res = self.validate_on_split(model, data_provider, split='train_valid')
        valid_csv_message += self.form_valid_csv_message(res=train_res, mode='head') + '\n'
        valid_csv_message += self.form_valid_csv_message(res=train_res, mode='body') + '\n'
        # validate on the validate data set
        valid_res = self.validate_on_split(model, data_provider, split='valid')
        valid_csv_message += self.form_valid_csv_message(res=valid_res, mode='body') + '\n'
        # validate on the test data set
        test_res = self.validate_on_split(model, data_provider, split='test')
        valid_csv_message += self.form_valid_csv_message(res=test_res, mode='body') + '\n'
        self.log_valid_csv_message(valid_csv_message)
        self.log_valid_csv_message('\n')
        return train_res, valid_res, test_res, valid_csv_message

    def validate_on_split(self, model, data_provider, split):
        results = self.tester.validate_on_split(model, data_provider, split)
        message = self.form_valid_message(results)
        self.log_train_message(message)
        return results

    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_loss * 100:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_train_message(message)
            return False
        # self.smooth_train_cost = loss
        return True