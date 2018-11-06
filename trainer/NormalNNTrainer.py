__author__ = 'SongJun-Dell'
import importlib
import logging
import time
import numpy as np
from ml_idiot.trainer.NormalTrainer import NormalTrainer


def create_optimizer(config):
    opt = config.get("optimizer", "sgd")
    logging.info("create optimizer: %s ..." % opt)
    opt_cls = getattr(importlib.import_module("ml_idiot.optimizer"), opt)
    opter = opt_cls(config)
    return opter


class NormalNNTrainer(NormalTrainer):
    def __init__(self, config):
        super(NormalNNTrainer, self).__init__(config)
        self.optimizer = create_optimizer(config)
        self.smooth_train_loss = float('inf')
        self.top_metric = 0.0
        self.valid_sample_count = 0
        self.valid_iter = config.get("valid_iter", None)
        self.valid_sample_num = None
        self.iter_count = 0
        self.max_epoch = config['max_epoch']
        self.batch_size = config['batch_size']
        self.train_size = None
        self.sample_count = 0
        self.smooth_rate = None
        self.to_valid_mode = config.get('to_valid_mode', 'num_epoch') # mode: num_epoch/num_iter/num_sample
        self.valid_epoch_stride = config.get('valid_epoch_stride', 1)

    def prepare_trainer(self, solver):
        super(NormalNNTrainer, self).prepare_trainer(solver)
        data_provider = solver.data_provider
        self.train_size = data_provider.split_size('train')
        self.batch_size = self.config.get("batch_size", self.train_size)
        v_num1 = int(self.train_size * self.valid_epoch_stride) if self.valid_epoch_stride is not None else self.train_size
        v_num2 = int(self.batch_size * self.valid_iter) if self.valid_iter is not None else self.train_size
        self.valid_sample_num = min(v_num1, v_num2)
        self.log_train_message('valid sample number: %d' % self.valid_sample_num)

    def train_model(self, model, data_provider, tester):
        self.train_size = data_provider.split_size('train')
        for epoch_i in range(self.max_epoch):
            for batch_data in data_provider.iter_train_batches(self.batch_size):
                self.train_one_batch(model, batch_data, epoch_i)
                # validation
                if self.to_validate(epoch_i):
                    # self.valid_sample_count = 0
                    train_res, valid_res, test_res, valid_csv_message = self.valid_model(model, data_provider, epoch_i)
                    validate_res = {'train': train_res, 'valid': valid_res, 'test': test_res}

                    self.save_or_not(valid_res, model, valid_csv_message, validate_res)

    def to_validate(self, epoch_i):
        return (self.valid_sample_count >= self.valid_sample_num) or \
           (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)

    def valid_prepare(self, model, data_provider):
        # over driver if needed.
        pass

    def train_one_batch(self, model, batch_data, epoch_i):
        batch_size = batch_data['batch_size']
        self.iter_count += 1
        t0 = time.time()

        batch_loss, score_loss, regu_loss = model.train_batch(batch_data)

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
        # detect loss exploding
        if not self.detect_loss_explosion(batch_loss):
            raise BaseException('Loss Explosion !!! ......')

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
        valid_csv_message += self.form_valid_csv_message(mode='head') + '\n'
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

    # def form_valid_csv(self, mode, res=None):
    #     if mode == 'head':
    #         head_message = 'sample_num,seconds,split,Loss,'+','.join(self.metrics)
    #         return head_message
    #     elif mode == 'body':
    #         body_message = '%d,%.3f,%s,%f' % (res['sample_num'], res['seconds'], res['split'], res['loss'])
    #         for met in self.metrics:
    #             body_message += ',%f' % res['metrics'][met]
    #         return body_message
    #     else:
    #         raise BaseException('form_valid_csv mode error.')

    def validate_on_split(self, model, data_provider, split):
        t0 = time.time()
        res = self.valid_split_metrics(model, data_provider, split)
        time_eclipse = time.time() - t0

        results = dict()
        # results.update(res)
        results['metrics'] = res['metrics']
        results['sample_num'] = res['sample_num']
        results['seconds'] = time_eclipse
        results['split'] = split
        results['loss'] = res['loss']*self.config['loss_scale']
        message = self.form_valid_message(results)
        self.log_train_message(message)
        return results

    def valid_split_metrics(self, model, data_provider, split):
        res = self.test_on_split(model, data_provider, split)
        metrics = self.tester.get_metrics(res, self.metrics)
        res['metrics'] = metrics
        return res

    def test_on_split(self, model, data_provider, split):
        total_loss = []
        sample_num = 0
        gth_feas = []
        pred_feas = []
        for batch_data in data_provider.iter_split_batches(self.valid_batch_size, split):
            res = self.test_one_batch(model, batch_data)
            total_loss.append(res['loss'])
            sample_num += res['sample_num']
            gth_feas.append(res['gth_feas'])
            pred_feas.append(res['pred_feas'])
        res = {
            'loss': np.mean(total_loss),
            'sample_num': sample_num,
            'gth_feas': np.concatenate(gth_feas),
            'pred_feas': np.concatenate(pred_feas)
        }
        return res

    def test_one_batch(self, model, batch_data):
        outs = model.loss_pred_on_batch(batch_data['x'], batch_data['y'])
        loss = outs[0]
        preds = outs[1]
        gth_feas = batch_data['y']
        res = {
            'loss': loss,
            'sample_num': batch_data['batch_size'],
            'gth_feas': np.concatenate(gth_feas) if isinstance(gth_feas, list) else gth_feas,
            'pred_feas': np.concatenate(preds) if isinstance(preds, list) else preds
        }
        return res

    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_loss * 100:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_train_message(message)
            return False
        # self.smooth_train_cost = loss
        return True