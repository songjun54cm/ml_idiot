__author__ = 'SongJun-Dell'
import os, json, time, sys, random
import numpy as np
import cPickle as pickle
import importlib
from copy import deepcopy
import datetime

from ml_idiot.tester.BasicTester import BasicTester

def add_new_line(message):
    if message[-1] != '\n': message += '\n'
    return message

def log_to_file(message, file_name, folder_path):
    message = add_new_line(message)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'ab') as f:
        f.write(message)

def get_optimizer(config):
    from ml_idiot import optimizer
    opti = getattr(optimizer, config.get('optimizer', 'sgd'))(config)
    return opti

class BasicSolver(object):
    def __init__(self, config):
        self.config = config
        self.optimizer = get_optimizer(config)
        self.tester = BasicTester()
        self.rng = random.Random(1234)
        self.train_log_file = None
        self.valid_log_file = None
        self.grad_cache = dict()
        self.metrics = []

        self.last_save_model_file_path = None
        self.top_performance_csv_message = None

        self.top_metric = 0
        self.top_metric_name = ''

    def create_out_folder(self):
        if not self.config.has_key('out_folder'):
            num = 1
            out_prefix = os.path.join(self.config['proj_folder'], 'output',
                                      self.config['model_name'], self.config['data_set_name'], 'out')
            self.config['out_prefix'] = out_prefix
            out_folder = os.path.join(out_prefix, str(num))
            while os.path.exists(out_folder):
                num += 1
                out_folder = os.path.join(out_prefix, str(num))
            self.config['out_folder'] = out_folder
        if not os.path.exists(self.config['out_folder']):
            print 'create out folder %s.' % self.config['out_folder']
            os.makedirs(self.config['out_folder'])
        config_pkl_file = os.path.join(self.config['out_folder'], 'config.pkl')
        pickle.dump(self.config, open(config_pkl_file, 'wb'))
        print('save current config into %s' % config_pkl_file)

    def create_log_files(self):
        self.config['train_log_file'] = os.path.join(self.config['out_folder'], 'train_log.log')
        self.config['train_log_csv_file'] = os.path.join(self.config['out_folder'], 'train_log_csv.csv')
        self.config['valid_log_csv_file'] = os.path.join(self.config['out_folder'], 'valid_log_csv.csv')
        if self.config['mode'] == 'train':
            num = 1
            while os.path.exists(self.config['train_log_file']):
                self.config['train_log_file'] = self.config['train_log_file'] + '_' + str(num)
                num += 1
            while os.path.exists(self.config['valid_log_csv_file']):
                self.config['valid_log_csv_file'] = self.config['valid_log_csv_file'] + '_' + str(num)
                num += 1
            while os.path.exists(self.config['train_log_csv_file']):
                self.config['train_log_csv_file'] = self.config['train_log_csv_file'] + '_' + str(num)
                num += 1
            self.train_log_file = open(self.config['train_log_file'], 'ab')
            self.train_log_csv_file = open(self.config['train_log_csv_file'], 'ab')
            self.valid_log_csv_file = open(self.config['valid_log_csv_file'], 'ab')

    def log_train_message(self, message, file_name=None):
        print message
        if message[-1] != '\n': message += '\n'
        if self.train_log_file is not None:
            self.train_log_file.write(message)
            self.train_log_file.flush()
        if file_name is not None:
            file_path = os.path.join(self.config['out_folder'], file_name)
            with open(file_path, 'ab') as f:
                f.write(message)

    def log_train_csv(self, csv_message):
        csv_message = add_new_line(csv_message)
        if self.train_log_csv_file is not None:
            self.train_log_csv_file.write(csv_message)
            self.train_log_csv_file.flush()

    def log_valid_message(self, message):
        print message
        message = add_new_line(message)
        if self.valid_log_file is not None:
            self.valid_log_file.write(message)
            self.valid_log_file.flush()

    def log_valid_csv_message(self, message, file_name=None):
        csv_message = add_new_line(message)
        if self.valid_log_csv_file is not None:
            self.valid_log_csv_file.write(csv_message)
            self.valid_log_csv_file.flush()
        if file_name is not None:
            log_to_file(message, file_name, self.config['out_folder'])

    def log_message(self, message, file_name=None):
        print message
        if message[-1] != '\n': message += '\n'
        self.log_train_message(message)
        self.log_valid_message(message)
        if file_name is not None:
            log_to_file(message, file_name, self.config['out_folder'])

    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_loss * 100:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_train_message(message)
            return False
        # self.smooth_train_cost = loss
        return True

    def create_checkpoint_dir(self):
        self.config['checkpoint_out_dir'] = os.path.join(self.config['out_folder'], 'check_point')
        if not os.path.exists(self.config['checkpoint_out_dir']):
            message = 'creating folder %s' % self.config['checkpoint_out_dir']
            self.log_train_message(message)
            os.makedirs(self.config['checkpoint_out_dir'])

    def save_or_not(self, res, model, valid_csv_message=None, validate_res=None):
        save_tag, cp_suffix = self.detect_to_save(res, model)
        modelcp_prefix = 'model_checkpoint_%s_%s' % (self.config['model_name'], self.config['data_set_name'])
        model_file_name = '%s_%s' % (modelcp_prefix, cp_suffix)
        # print save_tag
        if save_tag:
            model_file_path = os.path.join(self.config['checkpoint_out_dir'], model_file_name)
            model.save(model_file_path)
            message = 'save checkpoint models in %s.' % model_file_path
            self.log_train_message(message)
            self.log_valid_csv_message(message)
            self.log_valid_csv_message('\n')
            self.last_save_model_file_path = model_file_path
            if valid_csv_message is not None:
                self.top_performance_csv_message = valid_csv_message
            if validate_res is not None:
                self.top_performance_valid_res = validate_res

    def detect_to_save(self, res, model):
        metric_score = res['metrics'].get(self.top_metric_name)
        if metric_score > self.top_metric:
            self.top_metric = metric_score
            save_tag = True
        else:
            save_tag = False
        # cp_sufix = 'accuracy_%.3f_tt_%.3f_.pkl' % (accuracy, truetrue)
        cp_sufix = '%s_%.6f.%s' % (self.top_metric_name, metric_score, model.save_ext)
        return save_tag, cp_sufix

    def get_batch_size(self, batch_data):
        if isinstance(batch_data, list):
            return len(batch_data)
        elif isinstance(batch_data, dict):
            return batch_data['batch_size']
        elif isinstance(batch_data, np.ndarray):
            return batch_data.shape[0]
        else:
            raise BaseException('get batch size error!')

    def to_validate(self, epoch_i):
        return (self.valid_sample_count >= self.valid_sample_num) or \
               (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)

    def form_valid_message(self, res):
        message = ''
        for key in self.metrics:
            message += '%s: %.5f ' % (key, res['metrics'][key])
        message = '%s evaluate %10d %15s samples in %.3fs, Loss: %5.3f. ' \
                  % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     res['sample_num'], res['split'], res['seconds'], res['loss']) + message
        return message

    def form_valid_csv(self, mode, res=None):
        if mode == 'head':
            head_message = 'sample_num,seconds,split,Loss,'+','.join(self.metrics)
            return head_message
        elif mode == 'body':
            body_message = '%d,%.3f,%s,%f' % (res['sample_num'], res['seconds'], res['split'], res['loss'])
            for met in self.metrics:
                body_message += ',%f' % res['metrics'][met]
            return body_message
        else:
            raise BaseException('form_valid_csv mode error.')

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
            'sample_num': self.get_batch_size(batch_data),
            'gth_feas': np.concatenate(gth_feas) if isinstance(gth_feas, list) else gth_feas,
            'pred_feas': np.concatenate(preds) if isinstance(preds, list) else preds
        }
        return res

    def train(self, model, data_provider, method='normal'):
        if method == 'normal':
            return self.normal_train(model, data_provider)
        elif method == 'k_fold':
            return self.k_fold_train(model, data_provider)
        else:
            raise StandardError('train method error.')

    def normal_train(self, model, data_provider):
        self.create_out_folder()
        self.create_log_files()
        self.create_checkpoint_dir()
        self.log_train_message(json.dumps(self.config, indent=2), file_name='model_config.txt')

        self.train_model(model ,data_provider)

        self.log_valid_csv_message(self.top_performance_csv_message, 'top_performance_log.csv')
        for res_name in ['train', 'valid', 'test']:
            message = self.form_valid_message(self.top_performance_valid_res[res_name])
            self.log_train_message(message)
        return self.last_save_model_file_path, self.top_performance_valid_res

    def k_fold_train(self, model, data_provider):
        self.create_out_folder()
        self.config['exp_out_folder'] = self.config['out_folder']
        k = len(data_provider.fold_splits)
        orig_state = deepcopy(self.config)
        folds_validate_res = list()
        for fold_num in xrange(k):
            self.config = deepcopy(orig_state)
            train_folds, train_valid_fold, valid_fold, test_fold = data_provider.get_split_fold_nums(fold_num, k)
            data_provider.form_splits(train_folds, train_valid_fold, valid_fold, test_fold)
            self.config['out_folder'] = os.path.join(self.config['exp_out_folder'], 'fold_%d'%fold_num)
            self.config['train_folds'] = list(train_folds)
            self.config['train_valid_fold'] = train_valid_fold
            self.config['valid_fold'] = valid_fold
            self.config['test_fold'] = test_fold
            model.reinit()
            cp_path, validate_res = self.normal_train(model, data_provider)
            folds_validate_res.append(validate_res)
            message = 'save model check point into %s' % cp_path
            self.log_train_message(message)
            for res_name in ['train', 'valid', 'test']:
                message = self.form_valid_message(validate_res[res_name])
                self.log_train_message(message)
        print '\n'
        print '='*20
        message = '%d-Fold validation mean performance.' % k
        print message
        csv_log_file = '%d_fold_valid_log.csv' % k
        log_to_file(message, csv_log_file, self.config['exp_out_folder'])
        log_to_file(self.form_valid_csv(mode='head'), csv_log_file, self.config['exp_out_folder'])
        mean_result = {}
        for res_name in ['train', 'valid', 'test']:
            res_value = self.mean_res(folds_validate_res, res_name)
            mean_result[res_name] = res_value
            message = self.form_valid_message(res_value)
            print message
            csv_message = self.form_valid_csv(mode='body', res=res_value)
            log_to_file(csv_message, csv_log_file, self.config['exp_out_folder'])

        print '%d fold training and validation finish.' % k
        return self.last_save_model_file_path, mean_result

    def mean_res(self, res_list, key):
        key_res = dict()
        key_res['split'] = key
        for k in res_list[0][key].keys():
            if k not in ['metrics', 'split']:
                key_res[k] = np.mean([res[key][k] for res in res_list])
            elif k == 'metrics':
                key_res['metrics'] = {}
                for km in res_list[0][key]['metrics'].keys():
                    key_res['metrics'][km] = np.mean([res[key]['metrics'][km] for res in res_list])
        return key_res

    def test(self, model, data_provider, method='normal'):
        if method == 'normal':
            return self.normal_test(model, data_provider)
        elif method == 'k_fold':
            return self.k_fold_test(model, data_provider)
        else:
            raise StandardError('test method error.')

    def normal_test(self, model, data_provider, split=None):
        if split is None:
            train_res = self.validate_on_split(model, data_provider, 'train_valid')
            valid_res = self.validate_on_split(model, data_provider, 'valid')
            test_res = self.validate_on_split(model, data_provider, 'test')
            test_result = {'train': train_res, 'valid': valid_res, 'test': test_res}
        else:
            res = self.validate_on_split(model ,data_provider, split)
            test_result = {split: res}
        return test_result

    def k_fold_test(self, model, data_provider):
        self.create_out_folder()
        k = len(data_provider.fold_splits)
        csv_log_file = '%d_fold_test_log.csv' % k
        folds_validate_res = list()
        for fold_num in xrange(k):
            valid_fold = fold_num
            test_fold = (fold_num+1)%k
            train_valid_fold = (fold_num+2)%k
            train_folds = (np.arange(k-2) + fold_num + 2) % k
            data_provider.form_splits(train_folds, train_valid_fold, valid_fold, test_fold)
            print 'test %d fold' % test_fold
            test_result = self.normal_test(model, data_provider)
            folds_validate_res.append(test_result)
            csv_message = 'test fold,%d' % k
            log_to_file(csv_message, csv_log_file, self.config['out_folder'])
            log_to_file(self.form_valid_csv(mode='head'), csv_log_file, self.config['out_folder'])
            for res_name in ['train', 'valid', 'test']:
                # print self.form_valid_message(test_result[res_name])
                # print '\n'
                csv_message = self.form_valid_csv(mode='body', res=test_result[res_name])
                log_to_file(csv_message, csv_log_file, self.config['out_folder'])
            log_to_file('\n', csv_log_file, self.config['out_folder'])
        print '='*20
        message =  '%d-Fold validation mean performance.' % k
        print message
        log_to_file(message, csv_log_file, self.config['out_folder'])
        log_to_file(self.form_valid_csv(mode='head'), csv_log_file, self.config['out_folder'])
        for res_name in ['train', 'valid', 'test']:
            res_value = self.mean_res(folds_validate_res, res_name)
            print self.form_valid_message(res_value)
            log_to_file(self.form_valid_csv(mode='body', res=res_value), csv_log_file, self.config['out_folder'])
        print '%d fold test finish.' % k
        print 'test results have been saved into %s' % os.path.join(self.config['out_folder'], csv_log_file)

    def reform_dp(self, data_provider):
        pass

    def train_model(self, model, data_provider):
        raise NotImplementedError('not implemented.')