__author__ = 'JunSong<songjun54cm@gmail.com>'
from copy import deepcopy
import os

from ml_idiot.trainer.BasicTrainer import BasicTrainer, log_to_file

class KFoldTrainer(BasicTrainer):
    def __init__(self, config):
        super(KFoldTrainer, self).__init__(config)

    def train(self, model, data_provider):
        self.create_out_folder()
        self.config['exp_out_folder'] = self.config['out_folder']
        k = len(data_provider.fold_splits)
        orig_config = deepcopy(self.config)
        folds_validate_res = list()
        for fold_num in xrange(k):
            self.config = deepcopy(orig_config)
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

    def test(self, model, data_provider):
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