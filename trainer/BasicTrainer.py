__author__ = 'JunSong<songjun54cm@gmail.com>'
import logging
import os
import datetime
import abc
import json


def add_new_line(message):
    if message[-1] != '\n': message += '\n'
    return message


def log_to_file(message, file_name, folder_path):
    message = add_new_line(message)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'a') as f:
        f.write(message)


class BasicTrainer(object):
    def __init__(self, config):
        super(BasicTrainer, self).__init__()
        self.config = config
        self.loss_scale = config['loss_scale']
        self.optimizer = None
        self.tester = None
        self.smooth_train_loss = float('inf')
        self.train_log_file = None
        self.train_log_csv_file = None
        self.valid_log_file = None
        self.valid_log_csv_file = None
        self.last_save_model_file_path = None
        self.top_performance_csv_message = None
        self.metrics = []
        self.top_metric = 0
        self.top_metric_name = ""
        self.top_performance_valid_res = None

    @abc.abstractmethod
    def train(self, model, data_provider, tester):
        """
        train model, data comes from data provider
        :param model:   model to be trained
        :param data_provider:   provide data
        :param tester:  tester
        :return: None
        """
        raise NotImplementedError

    def create_out_folder(self):
        """
        create out put folder name 'output', and dump config to json
        :return:    None
        """
        if 'out_folder' not in self.config:
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
            logging.info('create out folder %s.' % self.config['out_folder'])
            os.makedirs(self.config['out_folder'])
        config_json_file = os.path.join(self.config['out_folder'], 'config.json')
        with open(config_json_file, 'w') as f:
            json.dump(json.dumps(self.config), f)
        logging.info('save current config into %s' % config_json_file)

    def create_log_files(self):
        """
        create train_log_file, train_log_csv_file, valid_log_csv_file
        :return:    None
        """
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
            self.train_log_file = open(self.config['train_log_file'], 'a')
            self.train_log_csv_file = open(self.config['train_log_csv_file'], 'a')
            self.valid_log_csv_file = open(self.config['valid_log_csv_file'], 'a')

    def log_message(self, message, file_name=None):
        logging.info(message)
        if message[-1] != '\n': message += '\n'
        self.log_train_message(message)
        self.log_valid_message(message)
        if file_name is not None:
            log_to_file(message, file_name, self.config['out_folder'])

    def log_train_message(self, message, file_name=None):
        logging.info('\n'+message)
        if message[-1] != '\n': message += '\n'
        if self.train_log_file is not None:
            self.train_log_file.write(message)
            self.train_log_file.flush()
        if file_name is not None:
            file_path = os.path.join(self.config['out_folder'], file_name)
            with open(file_path, 'a') as f:
                f.write(message)

    def log_train_csv(self, csv_message):
        csv_message = add_new_line(csv_message)
        if self.train_log_csv_file is not None:
            self.train_log_csv_file.write(csv_message)
            self.train_log_csv_file.flush()

    def log_valid_message(self, message):
        logging.info('\n'+message)
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

    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_loss * 100:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_train_message(message)
            return False
        # self.smooth_train_cost = loss
        return True

    def create_checkpoint_dir(self):
        """
        create checkpoint folder in outfolder/checkpint
        :return:    None
        """
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

    def to_validate(self, epoch_i, mode='num_epoch'):
        if mode == 'num_epoch':
            if epoch_i % self.valid_epoch_stride == 0:
                return True
            else:
                return False
        else:
            return (self.valid_sample_count >= self.valid_sample_num) or \
               (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)

    def form_valid_message(self, res):
        def from_split_valid_message(split_res):
            message = ''
            for key in split_res['metrics'].keys():
                message += '%s: %.5f ' % (key, split_res['metrics'][key])
            message = '%s evaluate %10d %15s samples in %.3fs, Loss: %5.3f. ' \
                      % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         split_res['sample_num'], split_res['split'], split_res['seconds'], split_res['loss']) + message
            return message + '\n'
        if 'metrics' in res:
            return from_split_valid_message(res)
        else:
            mesg = ''
            for key,val in res.items():
                mesg += from_split_valid_message(val)
            return mesg

    def form_valid_csv_message(self, res=None, mode=None):
        def form_split_valid_csv_body_message(split_res):
            body_message = '%d,%.3f,%s,%f' % (split_res['sample_num'], split_res['seconds'], split_res['split'], split_res['loss'])
            for met in split_res['metrics'].keys():
                body_message += ',%f' % split_res['metrics'][met]
            return body_message + '\n'

        def form_valid_csv_head(res):
            head_message = 'sample_num,seconds,split,Loss,'+','.join(res['metrics'].keys())
            return head_message + '\n'

        if mode == 'head':
            if 'metrics' in res:
                split_res = res
            else:
                split_res = res[list(res.keys())[0]]
            return form_valid_csv_head(split_res)
        elif mode == 'body':
            body_message = ''
            if 'metrics' in res:
                body_message = form_split_valid_csv_body_message(res)
            else:
                for key,val in res.items():
                    body_message += form_split_valid_csv_body_message(val)
            return body_message
        else:
            mesg = self.form_valid_csv_message(res, 'head')
            mesg += self.form_valid_csv_message(res, 'body')
            return mesg