__author__ = 'SongJun-Dell'
import cPickle as pickle
class BasicModel(object):
    def __init__(self, config):
        self.save_ext = 'pkl'
        self.state = config

    def save(self, file_path):
        print 'trying to save model into %s' % file_path
        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
        # self.splits = d['splits']
        for key in self.__dict__.keys():
            self.__dict__[key] = d[key]