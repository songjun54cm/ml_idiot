__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.tester.NormalTester import NormalTester


class LRTester(NormalTester):
    def __init__(self, config):
        super(LRTester, self).__init__(config)
