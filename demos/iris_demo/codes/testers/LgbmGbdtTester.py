__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/8
import argparse
from ml_idiot.tester.NormalTester import NormalTester


class LgbmGbdtTester(NormalTester):
    def __init__(self, config):
        super(LgbmGbdtTester, self).__init__(config)

