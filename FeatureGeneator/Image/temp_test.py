__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse


import os
from ExtractImageFeat_caffe import *
caffe_model = 'VGG_ILSVRC_19_layers'
caffe_layer = 'fc7'
caffe_dir = '/home/songjun/tools/caffe'
data_dir = '/home/songjun/projects/souhu2017/data'
img_paths = [os.path.join(data_dir, x) for x in ['3304.jpg', '5466.jpg', '13996.jpg']]
net = get_caffe_model(caffe_dir, caffe_model, True)

feats = get_caffe_feature(net, img_paths, caffe_layer)

feat0 = get_caffe_feature(net, [img_paths[0]], caffe_layer)
feat10 = get_caffe_feature(net, [img_paths[0]]*10, caffe_layer)
feat11 = get_caffe_feature(net, [img_paths[0]]*11, caffe_layer)
feat24 = get_caffe_feature(net, [img_paths[0]]*24, caffe_layer)

img_paths += img_paths
feats6 = get_caffe_feature(net, img_paths, caffe_layer)

img_paths += img_paths
feats12 = get_caffe_feature(net, img_paths, caffe_layer)

img_paths += img_paths
feats24 = get_caffe_feature(net, img_paths, caffe_layer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    state = vars(args)
    main(state)