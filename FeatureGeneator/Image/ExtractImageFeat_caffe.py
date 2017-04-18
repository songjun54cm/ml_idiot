__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import os
import numpy as np
import glob
import time
import copy
from tqdm import tqdm
import sys
import caffe
def get_caffe_model(caffe_dir, caffe_model, gpu=True,
                    image_dims=(256, 256),
                    mean_file='default',
                    raw_scale=255.0,
                    channel_swap=(2,1,0),
                    input_scale=None):
    if mean_file == 'default':
        mean_file = os.path.join(caffe_dir, 'python', 'caffe', 'imagenet', 'ilsvrc_2012_mean.npy')
    model_path = os.path.join(caffe_dir, 'models', caffe_model, '%s.caffemodel'%caffe_model)
    model_def = os.path.join(caffe_dir, 'models', caffe_model, 'deploy.prototxt')

    print('Loading mean file %s' % mean_file)
    mean = np.load(mean_file).mean(1).mean(1)
    if gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Classifier(model_def, model_path,
                           image_dims=image_dims, mean=mean,
                           input_scale=input_scale, raw_scale=raw_scale,
                           channel_swap=channel_swap)
    return net

def get_caffe_features_dir(image_dir, net, caffe_layer, batch_size=10, center_only=True, ext=None):
    file_pattern = '*' if ext is None else '*.%s'%ext
    print('Loading folder : %s' % image_dir)
    imgs_path =[im_f for im_f in glob.glob(os.path.join(image_dir, file_pattern))]
    feats = get_caffe_feature(net, imgs_path, caffe_layer, batch_size, center_only, ext)
    return feats

def get_caffe_feature(net, images_path, caffe_layer,
                      batch_size=10,
                      center_only=True, # enter crop alone or averaging predictions across crops
                      ext=None):
    print('Extract %d inputs...' % len(images_path)),
    all_feats = list()
    start = time.time()
    img_batches = [images_path[i:i+batch_size] for i in xrange(0, len(images_path), batch_size)]
    for img_b in tqdm(img_batches):
        input_batch = copy.deepcopy(img_b)
        # append data to match the batch size
        while len(input_batch)<batch_size:
            input_batch.append(img_b[-1])
        # print('file names: %s\n' % '\n'.join(input_batch))
        try:
            inputs =[caffe.io.load_image(im_f) for im_f in input_batch]
            pred = net.predict(inputs, oversample=False)
            features = net.blobs[caffe_layer].data[0:len(img_b)]
        except:
            print('image paths: \n %s' % '\n'.join(input_batch))
            print "Unexpected error:", sys.exc_info()[0]
            raise
        # must deepcopy the value, otherwise next forward will overwrite the blob.
        all_feats.append(copy.deepcopy(features))
    all_feats = np.concatenate(all_feats, axis=0)
    print('Done in %.2f seconds.' % (time.time()-start))
    return all_feats

def get_try_caffe_feature(net, images_path, caffe_layer,
                      batch_size=10,
                      center_only=True, # enter crop alone or averaging predictions across crops
                      ext=None):
    start = time.time()
    batch_image = list()
    succ_list = list()
    failed_list = list()
    all_features = list()
    for ii, img_p in tqdm(enumerate(images_path)):
        try:
            img = caffe.io.load_image(img_p)
            fflag = False
            if img.ndim>3 or img.ndim<2:
                fflag = True
            if img.ndim == 3 and img.shape[-1] > 4:
                fflag = True
        except:
            fflag = True
        if fflag:
            failed_list.append(img_p)
            print('total: %d, success: %d, failed: %d' % (ii, len(succ_list), len(failed_list)))
        else:
            succ_list.append(img_p)
            batch_image.append(img)

        if len(batch_image) >= batch_size:
            try:
                pred = net.predict(batch_image, oversample=False)
                features = net.blobs[caffe_layer].data[0:len(batch_image)]
                all_features.append(copy.deepcopy(features))
            except:
                print('image paths: \n%s' % '\n'.join(succ_list[-batch_size:]))
                print "Unexpected error:", sys.exc_info()[0]
                raise
            batch_image = list()
    if len(batch_image) > 0:
        temp_batch = batch_image
        while len(temp_batch) < batch_size:
            temp_batch.append(batch_image[-1])
        try:
            pred = net.predict(temp_batch, oversample=False)
            features = net.blobs[caffe_layer].data[0:len(batch_image)]
            all_features.append(copy.deepcopy(features))
        except:
            print('image paths: \n%s' % '\n'.join(succ_list[-batch_size:]))
            print "Unexpected error:", sys.exc_info()[0]
            raise

    all_features = np.concatenate(all_features, axis=0)
    print('total %d images, success %d, failed %d, in %.2f seconds' %
          (len(images_path), len(succ_list), len(failed_list), (time.time()-start)))
    return all_features, succ_list, failed_list

def main(state):
    net = get_caffe_model(state['caffe_dir'], state['caffe_model'], state['gpu'])
    feats = get_caffe_feature(net, state['image_dir'], state['caffe_layer'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', dest='image_dir', type=str, default='./images')
    parser.add_argument('--caffe_dir', dest='caffe_dir', type=str, default=os.environ['caffe_dir'])
    parser.add_argument('--caffe_model', dest='caffe_model', type=str, default='vgg19')
    parser.add_argument('--caffe_layer', dest='caffe_layer', type=str, default='fc7')
    parser.add_argument('--gpu', dest='gpu', type=int, default=1)
    args = parser.parse_args()
    state = vars(args)
    main(state)