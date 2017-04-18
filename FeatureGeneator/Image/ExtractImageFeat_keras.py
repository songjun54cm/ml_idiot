__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import os
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model
def ext_img_feat(image_folder, batch_size):
    base_model = ResNet50(weights='imagenet')
    img_model = Model(input=base_model.input, output=base_model.get_layer('res5c').output)

    img_list = os.listdir(image_folder)
    all_img_feats = list()
    si = 0
    while si < len(img_list):
        batch_img = img_list[si:si+batch_size]
        si += batch_size
        imgs = []
        for imgf in batch_img:
            img_path = os.path.join(image_folder, imgf)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            imgs.append(x)
        imgs = np.concatenate(imgs, axis=0)
        img_feats = img_model.predict(imgs)
        all_img_feats.append(img_feats)
        print('%d images extracted\r'%si),

def main(state):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', dest='image_folder', type=str, default='./')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=20)
    args = parser.parse_args()
    state = vars(args)
    main(state)