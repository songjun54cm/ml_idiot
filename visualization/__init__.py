__author__ = 'JunSong'
import numpy as np

def visualization_matrix(mat):
    """
    :param mat: MxN (float or int array)
    :return:
    """
    def scale(v):
        tv = v-np.min(v)
        tv = tv/np.max(tv)
        return tv
    image = np.ones((mat.shape[0], mat.shape[1], 3))
    red_c = np.zeros(mat.shape)
    green_c = np.zeros(mat.shape)
    blue_c = np.zeros(mat.shape)

    posidx = np.where(mat>0)
    red_c[posidx] = mat[posidx]
    red_c = scale(red_c)
    image[:,:,1] *= 1-red_c
    image[:,:,2] *= 1-red_c

    neg_idx = np.where(mat<0)
    blue_c[neg_idx] = abs(mat[neg_idx])
    blue_c = scale(blue_c)
    image[:,:,0] *= 1-blue_c
    image[:,:,1] *= 1-blue_c

    # image = np.array([red_c,green_c,blue_c]).transpose([1,2,0])

    import matplotlib.pyplot as plt
    plt.imshow(image,interpolation='none')
    # plt.axis('off')
    plt.show()
