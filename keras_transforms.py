import numpy as np
import scipy.ndimage as ndi
from six.moves import range

import time
import cv2
import random
import matplotlib.pyplot as plt

def channel_shift(xs, intensity, channel_axis):
    ys = []
    for x in xs:
        if x.ndim == 3: # image
            x = np.rollaxis(x, channel_axis, 0)
            min_x, max_x = np.min(x), np.max(x)
            channel_images = [np.clip(x_channel + intensity, min_x, max_x)
                            for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
            ys.append(x)

        else:
            ys.append(x)

    return ys

def apply_transforms(xs, transform_matrix, output_shape=None):
    """Apply the image transformation specified by a matrix.
    """
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    print('ke:', transform_matrix)
    ys = []
    for x in xs:
        if x.ndim == 3: # image
            x = np.rollaxis(x, 2, 0)
            channel_images = [ndi.interpolation.affine_transform(x_channel,
                                                                 final_affine_matrix,
                                                                 final_offset,
                                                                 order=1,
                                                                 output_shape=output_shape,
                                                                 mode='constant',
                                                                 cval=0) for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, 2 + 1)
            ys.append(x)

        else: # mask
            x = ndi.interpolation.affine_transform(x,
                                                   final_affine_matrix,
                                                   final_offset,
                                                   order=0,
                                                   output_shape=output_shape,
                                                   mode='constant',
                                                   cval=0)
            ys.append(x)

    
    return ys

def apply_transforms_cv(xs, M):
    """Apply the image transformation specified by a matrix.
    """
    dsize = (np.int(xs[0].shape[1]), np.int(xs[0].shape[0]))

    aff = M[:2, :2]
    off = M[:2, 2]
    cvM = np.zeros_like(M[:2, :])
    # cvM[:2,:2] = aff
    cvM[:2,:2] = np.flipud(np.fliplr(aff))
    # cvM[:2,:2] = np.transpose(aff)
    cvM[:2, 2] = np.flip(off, axis=0)
    ys = []
    for x in xs:
        if x.ndim == 3: # image
            x = cv2.warpAffine(x, cvM, dsize, flags=cv2.INTER_LINEAR)
            ys.append(x)
            # M = cv2.getRotationMatrix2D((dsize[0] // 2, dsize[1] // 2), angle, 1)
            #         _img = cv2.warpAffine(_img, M, dsize, flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REPLICATE)
            #         _mask = cv2.warpAffine(_mask, M, dsize, flags=cv2.INTER_NEAREST)#, borderMode=cv2.BORDER_REPLICATE)
            #         _oomk = cv2.warpAffine(_oomk, M, dsize, flags=cv2.INTER_NEAREST)

        else: # mask
            x = cv2.warpAffine(x, cvM, dsize, flags=cv2.INTER_NEAREST)
            ys.append(x)

    
    return ys


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def flip_axis(xs, axis):
    ys = []
    for x in xs:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        ys.append(x)

    return ys

def random_transform(xs, rnd,
                     rt=False, # rotation
                     hs=False, # height_shift
                     ws=False, # width_shift
                     sh=False, # shear
                     zm=[1,1], # zoom
                     sc=[1,1],
                     cs=False, # channel shift
                     hf=False): # horizontal flip
                    
    """Randomly augment a single image tensor.
    """
    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = 0
    img_col_axis = 1
    img_channel_axis = 2
    h, w = xs[0].shape[img_row_axis], xs[0].shape[img_col_axis]

    # use composition of homographies
    # to generate final transform that needs to be applied
    if rt:
        theta = np.pi / 180 * rnd.uniform(-rt, rt)
    else:
        theta = 0

    if hs:
        tx = rnd.uniform(-hs, hs) * h
    else:
        tx = 0 
 
    if ws:
        ty = rnd.uniform(-ws, ws) * w
    else:
        ty = 0

    if sh:
        shear = np.pi / 180 * rnd.uniform(-sh, sh)
    else:
        shear = 0

    if zm[0] == 1 and zm[1] == 1:
        zx, zy = 1, 1
    else:
        zx = rnd.uniform(zm[0], zm[1])
        zy = rnd.uniform(zm[0], zm[1])

    if sc[0] == 1 and sc[1] == 1:
        zx, zy = zx, zy
    else:
        s = rnd.uniform(sc[0], sc[1])
        zx = zx * s
        zy = zy * s

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix


    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                    [0, 1, ty],
                                    [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        if rnd.random() < 0.5:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
        else:
            shear_matrix = np.array([[np.cos(shear), 0, 0],
                                    [np.sin(shear), 1, 0],
                                    [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

        xs = apply_transforms_cv(xs, transform_matrix)

        # plt.figure(1)
        # plt.subplot(2,1,1); plt.imshow(xs[0])
        # plt.subplot(2,1,2); plt.imshow(xs[1])
        # plt.show()


    if cs != 0:
        intensity = rnd.uniform(-cs, cs)
        xs = channel_shift(xs,
                            intensity,
                            img_channel_axis)
    
    if hf:
        if rnd.random() < 0.5:
            xs = flip_axis(xs, img_col_axis)

    return xs
