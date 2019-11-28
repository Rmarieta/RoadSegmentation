import numpy as np
import glob
import cv2
import imutils

def augment_data(x, y):

    imgs_rot = []
    labels_rot = []

    for img, msk in zip(x, y):
        imgs_rot.append(img)
        labels_rot.append(msk)
        
        rot = [45, 90, 135, 180, 225, 270, 315]
        for i in range(len(rot)):
            imgs_rot.append(rotate_image(img, 45*(1+i)))
            labels_rot.append(rotate_image(msk, 45*(1+i)))
    imgs_sym = []
    labels_sym = []
    for img, msk in zip(imgs_rot, labels_rot):
        imgs_sym.append(img)
        labels_sym.append(msk)
        
        imgs_sym.append(make_symmetry(img, 0))
        labels_sym.append(make_symmetry(msk, 0))
        
        imgs_sym.append(make_symmetry(img, 1))
        labels_sym.append(make_symmetry(msk, 1))
    
    imgs_def = []
    labels_def = []
    for img, msk in zip(imgs_sym, labels_sym):
        imgs_def.append(img)
        labels_def.append(msk)
        
        img_def, lab_def=elastic_transform(img, msk, img.shape[1] * 2, img.shape[1] * 1.08, img.shape[1] * 0.38)
        imgs_def.append(img_def)
        labels_def.append(lab_def)

    return np.array(imgs_def), np.array(labels_def)


def warp_image(img, d=5):

    tps = cv2.createThinPlateSplineShapeTransformer()

    sx = img.shape[0]//3
    sy = img.shape[1]//3

    sshape = np.array([[sx, sy], [sx, 2*sy], [2*sx, 2*sy], [2*sx, sy]], np.float32)
    tshape = np.array([[sx, sy], [sx, 2*sy], [2*sx, 2*sy], [2*sx, sy]], np.float32) + np.random.randint(-d, d, size=sshape.shape)
    sshape = sshape.reshape(1, -1, 2)
    tshape = tshape.reshape(1, -1, 2)

    matches = list()
    matches.append(cv2.DMatch(0, 0, 0))
    matches.append(cv2.DMatch(1, 1, 0))
    matches.append(cv2.DMatch(2, 2, 0))
    matches.append(cv2.DMatch(3, 3, 0))

    tps.estimateTransformation(sshape, tshape, matches)

    ret, tshape_ = tps.applyTransformation(sshape)

    tps.estimateTransformation(tshape, sshape, matches)

    out_img = tps.warpImage(img)

    return out_img


def rotate_image(img, rot):
    out_img = imutils.rotate(img, rot)
    return out_img


def shift_image(img):
    dx = np.random.randint(-11, 11)
    dy = np.random.randint(-19, 19)
    out_img = img.copy()
    out_img[12-dx:-(12+dx), 20-dy:-(20+dy)] = img[12:-12, 20:-20]
    return out_img


def make_symmetry(img, hor_or_ver):
    
    out_img= cv2.flip(img, hor_or_ver)

    return out_img 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io
from skimage import color
from skimage import io
import glob
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def elastic_transform(image1, image2, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(3)
        
    shape = image1.shape
    shape_size = shape[:2]
    shape2 = image2.shape
    shape_size2 = shape2[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image1 = cv2.warpAffine(image1, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    image2 = cv2.warpAffine(image2, M, shape_size2[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    return image1, image2