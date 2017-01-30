#!/usr/bin/env python

# relabel images such that non continuous class ids become continues (1 ... max_class_id)

import sys, os
import glob
import skimage.io as io
import numpy as np

import warnings

files = glob.glob(os.path.join(sys.argv[1], "*.png"))

labels = set()

for f in files:
    img = io.imread(f)
    if img.ndim>2 and img.shape[2]>1:
        img = img[:,:,0] # first channel of gray label image

    ids = np.unique(img)

    labels.update(ids)

print labels

labels = sorted(labels)

print labels

label_map = dict()

for i in range(len(labels)):
    label_map[labels[i]] = i

print label_map

# relabel images
for f in files:
    img = io.imread(f)
    if img.ndim>2 and img.shape[2]>1:
        img = img[:,:,0] # first channel of gray label image

    ids = np.unique(img)
    for lbl in ids:
        img[img==lbl] = label_map[lbl]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(f, img)