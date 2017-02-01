#!/usr/bin/env python

# relabel images such that non continuous class ids become continues (1 ... max_class_id)

import sys, os
import glob
import skimage.io as io
import numpy as np
import csv

import warnings

files = glob.glob(os.path.join(sys.argv[1], "*.png"))

labels = set()

for f in files:
    img = io.imread(f)
    if img.ndim>2 and img.shape[2]>1:
        img = img[:,:,0] # first channel of gray label image

    ids = np.unique(img)

    labels.update(ids)

labels = sorted(labels)

# map labels from original label to collapsed label
label_map = dict()
for i in range(len(labels)):
    label_map[labels[i]] = i

base_folder = os.path.split(os.path.normpath(sys.argv[1]))[0]
csvwriter = csv.writer(open(os.path.join(base_folder, "label_map.csv"), 'w'), delimiter=' ')
for k,v in label_map.iteritems():
    csvwriter.writerow([k,v])

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
