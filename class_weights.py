#!/usr/bin/env python

import sys, os
import glob
import numpy as np
import skimage.io as io

files = glob.glob(os.path.join(sys.argv[1], "*.png"))
nfiles = len(files)

lbl_counts = {}
lbl_presence = {} # dict, store number of images where class is present

for f in files:
    img = io.imread(f)
    if img.ndim>2 and img.shape[2]>1:
        img = img[:,:,0] # first channel of gray label image

    id, counts = np.unique(img, return_counts=True)
    # normalize by total classes in image
    counts = counts / float(sum(counts))
    for i in range(len(id)):
        if id[i] in lbl_presence.keys():
            lbl_presence[id[i]] += 1
        else:
            lbl_presence[id[i]] = 1

        if id[i] in lbl_counts.keys():
            lbl_counts[id[i]] += counts[i]
        else:
            lbl_counts[id[i]] = counts[i]

# normalize by images in training set
for k in lbl_counts:
    lbl_counts[k] /= lbl_presence[k]

print "##########################"
print "class probability:"
for k in lbl_counts:
    print("%i: %f" % (k, lbl_counts[k]))
print "##########################"

# normalize on median freuqncy
med_frequ = np.median(lbl_counts.values())
lbl_weights = {}
for k in lbl_counts:
    lbl_weights[k] = med_frequ / lbl_counts[k]

print "##########################"
print "median frequency balancing:"
for k in lbl_counts:
    print("%i: %f" % (k, lbl_weights[k]))
print "##########################"

# class weight for classes that are not present in labeled image
missing_class_weight = 100000

max_class_id = np.max(lbl_weights.keys())

# print formated output for caffe prototxt
print "########################################################"
print "### caffe SoftmaxWithLoss format #######################"
print "########################################################"
print\
"  loss_param: {\n"\
"    weight_by_label_freqs: true"\
#"\n    ignore_label: 0"
for k in range(max_class_id+1):
    if k in lbl_weights:
        print "    class_weighting:", lbl_weights[k]
    else:
        print "    class_weighting:", missing_class_weight
print "  }"
print "########################################################"
