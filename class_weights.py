#!/usr/bin/env python

import sys, os
import glob
import numpy as np
import skimage.io as io

np.set_printoptions(precision=15)

files = glob.glob(os.path.join(sys.argv[1], "*.png"))
nfiles = len(files)

lbl_counts = {}

for f in files:
    img = io.imread(f)[:,:,0] # first channel of gray label image
    id, counts = np.unique(img, return_counts=True)
    # normalize on image
    counts = counts / float(sum(counts))
    for i in range(len(id)):
        if id[i] in lbl_counts.keys():
            lbl_counts[id[i]] += counts[i]
        else:
            lbl_counts[id[i]] = counts[i]

# normalize on training set
for k in lbl_counts:
    lbl_counts[k] /= nfiles

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

#max_class_id = np.max(lbl_weights.keys())+1
max_class_id = 60

# print formated output for caffe prototxt
print "########################################################"
print "### caffe SoftmaxWithLoss format #######################"
print "########################################################"
print\
"  loss_param: {\n"\
"    weight_by_label_freqs: true"\
#"\n    ignore_label: 0"
for k in range(max_class_id):
    if k in lbl_weights:
        print "    class_weighting:", lbl_weights[k]
    else:
        print "    class_weighting:", missing_class_weight
print "  }"
print "########################################################"
