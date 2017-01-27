#!/usr/bin/env python

import skimage.io as io

import warnings
import sys, os
import glob

from joblib import Parallel, delayed
import multiprocessing


def sample(f):
    img = io.imread(f)
    # only keep ever 4th row and column
    img_sampled = img[::4, ::4]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(f, img_sampled)

files = glob.glob(os.path.join(sys.argv[1],'*.png'))

num_cores = multiprocessing.cpu_count()

# start in parallel
Parallel(n_jobs=num_cores)(delayed(sample)(f) for f in files)