# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:24:49 2018

@author: Lasse
"""

import numpy as np
import h5py

# Load numpy data
primary_val = np.load("primary_val.npy")
evolutionary_val = np.load("evolutionary_val.npy")
tertiary_val = np.load("tertiary_val.npy")
mask_val = np.load("mask_val.npy")

# Save in h5py format
outfile_val = h5py.File("h5py_val", 'w')
dset_primary_val = outfile_val.create_dataset("primary_val", data = primary_val)
dset_evolutionary_val = outfile_val.create_dataset("evolutionary_val", data = evolutionary_val)
dset_tertiary_val = outfile_val.create_dataset("tertiary_val", data = tertiary_val)
dset_mask_val = outfile_val.create_dataset("mask_val", data = mask_val)
outfile_val.close()



