#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:31:43 2019

@author: aristizabal
"""
#%%

# files for global RTOFS output
Dir_rtofs= '/Volumes/aristizabal/ncep_model/RTOFS_global_Michael/rtofs.'

# RTOFS grid file name
gridfile = '/Volumes/aristizabal/ncep_model/RTOFS_global_Michael/rtofs_glo.navy_0.08.regional.grid'

url_RTOFS = 'https://ftp.ncep.noaa.gov/data/nccf/com/rtofs/prod/rtofs.20190709/'

prefix_ab = 'rtofs_glo.t00z.n00.archv.a' 

# Name of 3D variable
var_name = 'temp'

# RTOFS a/b file name
#prefix_ab = 'rtofs_glo.t00z.n-48.archv'



#%%

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readgrids, readVar

import os
import os.path
import glob





xr.open_dataset(url_RTOFS + file)

#%%
