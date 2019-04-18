#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:50:48 2019

@author: aristizabal
"""
#%% user input

# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

# date limits
date_ini = '2018-10-08T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

#%%

import xarray as xr
import netCDF4
import numpy as np
from datetime import datetime 
from matplotlib.dates import date2num,
import os

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

timestamp_glider = date2num(time_glider)[0]

tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = np.asarray(latglider[0,okg[0]])
long = np.asarray(longlider[0,okg[0]])
depthg = depthglider[okg[0],:]
tempg = tempglider[okg[0],:]
saltg = saltglider[okg[0],:]

inst_id = ncglider.id.split('_')[0].split('-')[0]

#%% Save variables
  
import pickle

files = ['ng288_lat.pickle','ng288_lon.pickle','ng288_time.pickle']
var = [latg,long,timeg]

for i,ll in enumerate(var):
    myfile = open(files[i], 'wb')    
    pickle.dump(ll,myfile)
    myfile.close()

#%% open the file

import pickle

names = ['latg','long','timeg']
files = ['ng288_lat.pickle','ng288_lon.pickle','ng288_time.pickle']

myfile = open(files[0], 'rb')
latg = pickle.load(myfile)

myfile = open(files[1], 'rb')
long = pickle.load(myfile)

myfile = open(files[2], 'rb')
timeg = pickle.load(myfile)

#%% better example

#%% save variables
import pickle
 
file = 'GOFS31_GoM_2018-10-07 18:00.pickle'

with open(file, 'wb') as f:
    pickle.dump([lon31g,lat31g,sst31,su31,sv31], f)
    
#%% open data from pickle file
     
with open('GOFS31_GoM_2018-10-07 18:00.pickle', 'rb') as f:
     lon31g,lat31g,sst31,su31,sv31 = pickle.load(f)    

