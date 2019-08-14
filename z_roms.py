#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:22:29 2019

@author: aristizabal
"""

#%% User input

fname = '/volumes/aristizabal/doppio_output/output_Aug_02_2018/doppio_his.nc'
gname = '/volumes/aristizabal/doppio_output/grid_doppio_JJA_v12.nc'
igrid = 1 #'density points'
tindex = 0

#%%

import numpy as np
import xarray as xr


#%% Get doppio 

doppio = xr.open_dataset(fname,decode_times=False)

#%% Read Doppio S-coordinate parameters

Vtransf = np.asarray(doppio.variables['Vtransform'])
Vstrect = np.asarray(doppio.variables['Vstretching'])
Cs_r = np.asarray(doppio.variables['Cs_r'])
Cs_w = np.asarray(doppio.variables['Cs_w'])
sc_r = np.asarray(doppio.variables['s_rho'])
sc_w = np.asarray(doppio.variables['s_w'])

#%% Get depth

if igrid == 1:
    h = np.asarray(doppio.variables['h'])

# critical depth parameter
hc = np.asarray(doppio.variables['hc'])

#%% get free surface

if igrid ==1:   
    zeta = np.asarray(doppio.variables['zeta'][tindex,:,:])

#%% calculate depths

z = np.empty((zeta.shape[0],zeta.shape[1],sc_r.shape[0]))
z[:] = np.nan

if Vtransf ==1:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            z0 = (sc_r[k]-Cs_r[k])*hc + Cs_r[k]*h
            z[:,:,k] = z0 + zeta * (1.0 + z0/h);
        
if Vtransf == 2:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            print(k)
            z0 = (hc*sc_r[k] + Cs_r[k]*h) / (hc+h)
            z[:,:,k] = zeta + (zeta+h)*z0