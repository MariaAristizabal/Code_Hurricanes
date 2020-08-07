#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:58:57 2020

@author: aristizabal
"""

#%%

# url Doppio
url_doppio = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Get time bounds for current day
'''
te = datetime.today() + timedelta(1)
tend = datetime(te.year,te.month,te.day)

#ti = datetime.today() - timedelta(1)
ti = datetime.today() 
tini = datetime(ti.year,ti.month,ti.day)
'''

tini = datetime(2020, 7, 9)
tend = datetime(2020, 7, 10)

lat_buoy = [39.2717, 39.2717]
lon_buoy = [-73.88919999999999,-73.8891999999999]

target_latDOPP = lat_buoy
target_lonDOPP = lon_buoy

#%% Read Doppio time, lat and lon
print('Retrieving coordinates and time from Doppio ')

doppio = xr.open_dataset(url_doppio,decode_times=False)

latrhodoppio = np.asarray(doppio.variables['lat_rho'][:])
lonrhodoppio = np.asarray(doppio.variables['lon_rho'][:])
srhodoppio = np.asarray(doppio.variables['s_rho'][:])
ttdoppio = doppio.variables['time'][:]
tdoppio = netCDF4.num2date(ttdoppio[:],ttdoppio.attrs['units'])

oktimeDOPP = np.where(np.logical_and(tdoppio >= tini, tdoppio <= tend))[0]
timeDOPP = tdoppio[oktimeDOPP]

#%%

# getting the model grid positions for target_lonDOPP and target_latDOPP
oklatDOPP = np.empty((len(target_lonDOPP)))
oklatDOPP[:] = np.nan
oklonDOPP= np.empty((len(target_lonDOPP)))
oklonDOPP[:] = np.nan

# search in xi_rho direction 
oklatmm = []
oklonmm = []
for x in np.arange(len(target_latDOPP)):
    print(x)
    for pos_xi in np.arange(latrhodoppio.shape[1]):
        pos_eta = np.round(np.interp(target_latDOPP[x],latrhodoppio[:,pos_xi],np.arange(len(latrhodoppio[:,pos_xi])),\
                                     left=np.nan,right=np.nan))
        if np.isfinite(pos_eta):
            oklatmm.append((pos_eta).astype(int))
            oklonmm.append(pos_xi)
        
    pos = np.round(np.interp(target_lonDOPP[x],lonrhodoppio[oklatmm,oklonmm],np.arange(len(lonrhodoppio[oklatmm,oklonmm])))).astype(int)    
    oklatdoppio1 = oklatmm[pos]
    oklondoppio1 = oklonmm[pos] 
    
    #search in eta-rho direction
    oklatmm = []
    oklonmm = []
    for pos_eta in np.arange(latrhodoppio.shape[0]):
        pos_xi = np.round(np.interp(target_lonDOPP[x],lonrhodoppio[pos_eta,:],np.arange(len(lonrhodoppio[pos_eta,:])),\
                                    left=np.nan,right=np.nan))
        if np.isfinite(pos_xi):
            oklatmm.append(pos_eta)
            oklonmm.append(pos_xi.astype(int))
    
    pos_lat = np.round(np.interp(target_latDOPP[x],latrhodoppio[oklatmm,oklonmm],np.arange(len(latrhodoppio[oklatmm,oklonmm])))).astype(int)
    oklatdoppio2 = oklatmm[pos_lat]
    oklondoppio2 = oklonmm[pos_lat] 
    
    #check for minimum distance
    dist1 = np.sqrt((oklondoppio1-target_lonDOPP[x])**2 + (oklatdoppio1-target_latDOPP[x])**2) 
    dist2 = np.sqrt((oklondoppio2-target_lonDOPP[x])**2 + (oklatdoppio2-target_latDOPP[x])**2) 
    if dist1 >= dist2:
        oklatDOPP[x] = oklatdoppio1
        oklonDOPP[x] = oklondoppio1
    else:
        oklatDOPP[x] = oklatdoppio2
        oklonDOPP[x] = oklondoppio2
        
    oklatDOPP = oklatDOPP.astype(int)
    oklonDOPP = oklonDOPP.astype(int)
    
#%% Read Doppio S-coordinate parameters

Vtransf = np.asarray(doppio.variables['Vtransform'])
Vstrect = np.asarray(doppio.variables['Vstretching'])
Cs_r = np.asarray(doppio.variables['Cs_r'])
Cs_w = np.asarray(doppio.variables['Cs_w'])
sc_r = np.asarray(doppio.variables['s_rho'])
sc_w = np.asarray(doppio.variables['s_w'])

# critical depth parameter
hc = np.asarray(doppio.variables['hc'])

igrid = 1
    
# depth
h = np.asarray(doppio.variables['h'][oklatDOPP[0],oklonDOPP[0]])
zeta = np.asarray(doppio.variables['zeta'][oktimeDOPP[0],oklatDOPP[0],oklatDOPP[0]])

depthDOPP = np.empty((sc_r.shape[0]))
depthDOPP[:] = np.nan

# Calculate doppio depth
if Vtransf ==1:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            z0 = (sc_r[k]-Cs_r[k])*hc + Cs_r[k]*h
            depthDOPP[k] = z0 + zeta * (1.0 + z0/h)

if Vtransf == 2:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            z0 = (hc*sc_r[k] + Cs_r[k]*h) / (hc+h)
            depthDOPP[k] = zeta + (zeta+h)*z0    
    
tempDOPP = np.asarray(doppio.variables['temp'][oktimeDOPP[0],:,oklatDOPP[0],oklonDOPP[0]]) 

#%%

plt.figure()
plt.plot(tempDOPP,depthDOPP,'.-')

