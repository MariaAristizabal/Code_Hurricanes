#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:42:11 2020

@author: aristizabal
"""

#%% User input

# RU33 (MAB + SAB)
lon_lim = [-75,-70]
lat_lim = [36,42]

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# Folder Fay Roms
folder_Fay_Roms = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/roms_his_fay.nc'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
#from datetime import datetime, timedelta
import cmocean

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
#oklatbath = oklatbath[:,np.newaxis]
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])
#oklonbath = oklonbath[:,np.newaxis]

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
#bath_elevsub = bath_elev[oklatbath,oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Defining cross transect

x1 = -74.1
y1 = 39.4
x2 = -73.0
y2 = 38.6
# Slope
m = (y1-y2)/(x1-x2)
# Intercept
b = y1 - m*x1

X = np.arange(x1,-72,0.05)
Y = b + m*X

# Conversion from glider longitude and latitude to Doppio convention
target_lonDOPP = X
target_latDOPP = Y

dist = np.sqrt((X-x1)**2 + (Y-y1)**2)*111 # approx along transect distance in km

#%% Read Doppio time, lat and lon
print('Retrieving coordinates and time from Doppio ')

doppio = xr.open_dataset(folder_Fay_Roms,decode_times=False)

latrhodoppio = np.asarray(doppio.variables['lat_rho'][:])
lonrhodoppio = np.asarray(doppio.variables['lon_rho'][:])
srhodoppio = np.asarray(doppio.variables['s_rho'][:])
ttdoppio = doppio.variables['ocean_time'][:]
timeDOPP = netCDF4.num2date(ttdoppio[:],ttdoppio.attrs['units'])

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

# depth
h = np.asarray(doppio.variables['h'])
# critical depth parameter
hc = np.asarray(doppio.variables['hc'])

igrid = 1

#%%

fig, ax = plt.subplots(figsize=(12, 6)) 
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
#ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)
ax.plot(X,Y,'-k')
ax.plot(x1,y1,'s',color='cyan',label='0 km')
ax.plot(x2,y2,'s',color='blue',label=str(np.round(dist[-1]))+' km')
ax.axis('scaled')
ax.legend(fontsize=14)
plt.title('Transect',fontsize=20)

file = folder + 'MAB_passage_transect'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Fay Roms
    
max_valt = 26
min_valt = 8   
nlevelst = max_valt - min_valt + 1

max_vals = 36.8
min_vals = 30.4  
nlevelss = 17

target_tempDOPP = np.empty((len(srhodoppio),len(target_lonDOPP)))
target_tempDOPP[:] = np.nan
target_saltDOPP = np.empty((len(srhodoppio),len(target_lonDOPP)))
target_saltDOPP[:] = np.nan
target_zDOPP = np.empty((len(srhodoppio),len(target_lonDOPP)))
target_zDOPP[:] = np.nan

tt = np.arange(12,73-24,6)
print('Getting glider transect from Doppio')
for t in tt:
    print('t = ',t)
    for pos in range(len(oklonDOPP)):
        print(len(oklonDOPP),pos)
        target_tempDOPP[:,pos] = doppio.variables['temp'][t,:,oklatDOPP[pos],oklonDOPP[pos]]
        target_saltDOPP[:,pos] = doppio.variables['salt'][t,:,oklatDOPP[pos],oklonDOPP[pos]]
        h = np.asarray(doppio.variables['h'][oklatDOPP[pos],oklonDOPP[pos]])
        zeta = np.asarray(doppio.variables['zeta'][t,oklatDOPP[pos],oklonDOPP[pos]])
        
        # Calculate doppio depth as a function of time
        if Vtransf ==1:
            if igrid == 1:
                for k in np.arange(sc_r.shape[0]):
                    z0 = (sc_r[k]-Cs_r[k])*hc + Cs_r[k]*h
                    target_zDOPP[k,pos] = z0 + zeta * (1.0 + z0/h)
    
        if Vtransf == 2:
            if igrid == 1:
                for k in np.arange(sc_r.shape[0]):
                    z0 = (hc*sc_r[k] + Cs_r[k]*h) / (hc+h)
                    target_zDOPP[k,pos] = zeta + (zeta+h)*z0
                    
    # change dist vector to matrix
    dist_matrix = np.tile(dist,(len(srhodoppio),1))
                    
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))     
    cs = plt.contourf(dist_matrix,target_zDOPP,target_tempDOPP,cmap=cmocean.cm.thermal,**kw)
    cbar = fig.colorbar(cs, orientation='vertical') 
    cbar.ax.set_ylabel('Temperature ($^oC$)',size=14)
    cs = plt.contour(dist_matrix,target_zDOPP,target_tempDOPP,[24],colors='k')
    fmt = '%i'
    plt.clabel(cs,fmt=fmt)
    cs = plt.contour(dist_matrix,target_zDOPP,target_tempDOPP,[12],colors='k')
    #plt.clabel(cs,fmt=fmt)
    plt.title('ROMS Transect on ' + timeDOPP[t].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-100,0)
    ax.set_xlim(0,200)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'MAB_passage_transect_temp_WRF_ROMS'+ \
                        timeDOPP[t].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_vals,max_vals,nlevelss)) 
    cs = plt.contourf(dist_matrix,target_zDOPP,target_saltDOPP,cmap=cmocean.cm.haline,**kw)
    cbar = fig.colorbar(cs, orientation='vertical') 
    cbar.ax.set_ylabel('Salinity',size=14)
    cs = plt.contour(dist_matrix,target_zDOPP,target_saltDOPP,[31.2],colors='orange')
    fmt = '%r'
    plt.clabel(cs,fmt=fmt)
    plt.title('ROMS Transect on ' + timeDOPP[t].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-100,0)
    ax.set_xlim(0,200)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'MAB_passage_transect_salt_WRF_ROMS'+ \
                        timeDOPP[t].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)    
