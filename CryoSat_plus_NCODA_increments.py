#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:17:58 2019

@author: aristizabal
"""

#%% User input

#cryosat_url = 'https://science-pds.cryosat.esa.int/#Cry0Sat2_data%2FSIR_GOP_P2P%2F2018%2F10'
cryosat_folder = '/Volumes/aristizabal/CryoSat_data/'
date = ['2018/09/','2018/10/']

# Directories where increments files reside 
Dir= '/Volumes/aristizabal/GOFS/'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# lat and lon bounds
lon_lim = [-100,-10]
lat_lim = [0,50] 

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z' 

#%%
import matplotlib.pyplot as plt
import xarray as xr
import os
import glob
import numpy as np
import netCDF4
from netCDF4 import Dataset
import datetime

#%% Access CryoSat files

nc_list = os.listdir(cryosat_folder)

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
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

#%% Reading ncoda data Florence

ncoda_files = sorted(glob.glob(os.path.join(Dir,'*seatmp*')))

#ncncoda = xr.open_dataset(ncoda_files[2]) # Michael
ncncoda = Dataset(ncoda_files[0]) # Florence
temp_incr = ncncoda.variables['pot_temp'][:]

time_ncoda = ncncoda.variables['MT'] # Michael
time_ncoda = np.transpose(netCDF4.num2date(time_ncoda[:],time_ncoda.units))
depth_ncoda = ncncoda.variables['Depth'][:]
lat_ncoda = ncncoda.variables['Latitude'][:]
lon_ncoda = ncncoda.variables['Longitude'][:]

#%% increment North Atlantic 60 m Florence

tini = datetime.datetime(2018,10,7,0,0,0)
tend = datetime.datetime(2018,10,11,0,0,0)

# Get rid off very high values
z=6 #depth level
tincr = temp_incr[0,z,:,:]
tincr[tincr < -4.1] = np.nan 
tincr[tincr > 4.1] = np.nan 

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,\
                 levels=np.linspace(-3,3,13),cmap=plt.get_cmap('seismic'),\
                 vmin=-4.0,vmax=4.0)
cbar = plt.colorbar(cs)
cbar.set_label('($^o$C)',rotation=270,size = 20,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

plt.title('{0} {1} {2} {3} \n {4} {5} {6} {7}'.format('Temperature Increments at ',np.round(depth_ncoda[z]),' m on',time_ncoda[0],\
         'Cryosat SSHA from ',tini ,'to ', tend),fontsize=20)

'''
for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:][0]
    argo_lon = ncargo.variables['LONGITUDE'][:][0]
    ax.plot(argo_lon,argo_lat,'ok',markersize = 5)    
    
ax.plot(argo_lon,argo_lat,'ok',markersize = 5,label='Argo Floats')    
ax.legend(loc='best',fontsize = 20) 

for id in gliders:
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        oktime = np.logical_and(df.index >= tini,df.index <= tend)
        if np.sum(np.where(oktime)) != 0:
            ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
            markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
            markerfacecolor = 'none') 
            
ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
        markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
        markerfacecolor = 'none',label='Gliders') 
ax.legend(loc='best',fontsize = 20)
'''
'''       
l=nc_list[5]
print(l)
ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
ssha = ncjason.ssha
lat_jason = ncjason.lat
lon_jason = ncjason.lon
kw = dict(c=ssha*0, marker='.',cmap=plt.get_cmap('gray'),s=1)
ax.scatter(lon_jason-360, lat_jason, **kw)    
'''

#for l in nc_list:
dates_florence = ['20180911','20180910','20180909','20180908','20180907']
nc_listsub = []
for l in nc_list:
    for m in dates_florence:
        if l.split('_')[6][0:8] == m:
            nc_listsub.append(l) 

for l in nc_listsub:    
    print(l)
    nccryosat = xr.open_dataset(cryosat_folder + l, decode_times=False) 

    ssha = nccryosat.ssha_20_ku
    lat_cryosat = nccryosat.lat_20_ku
    lon_cryosat = nccryosat.lon_20_ku
    #time_jason = ncjason.time
    #time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))
    
    oklon = np.logical_and(lon_cryosat > lon_lim[0],lon_cryosat < lon_lim[-1])
    sshasub = ssha[oklon]
    lonsub = lon_cryosat[oklon]
    latsub = lat_cryosat[oklon]
    oklat = np.logical_and(latsub > lat_lim[0],latsub < lat_lim[-1])
    if l.split('_')[6][6:8]=='11':
        print('gray')
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('gray'),s=1)
    else:
        print('blues')
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('Blues'),s=1)
    ax.scatter(lonsub[oklat], latsub[oklat], **kw) 
    
#ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw,label='Jason2') 
#ax.legend(loc='best',fontsize = 12)    

plt.show()

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}_{4}.png'.format('temp_increment_Atlant',np.round(depth_ncoda[z]),\
          time_ncoda[0].year,time_ncoda[0].month,time_ncoda[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  


#%% Reading ncoda data Michael

ncoda_files = sorted(glob.glob(os.path.join(Dir,'*seatmp*')))

ncncoda = Dataset(ncoda_files[2]) # Michael
#ncncoda = Dataset(ncoda_files[0]) # Florence
temp_incr = ncncoda.variables['pot_temp'][:]

time_ncoda = ncncoda.variables['MT'] # Michael
time_ncoda = np.transpose(netCDF4.num2date(time_ncoda[:],time_ncoda.units))
depth_ncoda = ncncoda.variables['Depth'][:]
lat_ncoda = ncncoda.variables['Latitude'][:]
lon_ncoda = ncncoda.variables['Longitude'][:]

#%% increment North Atlantic 60 m Michael

tini = datetime.datetime(2018,10,7,0,0,0)
tend = datetime.datetime(2018,10,11,0,0,0)

# Get rid off very high values
z=6 #depth level
tincr = temp_incr[0,z,:,:]
tincr[tincr < -4.1] = np.nan 
tincr[tincr > 4.1] = np.nan 

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,\
                 levels=np.linspace(-3,3,13),cmap=plt.get_cmap('seismic'),\
                 vmin=-4.0,vmax=4.0)
cbar = plt.colorbar(cs)
cbar.set_label('($^o$C)',rotation=270,size = 20,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

plt.title('{0} {1} {2} {3} \n {4} {5} {6} {7}'.format('Temperature Increments at ',np.round(depth_ncoda[z]),' m on',time_ncoda[0],\
         'Cryosat SSHA from ',tini ,'to ', tend),fontsize=20)

#for l in nc_list:
dates_michael = ['20181010','20181009','20181008','20181007','20181006']
nc_listsub = []
for l in nc_list:
    for m in dates_michael:
        if l.split('_')[6][0:8] == m:
            nc_listsub.append(l) 

for l in nc_listsub:    
    print(l)
    nccryosat = xr.open_dataset(cryosat_folder + l, decode_times=False) 

    ssha = nccryosat.ssha_20_ku
    lat_cryosat = nccryosat.lat_20_ku
    lon_cryosat = nccryosat.lon_20_ku
    #time_jason = ncjason.time
    #time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))
    
    oklon = np.logical_and(lon_cryosat > lon_lim[0],lon_cryosat < lon_lim[-1])
    sshasub = ssha[oklon]
    lonsub = lon_cryosat[oklon]
    latsub = lat_cryosat[oklon]
    oklat = np.logical_and(latsub > lat_lim[0],latsub < lat_lim[-1])
    if l.split('_')[6][6:8]=='10':
        print('gray')
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('gray'),s=1)
    else:
        print('blues')
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('Blues'),s=1)
    ax.scatter(lonsub[oklat], latsub[oklat], **kw) 
    
#ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw,label='Jason2') 
#ax.legend(loc='best',fontsize = 12)    

plt.show()

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}_{4}.png'.format('temp_increment_Atlant',np.round(depth_ncoda[z]),\
          time_ncoda[0].year,time_ncoda[0].month,time_ncoda[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  