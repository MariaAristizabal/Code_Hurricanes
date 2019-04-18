#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:50:17 2018

@author: aristizabal
"""

#%% User input

# (Virgin Islands)

lon_lim = [-68,-64];
lat_lim = [15,20];
#gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc'
gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc';

# GOFS 3.1 
catalog31_uv = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z';

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

#%%

import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr 
import matplotlib.dates as mdates

#%% Reading glider data
#ncglider = xr.open_dataset(gdata)
ncglider = Dataset(gdata)
latglider = ncglider.variables['lat_uv'][0,:]
longlider = ncglider.variables['lon_uv'][0,:]
#timeglid = ncglider.variables['time']
#timeglider = netCDF4.num2date(timeglid[0,:],timeglid.units)
timeglid = ncglider.variables['time_uv']
#timeglider = netCDF4.num2date(timeglid[0,:],timeglid.units)
timeglider = netCDF4.num2date(timeglid[0,:],'seconds since 1970-01-01T00:00:00Z')
uglider = ncglider.variables['u'][0,:]
vglider = ncglider.variables['v'][0,:]
inst_id = ncglider.id.split('_')[0]

#%% Reading GOFS3.1 data

GOFS31 = xr.open_dataset(catalog31_uv,decode_times=False)
#GOFS31 = Dataset(catalog31_uv,decode_times=False)

lat31 = GOFS31.variables['lat'][:]
lon31 = GOFS31.variables['lon'][:]
depth31 = GOFS31.variables['depth'][:]
tt31 = GOFS31.variables['time']
#t31 = netCDF4.num2date(tt31[:],tt31.units) 
time31 = netCDF4.num2date(tt31[:],'hours since 2000-01-01 00:00:00') 


#%%
#tmin = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
#tmax = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

tmin = timeglider[0]
tmax = timeglider[-1]

oktime31 = np.where(np.logical_and(time31 >= tmin, time31 <= tmax))

# Conversion from glider longitude and latitude to GOFS convention
target_lon = np.empty((len(longlider),))
target_lon[:] = np.nan
for i in range(len(longlider)):
    if longlider[i] < 0: 
        target_lon[i] = 360 + longlider[i]
    else:
        target_lon[i] = longlider[i]
target_lat = latglider[:]

#%%
 
#Changing times to timestamp
tstamp_glider = [mdates.date2num(timeglider[i]) for i in np.arange(len(timeglider))]
tstamp_31 = [mdates.date2num(time31[i]) for i in np.arange(len(time31))]
tt31 = np.asarray(tstamp_31)[oktime31]

# interpolating glider lon and lat to lat and lon on model time
sublon31 = np.interp(tt31,tstamp_glider,target_lon)
sublat31 = np.interp(tt31,tstamp_glider,target_lat)

# getting the model grid positions for sublonm and sublatm
oklon31 = np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
oklat31 = np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)

#%%

u31 = np.empty((len(depth31),len(oktime31[0])))
u31[:] = np.nan
v31 = np.empty((len(depth31),len(oktime31[0])))
v31[:] = np.nan
for i in range(len(oktime31[0])):
    print(len(oktime31[0]),' ',i)
    u31[:,i] = GOFS31.variables['water_u'][oktime31[0][i],:,oklat31[i],oklon31[i]]
    v31[:,i] = GOFS31.variables['water_v'][oktime31[0][i],:,oklat31[i],oklon31[i]]

#target_temp31[target_temp31 < -100] = np.nan

#%% Save variables
  
import pickle

ff = 'depth_aver_vel_'+inst_id.split('-')[0]+'.pickle'
myfile = open(ff, 'wb')


names = ['latglider','longlider','timeglider','uglider','vglider','inst_id',\
      'lat31','lon31','depth31','time31','u31','v31']
var = [latglider,longlider,timeglider,uglider,vglider,inst_id,\
      lat31,lon31,depth31,time31,u31,v31]
for l in var:
    pickle.dump(l,myfile)
myfile.close()

#%% open the file
'''
myfile = open('depth_aver_vel_ng467.pickle', 'rb')
for n in names:
    n = pickle.load(myfile)
'''

#%%  Velocity plot glider
    
Ug,oku = np.unique(uglider,return_index=True)
Vg = vglider[oku]
Xg = timeglider[oku]
Yg = np.zeros(len(Xg))

fig = plt.figure()
fig.set_size_inches(16.53,6.41)
Q = plt.quiver(Xg,Yg,Ug,Vg, units='width',color='b')
qk = plt.quiverkey(Q, 0.9, 0.9, 0.1, r'$0.1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.title('Depth Average Velocity ',inst_id,size=20)

#%%  Velocity plot model

X31 = time31[oktime31]
Y31 = np.zeros(len(X31))
# Depth average velocoty in to 200 m
okd = depth31 < 200
U31 = np.mean(u31[okd,:],0)
V31 = np.mean(v31[okd,:],0)


fig = plt.figure()
fig.set_size_inches(16.53,6.41)
Q = plt.quiver(X31,Y31,U31,V31, units='width',color='r')
qk = plt.quiverkey(Q, 0.9, 0.9, 0.1, r'$0.1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')

#%% Velocity plot glider vs model
ttg = np.asarray(tstamp_glider)[oku]
tt31 = np.asarray(tstamp_31)[oktime31]

U31_interp = np.interp(ttg,tt31,U31)
V31_interp = np.interp(ttg,tt31,V31)

fig,ax = plt.subplots()
fig.set_size_inches(12.48,2.9)
Q = plt.quiver(Xg,Yg,Ug,Vg, units='width',color='b',label=inst_id.split('-')[0],alpha=0.8)
plt.quiver(Xg,Yg,U31_interp,V31_interp, units='width',color='r',label='GOFS 3.1',alpha=0.5)
qk = plt.quiverkey(Q, 0.9, 0.9, 0.1, r'$0.1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.legend()
plt.title('Depth Average Velocity ',size=20)
xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis.label.set_size(30)
plt.yticks([])
plt.xticks(fontsize=12)

#file = folder + '{0}_{1}_{2}_{3}.png'.format('Depth_avg_velocity',inst_id.split('-')[0]\
#                     timeglider[0],timeglider[-1])

file = folder + 'Depth_Average_Velocity_' + inst_id.split('-')[0]

plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.2) 