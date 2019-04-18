#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:17:55 2019

@author: aristizabal
"""
#%% User input

#GOFS3.1 output model location
catalog31_ts = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

catalog31_uv = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

date_enterGoM = '2018/10/09/00/00'
date_midGoM = '2018/10/10/00/00'
date_landfallGoM = '2018/10/11/00/00'

# Guld Mexico
lon_lim = [-100,-75]
lat_lim = [14,33]

# Glider data 

# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime
from matplotlib.dates import date2num

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)

timestamp_glider = date2num(time_glider)[0]

#%% GOGF 3.1

GOFS31_ts = xr.open_dataset(catalog31_ts,decode_times=False)
GOFS31_uv = xr.open_dataset(catalog31_uv,decode_times=False)

latt31 = GOFS31_ts['lat'][:]
lonn31 = GOFS31_ts['lon'][:]
tt31 = GOFS31_ts['time']
t31 = netCDF4.num2date(tt31[:],tt31.units) 

# Conversion from glider longitude and latitude to GOFS convention
lon_limG = np.empty((len(lon_lim),))
lon_limG[:] = np.nan
for i in range(len(lon_lim)):
    if lon_lim[i] < 0: 
        lon_limG[i] = 360 + lon_lim[i]
    else:
        lon_limG[i] = lon_lim[i]
lat_limG = lat_lim

### Build the bbox for the xy data
botm  = int(np.where(latt31 > lat_limG[0])[0][0])
top   = int(np.where(latt31 > lat_limG[1])[0][0])
left  = np.where(lonn31 > lon_limG[0])[0][0]
right = np.where(lonn31 > lon_limG[1])[0][0]

lat31= latt31[botm:top]
lon31= lonn31[left:right]

# Conversion from GOFS convention to glider longitude and latitude
lon31g= np.empty((len(lon31),))
lon31g[:] = np.nan
for i in range(len(lon31)):
    if lon31[i] > 180: 
        lon31g[i] = lon31[i] - 360 
    else:
        lon31g[i] = lon31[i]
lat31g = lat31

#%% Best Track Michael

lonMc = np.array([-86.9,-86.7,-86.0,\
                  -85.3,-85.4,-85.07,-85.05,\
                  -85.2,-85.7,-86.1,-86.3,\
                  -86.5,-86.6,-86.3,-85.4,\
                  -84.5,-83.2,-81.7,-80.0])

latMc = np.array([18.4,18.7,19.0,\
                  19.7,20.2,20.9,21.7,\
                  22.7,23.6,24.6,25.6,\
                  26.6,27.8,29.0,30.2,\
                  31.5,32.8,34.1,35.6])

tMc = [                   '2018/10/07/06/00','2018/10/07/12/00','2018/10/07/18/00',\
       '2018/10/08/00/00','2018/10/08/06/00','2018/10/08/12/00','2018/10/08/18/00',\
       '2018/10/09/00/00','2018/10/09/06/00','2018/10/09/12/00','2018/10/09/18/00',\
       '2018/10/10/00/00','2018/10/10/06/00','2018/10/10/12/00','2018/10/10/18/00',
       '2018/10/11/00/00','2018/10/11/06/00','2018/10/11/12/00','2018/10/11/18/00']

timeMc = [None]*len(tMc) 
for x in range(len(tMc)):
    timeMc[x] = datetime.strptime(tMc[x], '%Y/%m/%d/%H/%M') # time in time zone  
        
#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]   

#%% Glider

#tt = datetime.strptime(date_enterGoM,'%Y/%m/%d/%H/%M')
#tt = datetime.strptime(date_midGoM,'%Y/%m/%d/%H/%M')
tt = datetime.strptime(date_landfallGoM,'%Y/%m/%d/%H/%M')

okg = np.where(time_glider.T >= tt)

timeg = time_glider[0,okg[0][0]]
timestampg = timestamp_glider[okg[0][0]]
latg = np.asarray(latglider[0,okg[0][0]])
long = np.asarray(longlider[0,okg[0][0]])

#%% GOFS  3.1
#t = datetime.strptime(date_enterGoM,'%Y/%m/%d/%H/%M')
t = datetime.strptime(date_midGoM,'%Y/%m/%d/%H/%M')
#t = datetime.strptime(date_landfallGoM,'%Y/%m/%d/%H/%M')

oktime31 = np.where(t31 == t)[0][0]
time31 = t31[oktime31]

# loading surface temperature and salinity
sst31 = GOFS31_ts['water_temp'][oktime31,0,botm:top,left:right]
su31 = GOFS31_uv['water_u'][oktime31,0,botm:top,left:right]
sv31 = GOFS31_uv['water_v'][oktime31,0,botm:top,left:right]

#%% Figure sst

kw = dict(levels = np.linspace(26,31,11))

plt.figure(figsize=(10, 8))
plt.contourf(lon31g,lat31g,sst31[:,:],cmap='RdYlBu_r',**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.axis('equal')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)
plt.title('GOFS 3.1 SST and surface velocity on '+str(time31)[0:16],size=22,y=1.03)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

# loop current and warm core rings
'''
plt.plot(-84.3,24.0,'*',color='darkorange',markersize=10)
plt.plot(-86.6,23.5,'*',color='darkorange',markersize=10)
plt.plot(-84.6,24.6,'*',color='darkorange',markersize=10)
plt.plot(-89.8,25.3,'*r',markersize=10)
plt.plot(-86.5,24.6,'*r',markersize=10)
plt.plot(-95.2,24.8,'*r',markersize=10)
plt.plot(-94.0,26.2,'*r',markersize=10)
'''

plt.quiver(lon31g[::2],lat31g[::2],su31[::2,::2],sv31[::2,::2] ,scale=3,scale_units='inches',\
           alpha=0.7)

# Michael track
#plt.plot(lonMc,latMc,'o-',markersize = 10,color = 'dimgray',markeredgecolor='k')

# Michael track and intensity
plt.plot(lonMc,latMc,'.-',markersize = 10,color = 'k',linewidth=2)
plt.plot(lonMc[0],latMc[0],'o',markersize = 10,\
         color = 'white',markeredgecolor='green',
         markeredgewidth=3,label='Tropical Storm')
plt.plot(lonMc[5],latMc[5],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow',label='Cat 1')
plt.plot(lonMc[9],latMc[9],'o',markersize = 10,\
         color = 'orange',markeredgecolor='orange',label='Cat 2')
plt.plot(lonMc[10],latMc[10],'o',markersize = 10,\
         color = 'red',markeredgecolor='red',label='Cat 3')
plt.plot(lonMc[12],latMc[12],'o',markersize = 10,\
         color = 'purple',markeredgecolor='purple',label='Cat 4')
plt.legend(loc='upper left',fontsize=14)
plt.plot(lonMc[0:5],latMc[0:5],'o',markersize = 10,\
         color = 'white',markeredgecolor='green',markeredgewidth=3)
plt.plot(lonMc[5:9],latMc[5:9],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow')
plt.plot(lonMc[9],latMc[9],'o',markersize = 10,\
         color = 'orange',markeredgecolor='orange')
plt.plot(lonMc[10:12],latMc[10:12],'o',markersize = 10,\
         color = 'red',markeredgecolor='red')
plt.plot(lonMc[12:15],latMc[12:15],'o',markersize = 10,\
         color = 'purple',markeredgecolor='purple')
plt.plot(lonMc[15],latMc[15],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow')
plt.plot(lonMc[16:],latMc[16:],'o',markersize = 10,\
         color = 'white',markeredgecolor='green')


props = dict(boxstyle='square', facecolor='white', alpha=0.5)
for x in range(0, len(tMc)-1, 3):
    plt.text(lonMc[x]+0.4,latMc[x],timeMc[x].strftime('%d, %H:%M'),\
             size = 16,color='k',weight='bold',bbox=props)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'  
file = 'GOFS31_sst_ssv_Michael_'+str(t)+'.png'
plt.savefig(folder+file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure sst zoom in

kw = dict(levels = np.linspace(26,31,11))

plt.figure(figsize=(10, 8))
plt.contourf(lon31g,lat31g,sst31[:,:],cmap='RdYlBu_r',**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
#plt.axis('equal')

plt.title('GOFS 3.1 SST and surface velocity on '+str(time31)[0:16],size=22,y=1.03)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

# Glider position
plt.plot(long,latg,'*',color='g',markersize = 18,markeredgecolor='k',\
         markeredgewidth=2)
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(long+0.3,latg,'ng288',\
             size = 16,color='k',weight='bold',bbox=props)


# loop current and warm core rings
'''
plt.plot(-84.3,24.0,'*',color='darkorange',markersize=10)
plt.plot(-86.6,23.5,'*',color='darkorange',markersize=10)
plt.plot(-84.6,24.6,'*',color='darkorange',markersize=10)
plt.plot(-89.8,25.3,'*r',markersize=10)
plt.plot(-86.5,24.6,'*r',markersize=10)
plt.plot(-95.2,24.8,'*r',markersize=10)
plt.plot(-94.0,26.2,'*r',markersize=10)
'''

plt.quiver(lon31g[::2],lat31g[::2],su31[::2,::2],sv31[::2,::2] ,scale=2,scale_units='inches',\
           alpha=0.7)

# Michael track
#plt.plot(lonMc,latMc,'o-',markersize = 10,color = 'dimgray',markeredgecolor='k')

# Michael track and intensity
plt.plot(lonMc,latMc,'.-',markersize = 10,color = 'k',linewidth=4)
plt.plot(lonMc[0],latMc[0],'o',markersize = 10,\
         color = 'white',markeredgecolor='green',
         markeredgewidth=3,label='Tropical Storm')
plt.plot(lonMc[5],latMc[5],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow',label='Cat 1')
plt.plot(lonMc[9],latMc[9],'o',markersize = 10,\
         color = 'orange',markeredgecolor='orange',label='Cat 2')
plt.plot(lonMc[10],latMc[10],'o',markersize = 10,\
         color = 'red',markeredgecolor='red',label='Cat 3')
plt.plot(lonMc[12],latMc[12],'o',markersize = 10,\
         color = 'purple',markeredgecolor='purple',label='Cat 4')
#legend = plt.legend(loc='upper left',fontsize=14,bbox_to_anchor=(0, -0.1))
plt.plot(lonMc[0:5],latMc[0:5],'o',markersize = 15,\
         color = 'white',markeredgecolor='green',markeredgewidth=3)
plt.plot(lonMc[5:9],latMc[5:9],'o',markersize = 15,\
         color = 'yellow',markeredgecolor='yellow')
plt.plot(lonMc[9],latMc[9],'o',markersize = 15,\
         color = 'orange',markeredgecolor='orange')
plt.plot(lonMc[10:12],latMc[10:12],'o',markersize = 15,\
         color = 'red',markeredgecolor='red')
plt.plot(lonMc[12:15],latMc[12:15],'o',markersize = 15,\
         color = 'purple',markeredgecolor='purple')
plt.plot(lonMc[15],latMc[15],'o',markersize = 15,\
         color = 'yellow',markeredgecolor='yellow')
plt.plot(lonMc[16:],latMc[16:],'o',markersize = 15,\
         color = 'white',markeredgecolor='green')


props = dict(boxstyle='square', facecolor='white', alpha=0.5)
for x in range(7, 7+len(tMc[7:16])):
    plt.text(lonMc[x]-1.8,latMc[x],timeMc[x].strftime('%d, %H:%M'),\
             size = 16,color='k',weight='bold',bbox=props)
    
plt.xlim(-89.5,-82)
plt.ylim(22.5,31.8)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'  
file = 'GOFS31_sst_ssv_Michael_zoomin_'+str(t)+'.png'
plt.savefig(folder+file,bbox_inches = 'tight',pad_inches = 0.1)     
plt.show()

#%% save variables
'''
import pickle
 
file = 'GOFS31_GoM_' + str(time31)[0:16] + '.pickle'

with open(file, 'wb') as f:
    pickle.dump([lon31g,lat31g,sst31,su31,sv31], f)
    
#%% open data from pickle file

import pickle 
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Code/'   
with open(folder + 'GOFS31_GoM_2018-10-07 18:00.pickle', 'rb') as f:
     lon31g,lat31g,sst31,su31,sv31 = pickle.load(f)    

'''