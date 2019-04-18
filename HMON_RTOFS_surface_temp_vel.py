#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:10:33 2019

@author: aristizabal
"""

#%% User input  

# Directories where RTOFS files reside 
Dir= '/Volumes/aristizabal/ncep_model//HMON-HYCOM_Michael/'
Dir_graph = '/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts/Figures/'

# RTOFS grid file name
gridfile = 'hwrf_rtofs_hat10.basin.regional.grid'

# RTOFS a/b file name
prefix_ab = 'michael14l.2018100718.hmon_rtofs_hat10_3z'

date_end = '2018-10-13T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# Guld Mexico
lon_lim = [-100,-75]
lat_lim = [14,33]

date_enterGoM = '2018/10/09/00/00'
date_midGoM = '2018/10/10/00/00'
date_landfallGoM = '2018/10/11/00/00'

# Glider data 

# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc'

#%% Modules to read HYCOM output 

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')

from utils4HYCOM import readBinz, readgrids

import os
import os.path
import glob
from datetime import datetime
from matplotlib.dates import date2num, num2date
import matplotlib.pyplot as plt
import numpy as np
import netCDF4

import xarray as xr

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)

timestamp_glider = date2num(time_glider)[0]

#%% Reading HYCOM lat and lon

lines_grid=[line.rstrip() for line in open(Dir+gridfile+'.b')]
hlon = np.array(readgrids(Dir+gridfile,'plon:',[0]))
hlat = np.array(readgrids(Dir+gridfile,'plat:',[0]))

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
botm  = int(np.where(hlat[:,0] > lat_limG[0])[0][0])
top   = int(np.where(hlat[:,0] > lat_limG[1])[0][0])
left  = np.where(hlon[0,:] > lon_limG[0])[0][0]
right = np.where(hlon[0,:] > lon_limG[1])[0][0]

Hlat= hlat[botm:top,left:right]
Hlon= hlon[botm:top,left:right]

# Conversion from GOFS convention to glider longitude and latitude
Hlong= np.empty((Hlon.shape[0],Hlon.shape[1]))
Hlong[:] = np.nan
for i in range(Hlon.shape[1]):
    if Hlon[0,i] > 180: 
        Hlong[:,i] = Hlon[:,i] - 360 
    else:
        Hlong[:,i] = Hlon[:,i]
Hlatg = Hlat

#%% Reading HYCOM ab files

afiles = sorted(glob.glob(os.path.join(Dir,prefix_ab+'*.a')))

sst_HYCOM = np.empty((len(afiles),Hlon.shape[0],Hlon.shape[1]))
sst_HYCOM[:] = np.nan
su_HYCOM = np.empty((len(afiles),Hlon.shape[0],Hlon.shape[1]))
su_HYCOM[:] = np.nan
sv_HYCOM = np.empty((len(afiles),Hlon.shape[0],Hlon.shape[1]))
sv_HYCOM[:] = np.nan
time_HYCOM = []
for x, file in enumerate(afiles):
    print(x)
    lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('.')[1][0:4])
    month = int(file.split('.')[1][4:6])
    day = int(file.split('.')[1][6:8])
    hour = int(file.split('.')[1][8:10])
    dt = int(file.split('.')[3][1:])
    timestamp_HYCOM = date2num(datetime(year,month,day,hour)) + dt/24
    time_HYCOM.append(num2date(timestamp_HYCOM))
    
    # Reading 3D variable from binary file 
    temp_HYCOM = readBinz(file[:-2],'3z','temp')
    sst_HYCOM[x,:,:] = temp_HYCOM[botm:top,left:right,0]
    
    u_HYCOM = readBinz(file[:-2],'3z','u-veloc.')
    su_HYCOM[x,:,:] = u_HYCOM[botm:top,left:right,0]
    
    v_HYCOM = readBinz(file[:-2],'3z','v-veloc.')
    sv_HYCOM[x,:,:] = v_HYCOM[botm:top,left:right,0]
    
time_HYCOM = np.asarray(time_HYCOM)
timestamp_HYCOM = date2num(time_HYCOM)

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

#%% Best Track Michael path

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
    
#%% Glider

#tt = datetime.strptime(date_enterGoM,'%Y/%m/%d/%H/%M')
#tt = datetime.strptime(date_midGoM,'%Y/%m/%d/%H/%M')
tt = datetime.strptime(date_landfallGoM,'%Y/%m/%d/%H/%M')

okg = np.where(time_glider.T >= tt)

timeg = time_glider[0,okg[0][0]]
timestampg = timestamp_glider[okg[0][0]]
latg = np.asarray(latglider[0,okg[0][0]])
long = np.asarray(longlider[0,okg[0][0]])

#%% HWRF-POM

nt = np.where(date2num(time_HYCOM) == date2num(tt))[0][0]    


#%% Figure sst

su_HYCOM0 = su_HYCOM[nt,:,:]
su_HYCOM0[np.abs(su_HYCOM0) > 1000] = np.nan
sv_HYCOM0 = sv_HYCOM[nt,:,:]
sv_HYCOM0[np.abs(sv_HYCOM0) > 1000] = np.nan

kw = dict(levels = np.linspace(26,31,11))

plt.figure(figsize=(10, 8))
plt.contourf(Hlong,Hlatg,sst_HYCOM[nt,:,:],cmap='RdYlBu_r',**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.axis('equal')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)
plt.title('HMON-HYCOM SST and surface velocity on '+str(time_HYCOM[nt])[0:16],size=22,y=1.03)
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

plt.quiver(Hlong[::2,::2],Hlatg[::2,::2],su_HYCOM0[::2,::2]/100,sv_HYCOM0[::2,::2]/100 ,scale=3,scale_units='inches',\
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
         color = 'white',markeredgecolor='green',markeredgewidth=3,)

props = dict(boxstyle='square', facecolor='white', alpha=0.5)
for x in range(0, len(tMc)-1, 3):
    plt.text(lonMc[x]+0.4,latMc[x],timeMc[x].strftime('%d, %H:%M'),\
             size = 16,color='k',weight='bold',bbox=props)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'  
file = 'HMON_HYCOM_sst_ssv_Michael_'+str(tt)+'.png'
plt.savefig(folder+file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure sst zoom in

su_HYCOM0 = su_HYCOM[nt,:,:]
su_HYCOM0[np.abs(su_HYCOM0) > 1000] = np.nan
sv_HYCOM0 = sv_HYCOM[nt,:,:]
sv_HYCOM0[np.abs(sv_HYCOM0) > 1000] = np.nan

#kw = dict(levels = np.linspace(27,31,9))
kw = dict(levels = np.linspace(26,31,11))

plt.figure(figsize=(10, 8))
plt.contourf(Hlong,Hlatg,sst_HYCOM[nt,:,:],cmap='RdYlBu_r',**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.title('HMON-HYCOM SST and surface velocity on '+str(time_HYCOM[nt])[0:16],size=22,y=1.03)
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


plt.quiver(Hlong[::2,::2],Hlatg[::2,::2],su_HYCOM0[::2,::2]/100,sv_HYCOM0[::2,::2]/100 ,scale=2,scale_units='inches',\
           alpha=0.7)

# Michael track
#plt.plot(lonMc,latMc,'o-',markersize = 10,color = 'dimgray',markeredgecolor='k')

# Michael track and intensity
plt.plot(lonMc,latMc,'.-',markersize = 10,color = 'k',linewidth=4)
plt.plot(lonMc[0],latMc[0],'o',markersize = 10,\
         color = 'white',markeredgecolor='green',
         markeredgewidth=3,label='Tropcal Storm')
plt.plot(lonMc[5],latMc[5],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow',label='Cat 1')
plt.plot(lonMc[9],latMc[9],'o',markersize = 10,\
         color = 'orange',markeredgecolor='orange',label='Cat 2')
plt.plot(lonMc[10],latMc[10],'o',markersize = 10,\
         color = 'red',markeredgecolor='red',label='Cat 3')
plt.plot(lonMc[12],latMc[12],'o',markersize = 10,\
         color = 'purple',markeredgecolor='purple',label='Cat 4')
#plt.legend(loc='upper left')
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
file = 'HMON_HYCOM_sst_ssv_Michael_zoomin_'+str(tt)+'.png'
#plt.savefig(folder+file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()