#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:45:46 2018

@author: aristizabal
"""

#%% User input

# SAB + MAB
lon_lim = [-81,-70]
lat_lim = [30,42]

#Initial and final date

dateini = '2018/09/10/00/00'
dateend = '2018/09/17/00/00'

# RAMSES
ramses_data = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc'

# Pelagia
pelag_data = 'http://data.ioos.us/thredds/dodsC/deployments/secoora/pelagia-20180910T0000/pelagia-20180910T0000.nc3.nc'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# NDBC buoys

#B41025 = 'https://www.ndbc.noaa.gov/data/realtime2/41025.txt'
B41025 = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/41025/41025h9999.nc'

#B41037 = 'https://www.ndbc.noaa.gov/data/realtime2/41037.txt'
B41037 = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/41037/41037h9999.nc'

#B41002 = 'https://www.ndbc.noaa.gov/data/realtime2/41002.txt'  # no atmospheric temperature
B41002 = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/41002/41002h9999.nc'

#B41004 = 'https://www.ndbc.noaa.gov/data/realtime2/41004.txt'
B41004 = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/41004/41004h9999.nc'

#GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

#%%

import netCDF4
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import datetime
import pytz
import numpy as np
#import xarray as xr 

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%% Reading glider data

# Ramses
ncramses = Dataset(ramses_data)
lat_ramses = ncramses.variables['latitude'][:]
lon_ramses = ncramses.variables['longitude'][:]
time_rams = ncramses.variables['time']
time_ramses = netCDF4.num2date(time_rams[:],time_rams.units)

# pelagia
ncpelag = Dataset(pelag_data)
lat_pelag = ncpelag.variables['latitude'][:]
lon_pelag = ncpelag.variables['longitude'][:]
time_pel = ncpelag.variables['time']
time_pelag = netCDF4.num2date(time_pel[:],time_pel.units)

#%% Reading NDBC data from .txt file (HORRIBLE!!)
'''
data = requests.get(B41025)
data = data.text.split('\n')

date = [None]*(len(data)-2) 
#wdir = [None]*(len(data)-2) 
#wspd = [None]*(len(data)-2) 
#pres = [None]*(len(data)-2) 
#atmp = [None]*(len(data)-2) 
#wtmp = [None]*(len(data)-2) 
wtmp = np.empty(len(data)-2)
wtmp[:] = np.nan
for i in range(len(data)-4):
    print(i)
    dat=list(filter(None,data[i+2].split(' ')))[0]+'/'+list(filter(None,data[i+2].split(' ')))[1]+'/'+ \
        list(filter(None,data[i+2].split(' ')))[2]+'/'+list(filter(None,data[i+2].split(' ')))[3]+'/'+ \
        list(filter(None,data[i+2].split(' ')))[4]
    date[i] = datetime.datetime.strptime(dat, '%Y/%m/%d/%H/%M') #Time already in UTC
    #wdir[i] = float(list(filter(None,data[i+2].split(' ')))[5])
    #wspd[i] = float(list(filter(None,data[i+2].split(' ')))[7])
    #pres[i] = float(list(filter(None,data[i+2].split(' ')))[12])
    #atmp[i] = float(list(filter(None,data[i+2].split(' ')))[13])
    if list(filter(None,data[i+2].split(' ')))[14] != 'MM':
        wtmp[i] = float(list(filter(None,data[i+2].split(' ')))[14])
'''

#%% Reading NDBC data from the thredds server
 
# NDBC 41025       
ncB41025 = Dataset(B41025)
lat_41025 = ncB41025.variables['latitude'][:] 
lon_41025 = ncB41025.variables['longitude'][:]
wdir_41025 = ncB41025.variables['wind_dir'][:]
wspd_41025 = ncB41025.variables['wind_spd'][:]
pres_41025 = ncB41025.variables['air_pressure'][:]
atmp_41025 = ncB41025.variables['air_temperature'][:]
wtmp_41025 =ncB41025.variables['sea_surface_temperature'][:]    
time_41025 = ncB41025.variables['time']
time_41025 = netCDF4.num2date(time_41025[:],time_41025.units) 

# NDBC 41037       
ncB41037 = Dataset(B41037)
lat_41037 = ncB41037.variables['latitude'][:] 
lon_41037 = ncB41037.variables['longitude'][:]
wdir_41037 = ncB41037.variables['wind_dir'][:]
wspd_41037 = ncB41037.variables['wind_spd'][:]
pres_41037 = ncB41037.variables['air_pressure'][:]
atmp_41037 = ncB41037.variables['air_temperature'][:]
wtmp_41037 =ncB41037.variables['sea_surface_temperature'][:]    
time_41037 = ncB41037.variables['time']
time_41037 = netCDF4.num2date(time_41037[:],time_41037.units) 

# NDBC 41002       
ncB41002 = Dataset(B41002)
lat_41002 = ncB41002.variables['latitude'][:] 
lon_41002 = ncB41002.variables['longitude'][:]
wdir_41002 = ncB41002.variables['wind_dir'][:]
wspd_41002 = ncB41002.variables['wind_spd'][:]
pres_41002 = ncB41002.variables['air_pressure'][:]
atmp_41002 = ncB41002.variables['air_temperature'][:]
wtmp_41002 =ncB41002.variables['sea_surface_temperature'][:]    
time_41002 = ncB41002.variables['time']
time_41002 = netCDF4.num2date(time_41002[:],time_41002.units)     

# NDBC 41004      
ncB41004 = Dataset(B41004)
lat_41004 = ncB41004.variables['latitude'][:] 
lon_41004 = ncB41004.variables['longitude'][:]
wdir_41004 = ncB41004.variables['wind_dir'][:]
wspd_41004 = ncB41004.variables['wind_spd'][:]
pres_41004 = ncB41004.variables['air_pressure'][:]
atmp_41004 = ncB41004.variables['air_temperature'][:]
wtmp_41004 =ncB41004.variables['sea_surface_temperature'][:]    
time_41004 = ncB41004.variables['time']
time_41004 = netCDF4.num2date(time_41004[:],time_41004.units) 

#%% Reading GOFS3.1 output

GOFS31 = Dataset(catalog31)

lat31 = GOFS31.variables['lat'][:]
lon31 = GOFS31.variables['lon'][:]
depth = GOFS31.variables['depth'][:]
time31 = GOFS31.variables['time']
time31 = netCDF4.num2date(time31[:],time31.units) 

date_ini = datetime.datetime.strptime(dateini, '%Y/%m/%d/%H/%M') #Time already in UTC
date_end = datetime.datetime.strptime(dateend, '%Y/%m/%d/%H/%M') #Time already in UTC
oktime31 = np.where(np.logical_and(time31 > date_ini, time31 < date_end))

# Conversion from glider longitude and latitude to GOFS convention
if lon_41025 < 0: 
    target_lon = 360 + lon_41025
else:
    target_lon = lon_41025
target_lat = lat_41025 

oklon31 = np.round(np.interp(target_lon,lon31,np.arange(len(lon31))))
oklat31 = np.round(np.interp(target_lat,lat31,np.arange(len(lat31))))
target_temp31_41025 = GOFS31.variables['water_temp'][oktime31[0],1,oklat31,oklon31]

if lon_41037 < 0: 
    target_lon = 360 + lon_41037
else:
    target_lon = lon_41037
target_lat = lat_41037 

oklon31 = np.round(np.interp(target_lon,lon31,np.arange(len(lon31))))
oklat31 = np.round(np.interp(target_lat,lat31,np.arange(len(lat31))))
target_temp31_41037 = GOFS31.variables['water_temp'][oktime31[0],1,oklat31,oklon31]

if lon_41002 < 0: 
    target_lon = 360 + lon_41002
else:
    target_lon = lon_41002
target_lat = lat_41002

oklon31 = np.round(np.interp(target_lon,lon31,np.arange(len(lon31))))
oklat31 = np.round(np.interp(target_lat,lat31,np.arange(len(lat31))))
target_temp31_41002 = GOFS31.variables['water_temp'][oktime31[0],1,oklat31,oklon31]

if lon_41004 < 0: 
    target_lon = 360 + lon_41004
else:
    target_lon = lon_41004
target_lat = lat_41004

oklon31 = np.round(np.interp(target_lon,lon31,np.arange(len(lon31))))
oklat31 = np.round(np.interp(target_lat,lat31,np.arange(len(lat31))))
target_temp31_41004 = GOFS31.variables['water_temp'][oktime31[0],1,oklat31,oklon31]

'''
ds = xr.open_dataset(GOFS31)  # NetCDF or OPeNDAP URL
# Extract a dataset closest to specified point
dsloc = ds.sel(lon=target_lon, lat=target_lat, 1,  method='nearest')
# select a variable to plot
dsloc['dswrfsfc'].plot() 
'''

#%% Tentative Florence path

lonFl = np.array([-61,-63,-66,-69,-72,-77,-78,-79])
latFl = np.array([25,26,27,29,31,34,36,37])
tFl = ['2018/09/10/17/00','2018/09/11/10/00','2018/09/11/14/00','2018/09/12/02/00',
       '2018/09/12/14/00','2018/09/13/14/00','2018/09/14/14/00','2018/09/15/14/00']

# Convert time to UTC
pst = pytz.timezone('America/New_York') # time zone
utc = pytz.UTC 

timeFl = [None]*len(tFl) 
for x in range(len(tFl)):
    d = datetime.datetime.strptime(tFl[x], '%Y/%m/%d/%H/%M') # time in time zone
    d = pst.localize(d) # add time zone to date
    d = d.astimezone(utc)
    timeFl[x] = d.astimezone(utc)
    print(timeFl[x].strftime('%d, %H %M'))

#%% Figure Florence path and data available
    
siz=16

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='k')
ax.plot(lon_ramses[0,:],lat_ramses[0,:],'*',markersize = 10, label = 'Gliders', color = 'C0')
ax.plot(lon_pelag[0,:],lat_pelag[0,:],'*',markersize = 10,color = 'C0')
ax.plot(lonFl,latFl,'o-',markersize = 10,label = 'Florence Tentative Track',color = 'indianred')
ax.plot(lon_41025[:],lat_41025[:],'^',markersize = 10, color = 'g',label = 'NDBC Buoys')
ax.plot(lon_41037[:],lat_41037[:],'^',markersize = 10, color = 'g')
ax.plot(lon_41002[:],lat_41002[:],'^',markersize = 10, color = 'g')
ax.plot(lon_41004[:],lat_41004[:],'^',markersize = 10, color = 'g')

plt.axis('equal')
plt.axis([-82,-73,30,40])

for x in range(len(tFl)):
    ax.text(lonFl[x],latFl[x],timeFl[x].strftime('%d, %H:%M'),size = siz)
ax.text(lon_41025[:],lat_41025[:],getattr(ncB41025, 'station'),size = siz)
ax.text(lon_41037[:],lat_41037[:]-0.5,getattr(ncB41037, 'station'),size = siz)
ax.text(lon_41002[:],lat_41002[:],getattr(ncB41002, 'station'),size = siz)
ax.text(lon_41004[:],lat_41004[:],getattr(ncB41004, 'station'),size = siz)
ax.text(np.mean(lon_ramses),np.mean(lat_ramses),'RAMSES',size = siz)
ax.text(np.mean(lon_pelag),np.mean(lat_pelag),'PELAGIA',size = siz)

ax.set_xlabel('Longitude ($^o$)', size=siz)
ax.set_ylabel('Latitude ($^o$)', size=siz)
ax.tick_params(axis='x', labelsize=siz)
ax.tick_params(axis='y', labelsize=siz)
ax.legend(loc='upper left',fontsize = siz)
#ax.grid(True)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Florence/Florence_track.png")
plt.show()

#%% Air Temperature vs Water Temperature

siz=16
tmin = datetime.datetime.strptime('2018/09/10','%Y/%m/%d')
tmax = datetime.datetime.strptime('2018/09/17','%Y/%m/%d')

#temp_min = np.min(wtmp_41025[ok_time,0,0])
#temp_max = np.max(wtmp_41025[ok_time,0,0])
temp_min = 20
temp_max = 30

tFlor = datetime.datetime.strptime('2018/09/13/18/00','%Y/%m/%d/%H/%M')
tFlor = np.tile(tFlor,len(np.arange(temp_min,temp_max)))

fig, plt.subplots(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='w')

ok_time = np.logical_and(time_41025[:] > tmin, time_41025[:] < tmax)
ax1 = plt.subplot(221)
ax1.plot(time_41025[ok_time],wtmp_41025[ok_time,0,0],label = 'Water Temp')
ax1.plot(time_41025[ok_time],atmp_41025[ok_time,0,0],label = 'Air Temp')
ax1.plot(tFlor,np.arange(temp_min,temp_max))
ax1.axis([tmin,tmax,temp_min,temp_max])
ax1.set_title('NDBC 41025', size=siz)
ax1.set_ylabel('Temperature ($^o$C)', size=siz)
ax1.tick_params(axis='x', labelsize=0)
ax1.tick_params(axis='y', labelsize=siz)
ax1.set_yticks(np.arange(temp_min, temp_max, 2))
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax1.legend(loc='lower left',fontsize = siz)
ax1.grid(True,color='gainsboro')

ok_time = np.logical_and(time_41037[:] > tmin, time_41037[:] < tmax)
ax2 = plt.subplot(222)
ax2.plot(time_41037[ok_time],wtmp_41037[ok_time,0,0],label = 'Water Temp')
ax2.plot(time_41037[ok_time],atmp_41037[ok_time,0,0],label = 'Air Temp')
ax2.plot(tFlor,np.arange(temp_min,temp_max))
ax2.axis([tmin,tmax,temp_min,temp_max])
ax2.set_title('NDBC 41037', size=siz)
ax2.tick_params(axis='x', labelsize=0)
ax2.tick_params(axis='y', labelsize=0)
ax2.set_yticks(np.arange(temp_min, temp_max, 2))
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax2.legend(loc='lower left',fontsize = siz)
ax2.grid(True,color='gainsboro')

ok_time = np.logical_and(time_41002[:] > tmin, time_41002[:] < tmax)
ax3 = plt.subplot(223)
ax3.plot(time_41002[ok_time],wtmp_41002[ok_time,0,0],label = 'Water Temp')
ax3.plot(time_41002[ok_time],atmp_41002[ok_time,0,0],label = 'Air Temp')
ax3.plot(tFlor,np.arange(temp_min,temp_max))
ax3.axis([tmin,tmax,temp_min,temp_max])
ax3.set_title('NDBC 41002', size=siz)
ax3.set_xlabel('Sep. 2018', size=siz)
ax3.set_ylabel('Temperature ($^o$C)', size=siz)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=siz)
ax3.set_yticks(np.arange(temp_min, temp_max, 2))
ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax3.legend(loc='lower left',fontsize = siz)
ax3.grid(True,color='gainsboro')

ok_time = np.logical_and(time_41004[:] > tmin, time_41004[:] < tmax)
ax4 = plt.subplot(224)
ax4.plot(time_41004[ok_time],wtmp_41004[ok_time,0,0],label = 'Water Temp')
ax4.plot(time_41004[ok_time],atmp_41004[ok_time,0,0],label = 'Air Temp')
ax4.plot(tFlor,np.arange(temp_min,temp_max))
ax4.axis([tmin,tmax,temp_min,temp_max])
ax4.set_title('NDBC 41004', size=siz)
ax4.set_xlabel('Sep. 2018', size=siz)
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=0)
ax4.set_yticks(np.arange(temp_min, temp_max, 2))
ax4.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax4.legend(loc='lower left',fontsize = siz)
ax4.grid(True,color='gainsboro')

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Florence/water_air_temp.png")
plt.show()


#%% Sea level pressure

siz=16
tmin = datetime.datetime.strptime('2018/09/10','%Y/%m/%d')
tmax = datetime.datetime.strptime('2018/09/17','%Y/%m/%d')
pres_min = np.min(pres_41037[:])
pres_max = np.max(pres_41037[:])

tFlor = datetime.datetime.strptime('2018/09/13/18/00','%Y/%m/%d/%H/%M')
tFlor = np.tile(tFlor,len(np.arange(pres_min,pres_max)))

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')

ax1.plot(time_41025[:],pres_41025[:,0,0],label = getattr(ncB41025, 'station'))
ax1.plot(time_41037[:],pres_41037[:,0,0],label = getattr(ncB41037, 'station'))
ax1.plot(time_41002[:],pres_41002[:,0,0],label = getattr(ncB41002, 'station'))
ax1.plot(time_41004[:],pres_41004[:,0,0],label = getattr(ncB41004, 'station'))
#ax1.plot(tFlor,np.arange(pres_min,pres_max))
ax1.axis([tmin,tmax,pres_min,pres_max])
ax1.set_ylabel('Sea Level Pressure (hpa)', size=siz)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=siz)
ax1.set_xlabel('Sep. 2018', size=siz)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax1.legend(loc='lower left',fontsize = siz)
ax1.grid(True,color='gainsboro')

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Florence/se_level_pressure.png")
plt.show()

#%% Wind speed and direction

siz=16
tmin = datetime.datetime.strptime('2018/09/10','%Y/%m/%d')
tmax = datetime.datetime.strptime('2018/09/17','%Y/%m/%d')

vel_min = 0
vel_max = 30
dir_min = 0
dir_max = 380

tFlor = datetime.datetime.strptime('2018/09/13/18/00','%Y/%m/%d/%H/%M')
tFlor = np.tile(tFlor,len(np.arange(temp_min,temp_max)))

fig, plt.subplots(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='w')

ok_time = np.logical_and(time_41025[:] > tmin, time_41025[:] < tmax)
ax1 = plt.subplot(221)
ax1.plot(time_41025[ok_time],wdir_41025[ok_time,0,0],\
         label = 'Wind Direction',color = 'C0')
ax1.axis([tmin,tmax,dir_min,dir_max])
ax1.set_title('NDBC 41025', size=siz)
ax1.set_ylabel('Wind Direction ($^o$)', size=siz,color = 'C0')
ax1.tick_params(axis='x', labelsize=0)
ax1.tick_params(axis='y', labelsize=siz,colors = 'C0')
ax1.yaxis.set_ticks([0,100,200,300,400])
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax1.grid(True,color='gainsboro')
ax2 = ax1.twinx()
ax2.plot(time_41025[ok_time],wspd_41025[ok_time,0,0],\
         label = 'Wind Speed (m/s)', color = 'seagreen')
ax2.grid(False)
ax2.axis([tmin,tmax,vel_min,vel_max])
#ax2.set_ylabel('Wind Speed', size=siz,color = 'seagreen')
ax2.tick_params(axis='x', labelsize=0)
ax2.tick_params(axis='y', labelsize=0 ,colors = 'seagreen')
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))

ok_time = np.logical_and(time_41037[:] > tmin, time_41037[:] < tmax)
ax1 = plt.subplot(222)
ax1.plot(time_41037[ok_time],wdir_41037[ok_time,0,0],\
         label = 'Wind Direction',color = 'C0')
ax1.axis([tmin,tmax,dir_min,dir_max])
ax1.set_title('NDBC 41037', size=siz)
#ax1.set_ylabel('Wind Direction ($^o$)', size=siz,color = 'C0')
ax1.tick_params(axis='x', labelsize=0)
ax1.tick_params(axis='y', labelsize=0 ,colors = 'C0')
ax1.yaxis.set_ticks([0,100,200,300,400])
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax1.grid(True,color='gainsboro')
ax2 = ax1.twinx()
ax2.plot(time_41037[ok_time],wspd_41037[ok_time,0,0],\
         label = 'Wind Speed (m/s)', color = 'seagreen')
ax2.grid(False)
ax2.axis([tmin,tmax,vel_min,vel_max])
ax2.set_ylabel('Wind Speed', size=siz,color = 'seagreen')
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=siz,colors = 'seagreen')
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))

ok_time = np.logical_and(time_41002[:] > tmin, time_41002[:] < tmax)
ax1 = plt.subplot(223)
ax1.plot(time_41002[ok_time],wdir_41002[ok_time,0,0],\
         label = 'Wind Direction',color = 'C0')
ax1.axis([tmin,tmax,dir_min,dir_max])
ax1.set_title('NDBC 41002', size=siz)
ax1.set_ylabel('Wind Direction ($^o$)', size=siz,color = 'C0')
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=siz,colors = 'C0')
ax1.yaxis.set_ticks([0,100,200,300,400])
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax1.set_xlabel('Sep. 2018', size=siz)
ax1.grid(True,color='gainsboro')
ax2 = ax1.twinx()
ax2.plot(time_41002[ok_time],wspd_41002[ok_time,0,0],\
         label = 'Wind Speed ', color = 'seagreen')
ax2.grid(False)
ax2.axis([tmin,tmax,vel_min,vel_max])
#ax2.set_ylabel('Wind Speed', size=siz,color = 'seagreen')
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=0 ,colors = 'seagreen')
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))

ok_time = np.logical_and(time_41004[:] > tmin, time_41004[:] < tmax)
ax1 = plt.subplot(224)
ax1.plot(time_41004[ok_time],wdir_41004[ok_time,0,0],\
         label = 'Wind Direction',color = 'C0')
ax1.axis([tmin,tmax,dir_min,dir_max])
ax1.set_title('NDBC 41004', size=siz)
#ax1.set_ylabel('Wind Direction ($^o$)', size=siz,color = 'C0')
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=0 ,colors = 'C0')
ax1.yaxis.set_ticks([0,100,200,300,400])
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax1.set_xlabel('Sep. 2018', size=siz)
ax1.grid(True,color='gainsboro')
ax2 = ax1.twinx()
ax2.plot(time_41004[ok_time],wspd_41004[ok_time,0,0],\
         label = 'Wind Speed (m/s)', color = 'seagreen')
ax2.grid(False)
ax2.axis([tmin,tmax,vel_min,vel_max])
ax2.set_ylabel('Wind Speed', size=siz,color = 'seagreen')
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=siz,colors = 'seagreen')
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Florence/wind_dir_speed.png")
plt.show()

#%%

siz=12
tmin = datetime.datetime.strptime('2018/09/10','%Y/%m/%d')
tmax = datetime.datetime.strptime('2018/09/17','%Y/%m/%d')

#temp_min = np.min(wtmp_41025[ok_time,0,0])
#temp_max = np.max(wtmp_41025[ok_time,0,0])
temp_min = 20
temp_max = 30

tFlor = datetime.datetime.strptime('2018/09/13/18/00','%Y/%m/%d/%H/%M')
tFlor = np.tile(tFlor,len(np.arange(temp_min,temp_max)))

fig, plt.subplots(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='w')

ok_time = np.logical_and(time_41025[:] > tmin, time_41025[:] < tmax)
ax1 = plt.subplot(221)
ax1.plot(time_41025[ok_time],wtmp_41025[ok_time,0,0],label = 'Water Temp In Situ')
ax1.plot(time31[oktime31[0]],target_temp31_41025,label = 'Water temp GOFS3.1')
ax1.plot(tFlor,np.arange(temp_min,temp_max))
ax1.axis([tmin,tmax,temp_min,temp_max])
ax1.set_title('NDBC 41025', size=siz)
ax1.set_ylabel('Temperature ($^o$C)', size=siz)
ax1.tick_params(axis='x', labelsize=0)
ax1.tick_params(axis='y', labelsize=siz)
ax1.set_yticks(np.arange(temp_min, temp_max, 2))
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax1.legend(loc='lower left',fontsize = siz)
ax1.grid(True,color='gainsboro')

ok_time = np.logical_and(time_41037[:] > tmin, time_41037[:] < tmax)
ax2 = plt.subplot(222)
ax2.plot(time_41037[ok_time],wtmp_41037[ok_time,0,0],label = 'Water Temp')
ax2.plot(time31[oktime31[0]],target_temp31_41037,label = 'Water temp GOFS3.1')
ax2.plot(tFlor,np.arange(temp_min,temp_max))
ax2.axis([tmin,tmax,temp_min,temp_max])
ax2.set_title('NDBC 41037', size=siz)
ax2.tick_params(axis='x', labelsize=0)
ax2.tick_params(axis='y', labelsize=0)
ax2.set_yticks(np.arange(temp_min, temp_max, 2))
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax2.legend(loc='lower left',fontsize = siz)
ax2.grid(True,color='gainsboro')

ok_time = np.logical_and(time_41002[:] > tmin, time_41002[:] < tmax)
ax3 = plt.subplot(223)
ax3.plot(time_41002[ok_time],wtmp_41002[ok_time,0,0],label = 'Water Temp')
ax3.plot(time31[oktime31[0]],target_temp31_41002,label = 'Water temp GOFS3.1')
ax3.plot(tFlor,np.arange(temp_min,temp_max))
ax3.axis([tmin,tmax,temp_min,temp_max])
ax3.set_title('NDBC 41002', size=siz)
ax3.set_xlabel('Sep. 2018', size=siz)
ax3.set_ylabel('Temperature ($^o$C)', size=siz)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=siz)
ax3.set_yticks(np.arange(temp_min, temp_max, 2))
ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax3.legend(loc='lower left',fontsize = siz)
ax3.grid(True,color='gainsboro')

ok_time = np.logical_and(time_41004[:] > tmin, time_41004[:] < tmax)
ax4 = plt.subplot(224)
ax4.plot(time_41004[ok_time],wtmp_41004[ok_time,0,0],label = 'Water Temp')
ax4.plot(time31[oktime31[0]],target_temp31_41004,label = 'Water temp GOFS3.1')
ax4.plot(tFlor,np.arange(temp_min,temp_max))
ax4.axis([tmin,tmax,temp_min,temp_max])
ax4.set_title('NDBC 41004', size=siz)
ax4.set_xlabel('Sep. 2018', size=siz)
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=0)
ax4.set_yticks(np.arange(temp_min, temp_max, 2))
ax4.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d'))
ax4.legend(loc='lower left',fontsize = siz)
ax4.grid(True,color='gainsboro')

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Florence/water_temp_vs_GOFS31.png")
plt.show()
