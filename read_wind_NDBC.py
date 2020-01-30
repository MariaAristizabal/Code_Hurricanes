#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:08:52 2020

@author: root
"""

#%% data from url_NDBC_41043 = 'https://www.ndbc.noaa.gov/data/realtime2/41043.txt'

csv_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/txt_csv_files/41043.csv'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

url_NDBC = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/41043/41043h2019.nc'

#%%


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import netCDF4

#%%

wind_NDBC = xr.open_dataset(url_NDBC,decode_times=False)

tt = wind_NDBC['time']
time_NDBC = netCDF4.num2date(tt[:],tt.units)

wspd_NDBC = np.asarray(wind_NDBC['wind_spd'][:])[:,0,0]
wdir_NDBC = np.asarray(wind_NDBC['wind_dir'][:])[:,0,0] 

#%%
'''

time_NDBC = []
wspd_NDBC = []
wdir_NDBC = []

with open(csv_file) as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    for j,row in enumerate(readcsv):
        if j == 0:
            ind_dir = [i for i,cont in enumerate(row[0].split()) if cont=='WDIR'][0]
            ind_wsp = [i for i,cont in enumerate(row[0].split()) if cont=='WSPD'][0]
        if j > 1:
            if np.logical_or(row[0].split()[ind_dir] == 'MM',row[0].split()[ind_dir]  == ''):
                wdir_NDBC.append(np.nan)
            else:
                wdir_NDBC.append(float(row[0].split()[ind_dir]))
            if np.logical_or(row[0].split()[ind_wsp] == 'MM',row[0].split()[ind_wsp]  == ''):
                wspd_NDBC.append(np.nan)
            else:
                wspd_NDBC.append(float(row[0].split()[ind_wsp]))
                
            year = int(row[0].split()[0])
            month = int(row[0].split()[1])
            day = int(row[0].split()[2])
            hour = int(row[0].split()[3])
            minu = int(row[0].split()[4])                
            time_NDBC.append(datetime(year,month,day,hour,minu))
 '''           
#%%
         
fig,ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(time_NDBC,wspd_NDBC,'.-',color='steelblue')
ax1.set_ylabel('Wind Speed (m/s)',color = 'steelblue',fontsize=14)
ax1.tick_params('y', colors='steelblue')

ax2 = ax1.twinx()
ax2.plot(time_NDBC,wdir_NDBC,'.-',color='seagreen',alpha=0.5)
ax2.set_ylabel('Wind Direction (degT)',color = 'seagreen',fontsize=14)
ax2.tick_params('y', colors='seagreen')
plt.xlim([datetime(2019,8,25),datetime(2019,9,8)])
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)

file = folder + 'Wind_41043_NDBC_station'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%
         
fig,ax1 = plt.subplots()
ax1.plot(time_NDBC,wspd_NDBC,'.-',color='steelblue')
ax1.set_ylabel('Wind Speed (m/s)',color = 'steelblue',fontsize=14)
ax1.tick_params('y', colors='steelblue')

ax2 = ax1.twinx()
ax2.plot(time_NDBC,wdir_NDBC,'.-',color='seagreen',alpha=0.5)
ax2.set_ylabel('Wind Direction (degT)',color = 'seagreen',fontsize=14)
ax2.tick_params('y', colors='seagreen')
plt.xlim([datetime(2019,8,28),datetime(2019,8,29,18)])
xfmt = mdates.DateFormatter('%d \n %H')
ax2.xaxis.set_major_formatter(xfmt)
plt.grid(True)

file = folder + 'Wind_41043_NDBC_station2'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Calculate rate of rotation of the wind

theta_wind = np.deg2rad(wdir_NDBC)
dtheta_wind = np.gradient(theta_wind)
dt = np.gradient(mdates.date2num(time_NDBC))*(60*24*60)
f = 0.524 * 10**(-4)#2*Omega*np.sin(lat) # Omega is rotation rate of earth
dthetadt_wind_over_f = dtheta_wind/(dt*f)

#%%
fig,ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(time_NDBC,dthetadt_wind_over_f,'.-')
plt.xlim([datetime(2019,8,25),datetime(2019,9,8)])
#plt.ylim([-0.003,0.003])
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.ylabel('$dtheta/dt * 1/f$')

file = folder + 'dthetadt_wind_over_f_41043_NDBC_station'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%
fig,ax1 = plt.subplots()
ax1.plot(time_NDBC,dthetadt_wind_over_f,'.-')
plt.xlim([datetime(2019,8,28),datetime(2019,8,29,18)])
plt.ylim([-20,20])
xfmt = mdates.DateFormatter('%d \n %H')
ax1.xaxis.set_major_formatter(xfmt)
plt.ylabel('$dtheta/dt * 1/f$')

file = folder + 'dthetadt_wind_over_f_41043_NDBC_station2'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 