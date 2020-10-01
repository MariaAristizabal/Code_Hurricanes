#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:00:39 2020

@author: aristizabal
"""

#%%

year = '2019'
basin = 'NATL'
name_storm = 'DORIAN05L'
cycles = ['2019082800','2019082806','2019082812','2019082818','2019082900',\
          '2019082906','2019082912','2019082918','2019083000','2019083006',\
          '2019083012','2019083018','2019083100','2019083106','2019083112',\
          '2019083118','2019090100','2019090106']

# figures
folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures'

#%%

import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
#from datetime import datetime

#%%
fig,ax1 = plt.subplots(figsize=(10, 5))
plt.title('Shear Magnitude Forecast Dorian Cycles ' + cycles[0], fontsize=16)
plt.ylabel('Shear Magnitude (KT)',fontsize=14)
plt.xlabel('Forecast Lead Time (Hours)',fontsize=14)
ax1.xaxis.set_major_locator(MultipleLocator(12))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(3))

for cycle in cycles[0:1]:
    print(cycle)
    url_HWRF2019 = 'https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HWRFForecast/RT' + \
                        year + '_' + basin + '/' + name_storm + '/' + name_storm + '.' + \
                        cycle + '/' + name_storm + '.' + cycle + '.txt'
    r = requests.get(url_HWRF2019)
    data = r.text
    
    for s in data.split('\n'):
        if s[0:4] == 'TIME':
            lead_h = s.split()[2:]
        if s[0:7] == 'SHR_MAG':
            SHR_M = s.split()[2:]
        
    lead_hours = np.asarray([int(hh) for hh in lead_h])
    SHR_MAG_HWRF2019 = np.asarray([float(ss) for ss in SHR_M])
         
    plt.plot(lead_hours,SHR_MAG_HWRF2019,'X-',color='mediumorchid',label='HWRF2019-POM (IC clim.)',markeredgecolor='k',markersize=7)
    plt.xlim(lead_hours[0],lead_hours[-1])
    plt.ylim([0,30])
    plt.legend()

file_name = folder_fig + 'Dorian_shear_magnitude_cycle_' + cycle
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)

#%% Shear all cycles

SHR_MAG_model1 = np.empty((len(np.arange(0,127,6)),len(cycles)))
SHR_MAG_model1[:] = np.nan
for c,cycle in enumerate(cycles):
    print(cycle)
    url_HWRFForecast = 'https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HWRFForecast/RT' + \
                        year + '_' + basin + '/' + name_storm + '/' + name_storm + '.' + \
                        cycle + '/' + name_storm + '.' + cycle + '.txt'
    r = requests.get(url_HWRFForecast)
    data = r.text
    
    for s in data.split('\n'):
        if s[0:4] == 'TIME':
            lead_h = s.split()[2:]
        if s[0:7] == 'SHR_MAG':
            SHR_M = s.split()[2:]
        
    lead_hours = np.asarray([int(hh) for hh in lead_h])
    SHR_MAG_model1[:,c] = np.asarray([float(ss) for ss in SHR_M])

SHR_MAG_model1_mean = np.nanmean(SHR_MAG_model1,1)
SHR_MAG_model1_min = np.nanmin(SHR_MAG_model1,1)
SHR_MAG_model1_max = np.nanmax(SHR_MAG_model1,1)

#%%
fig,ax = plt.subplots(figsize=(10, 5))
plt.title('Shear Magnitude Forecast Dorian Cycles ' + cycles[0] + '-' + cycles[-1], fontsize=16)
plt.ylabel('Shear Magnitude (KT)',fontsize=14)
plt.xlabel('Forecast Lead Time (Hours)',fontsize=14)
ax.xaxis.set_major_locator(MultipleLocator(12))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(MultipleLocator(3))  

plt.plot(lead_hours,SHR_MAG_model1_mean,'X-',color='mediumorchid',label='HWRF2019 (IC clim.)',markeredgecolor='k',markersize=7)
ax.fill_between(lead_hours,SHR_MAG_model1_min,SHR_MAG_model1_max,color='mediumorchid',alpha=0.1)
plt.plot(lead_hours,SHR_MAG_model1_min,'-',color='mediumorchid',alpha=0.5)
plt.plot(lead_hours,SHR_MAG_model1_max,'-',color='mediumorchid',alpha=0.5)
plt.xlim(lead_hours[0],lead_hours[-1])
plt.ylim([0,30])
plt.legend()

file_name = folder_fig + 'Dorian_shear_magnitude_cycles_' + cycle[0] + '_' + cycles[-1]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)    
    
