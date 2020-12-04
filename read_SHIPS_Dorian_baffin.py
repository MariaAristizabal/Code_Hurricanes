#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:40:58 2020

@author: aristizabal
"""

#%%

year = '2019'
basin = 'NATL'
name_storm = 'DORIAN05L'
cycle = '2019082800'

folder_SHIPS_Dorian = '/home/aristizabal/SHIPS_shear_Dorian_2019/'

# figures
folder_fig = '/www/web/rucool/aristizabal/Figures/'

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import os
import glob

#%%

SHIPS_Dorian1 = sorted(glob.glob(os.path.join(folder_SHIPS_Dorian,'*POM*.txt')))
SHIPS_Dorian2 = sorted(glob.glob(os.path.join(folder_SHIPS_Dorian,'*HYCOM*.txt')))
SHIPS_Dorian = np.hstack([SHIPS_Dorian1,SHIPS_Dorian2])

markers = ['X-','^-','H-']
colors = ['mediumorchid','teal','darkorange']
labels = ['HWRF2019-POM (IC clim.) ','HWRF2020-POM (IC RTOFS) ','HWRF2020-HYCOM (IC RTOFS) ']

#%%
fig,ax1 = plt.subplots(figsize=(10, 5))
plt.title('Shear Magnitude Forecast Dorian Cycle ' + cycle, fontsize=16)
plt.ylabel('Shear Magnitude (KT)',fontsize=14)
plt.xlabel('Forecast Lead Time (Hours)',fontsize=14)
ax1.xaxis.set_major_locator(MultipleLocator(12))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(3))

for f,file in enumerate(SHIPS_Dorian):
    print(file)
    r = open(file)
    data = r.read()
    
    for s in data.split('\n'):
        if s[0:4] == 'TIME':
            lead_h = s.split()[2:]
        if s[0:7] == 'SHR_MAG':
            SHR_M = s.split()[2:]
        
    lead_hours = np.asarray([int(hh) for hh in lead_h])
    SHR_MAG_HWRF = np.asarray([float(ss) for ss in SHR_M])
         
    plt.plot(lead_hours,SHR_MAG_HWRF,markers[f],color=colors[f],label=labels[f],markeredgecolor='k',markersize=7)
    plt.xlim(lead_hours[0],lead_hours[-1])
    plt.ylim([0,30])
    plt.legend()

file_name = folder_fig + 'Dorian_shear_magnitude_cycle_' + cycle
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)

#%%
fig,ax1 = plt.subplots(figsize=(10, 5))
plt.title('Translation Speed Forecast Dorian Cycle ' + cycle, fontsize=16)
plt.ylabel('Translation Speed (m/s)',fontsize=14)
plt.xlabel('Forecast Lead Time (Hours)',fontsize=14)
ax1.xaxis.set_major_locator(MultipleLocator(12))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(3))

for f,file in enumerate(SHIPS_Dorian):
    print(file)
    r = open(file)
    data = r.read()
    
    for s in data.split('\n'):
        if s[0:4] == 'TIME':
            lead_h = s.split()[2:]
        if s[0:7] == 'STM_SPD':
            STM_S = s.split()[2:] 
        
    lead_hours = np.asarray([int(hh) for hh in lead_h])
    STM_SPD_HWRF = np.asarray([float(ss) for ss in STM_S])* 0.5144
         
    plt.plot(lead_hours,STM_SPD_HWRF,markers[f],color=colors[f],label=labels[f],markeredgecolor='k',markersize=7)
    plt.xlim(lead_hours[0],lead_hours[-1])
    plt.ylim([0,10])
    plt.legend()

file_name = folder_fig + 'Dorian_trans_speed_cycle_' + cycle
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)
    
