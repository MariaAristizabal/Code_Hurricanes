#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:42:43 2019

@author: aristizabal
"""
#%%
url_NDBC_41043 = 'https://www.ndbc.noaa.gov/data/realtime2/41043.txt'
url_NDBC_LTBV3 = 'https://www.ndbc.noaa.gov/data/realtime2/LTBV3.txt'

#%%

import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#%%

r = requests.get(url_NDBC_41043)
data = r.text

time_NDBC = []
wspd_NDBC = []
for s in data.split('\n')[2:]:
    if len(s) != 0:
        year = int(s.split(' ')[0])
        month = int(s.split(' ')[1])
        day = int(s.split(' ')[2])
        hour = int(s.split(' ')[3])
        mi = int(s.split(' ')[4])
        time_NDBC.append(datetime(year,month,day,hour,mi))
        if len(s.split(' ')[7]) != 0:
            wspd_NDBC.append(float(s.split(' ')[7]))
        else:
            wspd_NDBC.append(np.nan)
        #print(float(s.split(' ')[7]))

time_NDBC = np.array(time_NDBC)         
wspd_NDBC = np.asarray(wspd_NDBC)

         
#%%
         
plt.figure()
plt.plot(time_NDBC,wspd_NDBC,'.-')

#%%

r = requests.get(url_NDBC_LTBV3 )
data = r.text

time_NDBC = []
wspd_NDBC = []
for s in data.split('\n')[2:]:
    if len(s) != 0:
        year = int(s.split(' ')[0])
        month = int(s.split(' ')[1])
        day = int(s.split(' ')[2])
        hour = int(s.split(' ')[3])
        mi = int(s.split(' ')[4])
        time_NDBC.append(datetime(year,month,day,hour,mi))
        if len(s.split(' ')[7]) != 0:
            wspd_NDBC.append(float(s.split(' ')[7]))
        else:
            wspd_NDBC.append(np.nan)
        #print(float(s.split(' ')[7]))

time_NDBC = np.array(time_NDBC)         
wspd_NDBC = np.asarray(wspd_NDBC)
