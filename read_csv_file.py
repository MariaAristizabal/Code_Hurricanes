#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:20:48 2019

@author: aristizabal
"""

#%% data from https://www.ncdc.noaa.gov/cdo-web/datasets/LCD/stations/WBAN:11640/detail

csv_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/txt_csv_files/1899543.csv'

#%%

import csv
import numpy as np
from datetime import datetime

#%%
hourly_precip = []
time_hourly_precip = []

with open(csv_file) as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    for j,row in enumerate(readcsv):
        if j == 0:
            ind = [i for i,cont in enumerate(row) if cont=='HourlyPrecipitation'][0]
        else:
            if np.logical_or(row[ind] == 'T',row[ind] == ''):
                hourly_precip.append(np.nan)
            else:
                hourly_precip.append(float(row[ind]))
                
            time_hourly_precip.append(datetime.strptime(row[1],'%Y-%m-%dT%H:%M:%S'))