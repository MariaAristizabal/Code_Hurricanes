#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:39:56 2020

@author: aristizabal
"""

file_track = '/home/aristizabal/HWRF_POM_13l_2020/HWRF_POM_13l_2020082318/laura13l.2020082318.trak.hwrf.atcfunix'

temp_lim = [25,31.6]
salt_lim = [31,37.1]
temp200_lim = [5,24.6]
salt200_lim = [35.5,37.6]
tempb_lim = [0,25.6]
tempt_lim = [6,31.1]

folder_fig = '/home/aristizabal/Figures/'

#%% 
import numpy as np
import sys

sys.path.append('/home/aristizabal/Code/surf_fields_and_Argo_compar_hurric')

from Surf_fields_models_baffin import GOFS31_baffin, RTOFS_oper_baffin

#%% Read track files 
ff_oper = open(file_track,'r')
f_oper = ff_oper.readlines()

latt = []
lonn = []
intt = []
lead_time = []
for l in f_oper:
    lat = float(l.split(',')[6][0:4])/10
    if l.split(',')[6][4] == 'N':
        lat = lat
    else:
        lat = -lat
    lon = float(l.split(',')[7][0:5])/10
    if l.split(',')[7][4] == 'E':
        lon = lon
    else:
        lon = -lon
    latt.append(lat)
    lonn.append(lon)
    intt.append(float(l.split(',')[8]))
    lead_time.append(int(l.split(',')[5][1:4]))

latt = np.asarray(latt)
lonn = np.asarray(lonn)
intt = np.asarray(intt)
lead_time_track, ind = np.unique(lead_time,return_index=True)
lat_forec_track = latt[ind]
lon_forec_track = lonn[ind]
int_forec_track = intt[ind]

lon_forec_cone = []
lat_forec_cone = []
lon_best_track = []
lat_best_track = []

lon_lim = [np.min(lon_forec_track)-5,np.max(lon_forec_track)+5]
lat_lim = [np.min(lat_forec_track)-5,np.max(lat_forec_track)+5]

RTOFS_oper_baffin(lon_forec_track,lat_forec_track,lon_forec_cone,lat_forec_cone,\
                  lon_best_track,lat_best_track,lon_lim,lat_lim,temp_lim,salt_lim,
                  temp200_lim,salt200_lim,tempb_lim,tempt_lim,folder_fig)

       
GOFS31_baffin(lon_forec_track,lat_forec_track,lon_forec_cone,lat_forec_cone,\
              lon_best_track,lat_best_track,lon_lim,lat_lim,temp_lim,salt_lim,\
                  temp200_lim,salt200_lim,tempb_lim,tempt_lim,folder_fig)