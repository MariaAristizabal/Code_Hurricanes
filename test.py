#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:49:15 2020

@author: root
"""
#%% User input

# Servers location
url_erddap = 'https://data.ioos.us/gliders/erddap'
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
#url_glider = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
url_thredds = 'http://gliders.ioos.us/thredds/dodsC/deployments/secoora/bass-20180808T0000/bass-20180808T0000.nc3.nc'

#date_ini = '2018/09/01/00' # year/month/day/hour
#date_end = '2018/11/30/00' # year/month/day/hour
date_ini = '2018/08/10/00'
date_end = '2018/08/13/00'
scatter_plot = 'no'

# SAB
lon_lim = [-85.0,-75.0]
lat_lim = [25.0,35.0]

# glider variable to retrieve
var_name_glider = 'temperature'
#var_glider = 'salinity'
delta_z = 0.4 # bin size in the vertical when gridding the variable vertical profile 
              # default value is 0.3  

# model variable name
model_name = 'GOFS 3.1'
var_name_model = 'water_temp'
#var_model = 'salinity'

#%%

import sys
sys.path
#sys.path.append('/home/aristizabal/glider_model_comparisons_Python/')
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import retrieve_glider_id_erddap_server
from glider_transect_model_com import glider_transect_erddap_server_vs_model

#%%
url_server = url_erddap

gliders = retrieve_glider_id_erddap_server(url_server,lat_lim,lon_lim,date_ini,date_end)
print('The gliders found are ')
print(gliders)   

#%%                    
    
dataset_id = gliders[0]
print(dataset_id)
kwargs = dict(date_ini=date_ini,date_end=date_end)

timeg,depthg_gridded,varg_gridded,timem,depthm,target_varm = \
glider_transect_erddap_server_vs_model(url_erddap,dataset_id,url_model,\
                          lat_lim,lon_lim,var_name_glider,var_name_model,model_name,\
                          delta_z=0.4,**kwargs)   
    