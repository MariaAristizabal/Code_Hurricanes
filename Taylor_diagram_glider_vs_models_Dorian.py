#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:18:38 2019

@author: root
"""

#%% User input

#lon_lim = [-100.0,-55.0]
#lat_lim = [10.0,45.0]

lon_lim = [-80.0,-60.0]
lat_lim = [15.0,35.0]

# Server erddap url IOOS glider dap
server = 'https://data.ioos.us/gliders/erddap'

#gliders sg666, sg665, sg668, silbo
gdata_ng665 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG665-20190718T1155/SG665-20190718T1155.nc3.nc'
gdata_ng666 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG666-20190718T1206/SG666-20190718T1206.nc3.nc'
gdata_ng668 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
gdata_silbo ='http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/silbo-20190717T1917/silbo-20190717T1917.nc3.nc'

#Time window
date_ini = '2019/08/28/00/00'
date_end = '2019/09/02/00/00'

# url for GOFS 3.1
url_GOFS31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator,
                                                 DictFormatter)
import requests

from matplotlib import pyplot as plt
import xarray as xr
import netCDF4
from netCDF4 import Dataset
import cmocean
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
from erddapy import ERDDAP
import pandas as pd
import seawater as sw

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import read_glider_data_thredds_server
#from process_glider_data import grid_glider_data_thredd

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading glider data
    
url_glider = gdata_ng665
#url_glider = gdata_ng666
#url_glider = gdata_ng668
#url_glider = gdata_silbo

#del depthg_gridded, tempg_gridded, saltg_gridded, densg_gridded

var = 'temperature'
#Time window
#date_ini = '2019/08/28/00'
#date_end = '2019/09/02/00'
scatter_plot = 'no'
kwargs = dict(date_ini=date_ini[0:-3],date_end=date_end[0:-3])

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
tempg = varg  

var = 'salinity'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)         
             
 
saltg = varg
 
var = 'density'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
                         
             
densg = varg
depthg = depthg             
  
#contour_plot='yes'    
#depthg_gridded, varg_gridded, timegg = \
#                    grid_glider_data_thredd(timeg,latg,long,depthg,varg,var,inst_id) 
                    
#%% Grid glider variables according to depth
             
depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
tempg_gridded[:] = np.nan
saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
saltg_gridded[:] = np.nan
densg_gridded = np.empty((len(depthg_gridded),len(timeg)))
densg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    print(tt)
    depthu,oku = np.unique(depthg[:,t],return_index=True)
    tempu = tempg[oku,t]
    saltu = saltg[oku,t]
    densu = densg[oku,t]
    okdd = np.isfinite(depthu)
    depthf = depthu[okdd]
    tempf = tempu[okdd]
    saltf = saltu[okdd]
    densf = densu[okdd]
 
    okt = np.isfinite(tempf)
    if np.sum(okt) < 3:
        tempg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                            depthg_gridded < np.max(depthf[okt]))
        tempg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okt],tempf[okt])
        
    oks = np.isfinite(saltf)
    if np.sum(oks) < 3:
        saltg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                            depthg_gridded < np.max(depthf[okt]))
        saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])
    
    okdd = np.isfinite(densf)
    if np.sum(okdd) < 3:
        densg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okdd]),\
                            depthg_gridded < np.max(depthf[okdd]))
        densg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okdd],densf[okdd])

#%% GOGF 3.1

df = xr.open_dataset(url_GOFS31,decode_times=False)

#%%
## Decode the GOFS3.1 time into standardized mdates datenums 
hours_since2000 = df.time
time_naut       = datetime(2000,1,1)
time31 = np.ones_like(hours_since2000)
for ind, hrs in enumerate(hours_since2000):
    time31[ind] = mdates.date2num(time_naut+timedelta(hours=int(hrs)))

## Find the dates of import
dini = mdates.date2num(datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')) 
dend = mdates.date2num(datetime.strptime(date_end,'%Y/%m/%d/%H/%M'))
formed  = int(np.where(time31 >= dini)[0][0])
dissip  = int(np.where(time31 >= dend)[0][0])
oktime31 = np.arange(formed,dissip+1,dtype=int)

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
botm  = int(np.where(df.lat > lat_limG[0])[0][0])
top   = int(np.where(df.lat > lat_limG[1])[0][0])
left  = np.where(df.lon > lon_limG[0])[0][0]
right = np.where(df.lon > lon_limG[1])[0][0]
#oklat31 = np.where(np.logical_and(df.lat >= lat_limG[0], df.lat <= lat_lim[-1]))[0]
#oklon31 = np.where(np.logical_and(df.lon >= lon_limG[0], df.lon <= lon_lim[-1]))[0]
lat31= np.asarray(df.lat[botm:top])
lon31= np.asarray(df.lon[left:right])
depth31 = np.asarray(df.depth[:])

# Conversion from GOFS convention to glider longitude and latitude
lon31g= np.empty((len(lon31),))
lon31g[:] = np.nan
for i in range(len(lon31)):
    if lon31[i] > 180: 
        lon31g[i] = lon31[i] - 360 
    else:
        lon31g[i] = lon31[i]
lat31g = lat31

#%% Read GOFS 3.1 output
    
print('Retrieving coordinates from model')
model = xr.open_dataset(url_GOFS31,decode_times=False)
    
lat31 = np.asarray(model.lat[:])
lon31 = np.asarray(model.lon[:])
depth31 = np.asarray(model.depth[:])
tt31 = model.time
t31 = netCDF4.num2date(tt31[:],tt31.units) 

tmin = datetime.strptime(date_ini[0:-3],'%Y/%m/%d/%H')
tmax = datetime.strptime(date_end[0:-3],'%Y/%m/%d/%H')

oktime31 = np.where(np.logical_and(t31 >= tmin, t31 <= tmax))
time31 = np.asarray(t31[oktime31])
    
#%%

# Conversion from glider longitude and latitude to GOFS convention
target_lon = np.empty((len(long),))
target_lon[:] = np.nan
for i,ii in enumerate(long):
    if ii < 0: 
        target_lon[i] = 360 + ii
    else:
        target_lon[i] = ii
target_lat = latg

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_model = [mdates.date2num(time31[i]) for i in np.arange(len(time31))]

# interpolating glider lon and lat to lat and lon on model time
sublon31 = np.interp(tstamp_model,tstamp_glider,target_lon)
sublat31 = np.interp(tstamp_model,tstamp_glider,target_lat)

# Conversion from GOFS convention to glider longitude and latitude
sublon31g= np.empty((len(sublon31),))
sublon31g[:] = np.nan
for i in range(len(sublon31)):
    if sublon31[i] > 180: 
        sublon31g[i] = sublon31[i] - 360 
    else:
        sublon31g[i] = sublon31[i]
sublat31g = sublat31

# getting the model grid positions for sublonm and sublatm
oklon31=np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
oklat31=np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)
    
# Getting glider transect from model
print('Getting glider transect from model. If it breaks is because GOFS 3.1 server is not responding')
target_temp31 = np.empty((len(depth31),len(oktime31[0])))
target_temp31[:] = np.nan
target_salt31 = np.empty((len(depth31),len(oktime31[0])))
target_salt31[:] = np.nan
for i in range(len(oktime31[0])):
    print(len(oktime31[0]),' ',i)
    target_temp31[:,i] = model.variables['water_temp'][oktime31[0][i],:,oklat31[i],oklon31[i]]
    target_salt31[:,i] = model.variables['salinity'][oktime31[0][i],:,oklat31[i],oklon31[i]]

#%% Interpolate glider transect onto GOFS time and depth
    
oktimeg_gofs = np.round(np.interp(tstamp_model,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)    
    
tempg_to31 = np.empty((target_temp31.shape[0],target_temp31.shape[1]))
tempg_to31[:] = np.nan
for i in np.arange(len(oktimeg_gofs)):
    pos = np.argsort(depthg[:,oktimeg_gofs[i]])
    tempg_to31[:,i] = np.interp(depth31,depthg[pos,oktimeg_gofs[i]],tempg[pos,oktimeg_gofs[i]])   
    
#%% POM Experimental
    
timestamp_pom = np.array([737299.  , 737299.25, 737299.5 , 737299.75, 737300.  , 737300.25,
       737300.5 , 737300.75, 737301.  , 737301.25, 737301.5 , 737301.75,
       737302.  , 737302.25, 737302.5 , 737302.75, 737303.  , 737303.25,
       737303.5 , 737303.75, 737304.  ])
    
target_temp_pom = np.array([[29.3775425 , 29.25422478, 29.1914196 , 29.19390869, 29.12872124,
        29.05097771, 28.92534447, 28.9560833 , 28.89664459, 28.911129  ,
        28.96275139, 29.13430977, 29.03517914, 28.87983704, 28.83248138,
        29.07321167, 28.93194008, 28.72905922, 28.54187393, 28.98860168,
        29.01041412],
       [29.3435936 , 29.25735283, 29.19444084, 29.19709778, 29.13141251,
        29.05488205, 28.93005753, 28.9206543 , 28.90005112, 28.9139328 ,
        28.96352768, 29.03181458, 29.01563263, 28.88182831, 28.83518219,
        28.89757729, 28.93524361, 28.73071861, 28.53689384, 28.47193336,
        28.5585556 ],
       [29.30026054, 29.25836945, 29.19587898, 29.19932365, 29.13381004,
        29.05728531, 28.93194008, 28.89338493, 28.90116119, 28.91477013,
        28.96327591, 28.98039818, 28.92357635, 28.87158585, 28.83753395,
        28.89945793, 28.90022469, 28.74077225, 28.55186844, 28.47226715,
        28.53502655],
       [29.24204063, 29.2387619 , 29.20902824, 29.20444679, 29.13715172,
        29.05796814, 28.93229103, 28.87542915, 28.89620781, 28.92361641,
        28.9627018 , 28.98000145, 28.92840385, 28.88468552, 28.84941864,
        28.90870094, 28.89911079, 28.74593163, 28.57684898, 28.50916862,
        28.57093239],
        [29.14921188, 29.1290741 , 29.05426025, 29.08709717, 29.13732147,
        29.04262543, 28.92093086, 28.85131645, 28.85923004, 28.92778969,
        28.95201683, 28.96247292, 28.86304665, 28.815485  , 28.85790634,
        28.88248253, 28.85087204, 28.71117401, 28.56221771, 28.51286697,
        28.55457115],
       [28.95604324, 28.89825058, 28.72317123, 28.76905441, 28.97997856,
        29.0166378 , 28.88539886, 28.78442955, 28.80012321, 28.87294769,
        28.87373734, 28.88072014, 28.70861244, 28.64842987, 28.76346779,
        28.81417656, 28.75401497, 28.65254021, 28.54261589, 28.50478172,
        28.54371452],
       [28.5512104 , 28.45577621, 28.132761  , 28.17053032, 28.53658676,
        28.65947914, 28.55939102, 28.38376427, 28.43063164, 28.68678856,
        28.6427002 , 28.782938  , 28.45482063, 28.40474319, 28.85623741,
        28.75809669, 28.57553291, 28.64868355, 28.49873543, 28.51408005,
        28.56644821],
       [27.86513519, 27.77974129, 27.37233162, 27.37616539, 27.81879044,
        27.96372032, 27.74902534, 27.45029831, 27.53647232, 27.88495064,
        27.8507328 , 28.17667198, 27.77188301, 27.67276382, 28.5198307 ,
        28.27397919, 27.96377563, 28.28446579, 28.08138657, 28.14323425,
        28.24124527],
       [27.07060432, 27.02925491, 26.59326744, 26.56984138, 27.04814529,
        27.20523834, 26.89972305, 26.57858849, 26.71965408, 27.09629631,
        27.08955765, 27.52500534, 27.05138969, 26.87840271, 27.90010452,
        27.4907589 , 27.12863922, 27.65821648, 27.40789032, 27.46869087,
        27.59189034],
       [26.3024044 , 26.30626106, 25.93179512, 25.8953495 , 26.30980873,
        26.4462986 , 26.12925148, 25.83297539, 26.01301956, 26.35877037,
        26.34342384, 26.75833893, 26.29172707, 26.10573196, 27.08976364,
        26.65069199, 26.33905983, 26.91044998, 26.66125488, 26.69241142,
        26.80404472],
       [25.71424294, 25.74100876, 25.33169746, 25.28985214, 25.71364021,
        25.85598564, 25.43140793, 25.0880146 , 25.36990547, 25.80108261,
        25.75805664, 26.15081024, 25.56564331, 25.32755661, 26.46210861,
        25.95733833, 25.65044594, 26.30265427, 25.93304443, 25.96995735,
        26.15925217],
        [24.52645683, 24.59450722, 24.15164375, 24.07573128, 24.54190254,
        24.72651672, 24.16237259, 23.76218987, 24.16559792, 24.74545479,
        24.65112495, 25.11598206, 24.32904243, 23.95717049, 25.44431686,
        24.81894684, 24.42262077, 25.28668594, 24.66107368, 24.64061928,
        25.06227303],
       [23.42571259, 23.54863167, 23.14047432, 23.02121162, 23.43990517,
        23.64089966, 23.02903557, 22.63426781, 23.06625366, 23.64751816,
        23.45133209, 23.98769188, 23.13370323, 22.65515518, 24.30245399,
        23.56213951, 23.11342621, 24.19706726, 23.36198044, 23.20563126,
        23.78123856],
       [22.19639206, 22.3708725 , 22.02945518, 21.90457916, 22.25644493,
        22.45459938, 21.82214928, 21.4664917 , 21.91694069, 22.42234802,
        22.09370804, 22.64410973, 21.83059692, 21.32597542, 22.93971252,
        22.12265968, 21.68835258, 22.91341591, 21.94472504, 21.70913124,
        22.37364197],
       [20.86810684, 21.04883385, 20.76785469, 20.6519928 , 20.92656708,
        21.09707642, 20.4756012 , 20.18356323, 20.64126205, 21.02568626,
        20.56289673, 21.09656525, 20.4094677 , 19.91249847, 21.40262604,
        20.5134716 , 20.09741974, 21.40202141, 20.3866024 , 20.08604622,
        20.76994324],
       [19.73682022, 19.86889076, 19.65075111, 19.54855728, 19.73159981,
        19.85120773, 19.29213333, 19.03881264, 19.45226097, 19.69048691,
        19.19363976, 19.65175056, 19.14886856, 18.64984322, 19.9312458 ,
        19.05983353, 18.66184616, 19.944561  , 18.97470665, 18.60122871,
        19.31775856],
       [18.65079308, 18.76067734, 18.57610893, 18.48551941, 18.6351223 ,
        18.75429916, 18.16383743, 17.90423775, 18.3599968 , 18.52144432,
        17.9751339 , 18.41793823, 17.96376228, 17.33999252, 18.53609848,
        17.62283897, 17.16814995, 18.49301338, 17.43294525, 16.91042709,
        17.70897102],
       [17.48537636, 17.55583572, 17.3374424 , 17.23768616, 17.37877846,
        17.52526093, 16.79461479, 16.50867462, 17.09024811, 17.20308113,
        16.50878906, 17.0416069 , 16.56563759, 15.75698662, 17.04049683,
        16.00532913, 15.44428062, 16.91108513, 15.71621227, 15.05357647,
        15.95011425],
        [16.16467285, 16.17445564, 15.8443203 , 15.703269  , 15.87908268,
        16.08985329, 15.10455513, 14.75386333, 15.5248394 , 15.57876587,
        14.65317631, 15.38337994, 14.85595512, 13.77468967, 15.30681705,
        14.06351662, 13.35002136, 15.15377045, 13.71017838, 12.88881016,
        14.05955982],
       [14.4101572 , 14.34763813, 13.86913872, 13.64709091, 13.87837696,
        14.18036175, 12.93823433, 12.5340004 , 13.46561432, 13.39879799,
        12.25154495, 13.21228409, 12.66652489, 11.30000877, 13.05878258,
        11.59866238, 10.71249485, 12.93520927, 11.15969849, 10.17883682,
        11.72655678],
       [12.24827766, 12.10140514, 11.50961494, 11.16824913, 11.40449715,
        11.77487469, 10.38495827,  9.94359589, 10.90037823, 10.64863873,
         9.38253784, 10.40212154,  9.88721275,  8.39266586, 10.0362854 ,
         8.51738644,  7.55051327,  9.73980331,  7.86743593,  6.87745857,
         8.43319416],
       [ 9.97818947,  9.78993416,  9.27035141,  8.85765171,  8.94542313,
         9.21249676,  8.10111523,  7.72884274,  8.34276772,  7.98567772,
         7.02830791,  7.60746479,  7.26596117,  6.22865725,  7.06748343,
         6.03814888,  5.35943365,  6.47285652,  5.34931087,  4.82552814,
         5.5812521 ],
       [ 8.4131794 ,  8.27684498,  7.96643639,  7.64004326,  7.61019897,
         7.73380756,  7.08030224,  6.840662  ,  7.09785938,  6.83500862,
         6.29486465,  6.43128967,  6.21130943,  5.6540308 ,  5.83472538,
         5.32163095,  5.04387093,  5.23874998,  5.0383234 ,  5.03707075,
         5.1478157 ],
        [ 7.51633883,  7.46090174,  7.24643898,  6.96062756,  6.91706419,
         7.03486395,  6.5345602 ,  6.35130119,  6.51541233,  6.278409  ,
         5.82557678,  5.86782646,  5.66462231,  5.31128979,  5.28227472,
         5.09066677,  5.14100456,  5.15876484,  5.43243551,  5.5263052 ,
         5.68195629],
       [ 6.54669571,  6.52803659,  6.33926487,  6.01803637,  5.9457736 ,
         6.09543037,  5.68719482,  5.60891581,  5.79039955,  5.57130146,
         5.23373985,  5.25793648,  5.16651249,  5.21929312,  5.24641228,
         5.32265234,  5.46687603,  5.54980421,  5.57829475,  5.21399307,
         5.27523994],
       [ 5.70089579,  5.65508556,  5.50058794,  5.20464754,  5.10669374,
         5.28625441,  5.07034111,  5.08506298,  5.24105215,  5.1062026 ,
         4.97890759,  5.06430197,  5.19351768,  5.45167637,  5.47798443,
         5.38676023,  5.2700119 ,  5.22041082,  4.91577291,  4.53905296,
         4.43187094],
       [ 5.05803204,  4.95265007,  4.87642479,  4.70668983,  4.5883522 ,
         4.66576099,  4.54343414,  4.57346344,  4.72622156,  4.7697978 ,
         4.84089088,  5.021101  ,  5.25932026,  5.36848736,  5.36574888,
         5.23722696,  5.1442852 ,  5.0792284 ,  4.94094324,  4.91952133,
         4.80124283],
       [ 4.68123579,  4.56377172,  4.53074789,  4.45792246,  4.37115431,
         4.35654926,  4.36282349,  4.44876528,  4.57166433,  4.67401457,
         4.78619671,  4.91562843,  5.09631062,  5.14248896,  5.12900877,
         5.11730814,  5.10961771,  5.04480982,  5.0042181 ,  5.00526237,
         4.89514208],
       [ 4.3755126 ,  4.31512642,  4.33571291,  4.34634113,  4.35092592,
         4.41808224,  4.52673054,  4.63038778,  4.71947336,  4.81043386,
         4.90124273,  4.98557091,  5.09572601,  5.20138454,  5.21691799,
         5.22404909,  5.20096779,  5.20562315,  5.18429041,  5.16308403,
         5.21482563],
       [ 4.09939051,  4.07613707,  4.15453577,  4.21899176,  4.26466465,
         4.35490894,  4.41802931,  4.49265337,  4.60651398,  4.70738363,
         4.80760908,  4.95985413,  5.09576082,  5.14730644,  5.16311646,
         5.02020454,  4.89284611,  4.91878891,  4.83665371,  4.75685024,
         4.95109081],
        [ 3.8757298 ,  3.88293982,  3.95358896,  4.00993824,  4.07946873,
         4.23250961,  4.31099796,  4.36829758,  4.44527006,  4.49384594,
         4.51244354,  4.60491228,  4.66068792,  4.61289501,  4.62681341,
         4.47506762,  4.44917965,  4.51042938,  4.48066473,  4.45002413,
         4.57506418],
       [ 3.73110366,  3.73304677,  3.77626538,  3.82045794,  3.86061025,
         3.98797059,  4.00682211,  4.01210403,  4.05254269,  4.05303097,
         4.03006411,  4.09348297,  4.11843157,  4.09019566,  4.14616728,
         4.1283989 ,  4.24848175,  4.38145256,  4.37118912,  4.33422756,
         4.3796072 ],
       [ 3.50528574,  3.48095036,  3.47693229,  3.47413278,  3.46177268,
         3.54744792,  3.56183004,  3.57046652,  3.62289333,  3.63726234,
         3.61886692,  3.65653515,  3.69141197,  3.71871161,  3.78194213,
         3.78716326,  3.88151574,  3.99901366,  3.97630072,  3.92656898,
         3.9503684 ],
       [ 3.2680223 ,  3.23132586,  3.20671487,  3.18805742,  3.16313982,
         3.190413  ,  3.1878345 ,  3.18762779,  3.22398376,  3.23962164,
         3.24318838,  3.28671408,  3.32802677,  3.38321304,  3.46056056,
         3.49339056,  3.59654093,  3.71596766,  3.73086977,  3.71015453,
         3.70373225],
       [ 3.06356192,  3.03554106,  3.01693726,  3.02687716,  3.03124356,
         3.04232788,  3.01726747,  2.99540806,  2.99935007,  2.98873186,
         2.97605062,  2.99227047,  3.00503445,  3.04576087,  3.11360979,
         3.15557528,  3.2641654 ,  3.40280008,  3.47608447,  3.49072981,
         3.51529551],
       [ 2.70898509,  2.68476486,  2.68376923,  2.72020507,  2.72268271,
         2.73325539,  2.7070446 ,  2.69143867,  2.69957137,  2.67735338,
         2.65350723,  2.652776  ,  2.64625525,  2.67726088,  2.73967147,
         2.7849915 ,  2.89022088,  2.99528027,  3.03814435,  3.08511257,
         3.15815926],
       [ 2.35342741,  2.30247355,  2.28649855,  2.32690835,  2.33321524,
         2.32074523,  2.27882051,  2.2697401 ,  2.2982955 ,  2.28166461,
         2.25007772,  2.2415874 ,  2.24205756,  2.27121544,  2.3213768 ,
         2.37551975,  2.46621156,  2.55871439,  2.63880897,  2.70321441,
         2.7477119 ],
        [ 1.9695704 ,  1.90600169,  1.87886906,  1.92075431,  1.92273378,
         1.87913156,  1.7955296 ,  1.79510331,  1.85373855,  1.85309505,
         1.86568034,  1.91241431,  1.95265603,  2.03287935,  2.15467167,
         2.23511219,  2.25584888,  2.24465513,  2.26965451,  2.30513525,
         2.30868149],
       [ 1.57320249,  1.46595001,  1.31550741,  1.18043661,  1.04311585,
         0.8760373 ,  0.6584779 ,  0.50445235,  0.42764819,  0.34536636,
         0.318591  ,  0.34360659,  0.41569793,  0.55796242,  0.65119445,
         0.70284212,  0.77141982,  0.82422417,  0.85692918,  0.91867346,
         0.9497605 ],
       [np.nan,np.nan,np.nan,np.nan,np.nan,
        np.nan,np.nan,np.nan,np.nan,np.nan,
        np.nan,np.nan,np.nan,np.nan,np.nan,
        np.nan,np.nan,np.nan,np.nan,np.nan,
        np.nan]])
    

zlevc = np.array([-4.54545458e-04, -1.36363634e-03, -2.36363639e-03, -3.54545447e-03,
       -4.90909070e-03, -6.45454554e-03, -8.18181783e-03, -1.01818182e-02,
       -1.24545451e-02, -1.49999997e-02, -1.79999992e-02, -2.14545447e-02,
       -2.52727270e-02, -2.96363644e-02, -3.46363634e-02, -4.02727276e-02,
       -4.67272699e-02, -5.41818179e-02, -6.26363605e-02, -7.22727254e-02,
       -8.32727253e-02, -9.57272723e-02, -1.09999999e-01, -1.26272723e-01,
       -1.44727275e-01, -1.65818185e-01, -1.89909101e-01, -2.17363626e-01,
       -2.48636365e-01, -2.84272730e-01, -3.24909091e-01, -3.71272743e-01,
       -4.24181819e-01, -4.84363616e-01, -5.52999973e-01, -6.31272733e-01,
       -7.20454574e-01, -8.22181821e-01, -9.38181818e-01, -1.00000000e+00])

target_topoz_pom = np.array([5500., 5500., 5500., 5500., 5500., 5500., 5500., 5500., 5500.,
       5500., 5500., 5500., 5500., 5500., 5500., 5500., 5500., 5500.,
       5500., 5500., 5500.])  
    
z_matrix_pom = np.dot(target_topoz_pom.reshape(-1,1),zlevc.reshape(1,-1)).T

    
#%% Interpolate glider transect onto POM time and depth
    
oktimeg_pom = np.round(np.interp(timestamp_pom,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)    
    
tempg_topom = np.empty((target_temp_pom.shape[0],target_temp_pom.shape[1]))
tempg_topom[:] = np.nan
for i in np.arange(len(oktimeg_pom)):
    pos = np.argsort(depthg[:,oktimeg_pom[i]])
    tempg_topom[:,i] = np.interp(-z_matrix_pom[:,i],depthg[pos,oktimeg_pom[i]],tempg[pos,oktimeg_pom[i]])

#%% Define dataframe

DF_GOFS = pd.DataFrame(data=np.array([np.ravel(tempg_to31,order='F'),np.ravel(target_temp31,order='F')]).T,\
                  columns=['Obs_value','GOFS_value'])

#%% Define dataframe

DF_POM = pd.DataFrame(data=np.array([np.ravel(tempg_topom,order='F'),np.ravel(target_temp_pom,order='F')]).T,\
                  columns=['Obs_value','POM_exp_value'])

#%% And calculate the statistics.

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['Obs_value']['GOFS_value']
#tskill[1,0] = DF_POM.corr()['Obs_value']['POM_oper_value']
tskill[2,0] = DF_POM.corr()['Obs_value']['POM_exp_value']

#OSTD
tskill[0,1] = DF_GOFS.std().Obs_value
tskill[1,1] = DF_POM.std().Obs_value
tskill[2,1] = DF_POM.std().Obs_value

#MSTD
tskill[0,2] = DF_GOFS.std().GOFS_value
#tskill[1,3] = stdevs.hycom_value
tskill[2,2] = DF_POM.std().POM_exp_value

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.Obs_value-DF_GOFS.mean().Obs_value)-(DF_GOFS.GOFS_value-DF_GOFS.mean().GOFS_value))**2)/NGOFS)
#tskill[1,3] = np.sqrt(np.nansum(((df1.obs_value-means.obs_value)-(df1.hycom_value-means.hycom_value))**2)/N)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.Obs_value-DF_POM.mean().Obs_value)-(DF_POM.POM_exp_value-DF_POM.mean().POM_exp_value))**2)/NGOFS)

#BIAS
tskill[0,4] = DF_GOFS.mean().Obs_value - DF_GOFS.mean().GOFS_value
#tskill[1,4] = means.obs_value-means.hycom_value
tskill[2,4] = DF_POM.mean().Obs_value - DF_POM.mean().POM_exp_value

#color
colors = ['indianred','seagreen','darkorchid']
    
skillscores=pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp'],
                        columns=cols)
print(skillscores)

#%% Create a plotting function. In this case for Taylor diagrams.

def taylor(scores,colors):

    fig = plt.figure()
    tr = PolarAxes.PolarTransform()
    
    CCgrid= np.concatenate((np.arange(0,10,2)/10.,[0.9,0.95,0.99]))
    CCpolar=np.arccos(CCgrid)
    gf=FixedLocator(CCpolar)
    tf=DictFormatter(dict(zip(CCpolar, map(str,CCgrid))))
    
    STDgrid=np.arange(0,np.round(skillscores.MSTD[0]+1,1),2)
    gfs=FixedLocator(STDgrid)
    tfs=DictFormatter(dict(zip(STDgrid, map(str,STDgrid))))
    
    max_std = np.nanmax([skillscores.OSTD,skillscores.MSTD])
    
    ra0, ra1 =0, np.pi/2
    cz0, cz1 = 0, np.round(max_std+1,1)
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=gf,
        tick_formatter1=tf,
        grid_locator2=gfs,
        tick_formatter2=tfs)
    
    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax1)
    
    ax1.axis["top"].set_axis_direction("bottom")  
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    ax1.axis["top"].label.set_text("Correlation")
   
    
    ax1.axis["left"].set_axis_direction("bottom") 
    ax1.axis["left"].label.set_text("Standard deviation")

    ax1.axis["right"].set_axis_direction("top")  
    ax1.axis["right"].toggle(ticklabels=True)
    ax1.axis["right"].major_ticklabels.set_axis_direction("left")
    
    ax1.axis["bottom"].set_visible(False) 
    ax1 = ax1.get_aux_axes(tr)
    
    plt.grid(linestyle=':',alpha=0.5)
    
    for i,r in enumerate(skillscores.iterrows()):
        theta=np.arccos(r[1].CORRELATION)
        rr=r[1].MSTD
        
        ax1.plot(theta,rr,'o',label=r[0],color = colors[i])
    
    ax1.plot(0,r[1].OSTD,'o',label='Obs')    
    plt.legend(loc='upper right',bbox_to_anchor=[1.3,1.15])    
    plt.show()
    
    rs,ts = np.meshgrid(np.linspace(0,11),np.linspace(0,np.pi))
    
    rms = np.sqrt(skillscores.OSTD[0]**2 + rs**2 - 2*rs*skillscores.OSTD[0]*np.cos(ts))
    
    contours = ax1.contour(ts, rs, rms,5,colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10)
    plt.grid(linestyle=':',alpha=0.5)
    
    for i,r in enumerate(skillscores.iterrows()):
        print(r)
        crmse = np.sqrt(r[1].OSTD**2 + r[1].MSTD**2 \
                   - 2*r[1].OSTD*r[1].MSTD*r[1].CORRELATION) 
        c1 = ax1.contour(ts, rs, rms,[crmse],colors=colors[i])
        plt.clabel(c1, inline=1, fontsize=10)
    
#%%    
    
taylor(skillscores,colors)
    