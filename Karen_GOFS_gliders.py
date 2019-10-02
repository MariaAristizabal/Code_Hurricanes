#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:40:04 2019

@author: aristizabal
"""

#%% User input

#lon_lim = [-100.0,-55.0]
#lat_lim = [10.0,45.0]

lon_lim = [-100.0,-40.0]
lat_lim = [5.0,45.0]

# Server erddap url 
server = 'https://data.ioos.us/gliders/erddap'

#gliders sg664, sg668, ng278
gdata_sg664 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG664-20190716T1218/SG664-20190716T1218.nc3.nc'
gdata_sg668 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
gdata_ng278 = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng278-20190901T0000/ng278-20190901T0000.nc3.nc'
gdata_ru29 = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ru29-20190906T1535/ru29-20190906T1535.nc3.nc'

#Time window
date_ini = '2019/09/20/00/00'
date_end = '2019/09/26/00/00'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# KMZ file
kmz_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/al122019_best_track.kmz'
#url_nhc = 'https://www.nhc.noaa.gov/gis/'

# Hurricane category figures
#ts_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/ts_nhemi.png'

# url for GOFS 3.1
url_GOFS31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xarray as xr
import netCDF4
from netCDF4 import Dataset
import cmocean
import os
from datetime import datetime,timedelta
import matplotlib.dates as mdates
from bs4 import BeautifulSoup
from zipfile import ZipFile
import sys
from erddapy import ERDDAP
import pandas as pd

'''
import requests
import urllib.request
from bs4 import BeautifulSoup
import glob
from zipfile import ZipFile
'''

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import read_glider_data_thredds_server
#from process_glider_data import grid_glider_data_thredd

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Download kmz files
'''
os.system('rm -rf *best_track*')
os.system('rm -rf *TRACK_latest*')
os.system('rm -rf *CONE_latest*')

r = requests.get(url_nhc)
data = r.text

soup = BeautifulSoup(data,"html.parser")

for i,s in enumerate(soup.find_all("a")):
    ff = s.get('href')
    if type(ff) == str:
        if np.logical_and('kmz' in ff, str(tini.year) in ff):
            if 'CONE_latest' in ff:
                file_name = ff.split('/')[3]
                print(ff, file_name)
                urllib.request.urlretrieve(url_nhc[:-4] + ff , file_name)
            if 'TRACK_latest' in ff:
                file_name = ff.split('/')[3]
                print(ff, file_name)
                urllib.request.urlretrieve(url_nhc[:-4] + ff ,file_name)
            if 'best_track' in ff:
                file_name = ff.split('/')[1]
                print(ff,file_name)
                urllib.request.urlretrieve(url_nhc + ff ,file_name)

#%%
kmz_files = glob.glob('*.kmz')

# NOTE: UNTAR  the .kmz FILES AND THEN RUN FOLLOWING CODE
for f in kmz_files:
    os.system('cp ' + f + ' ' + f[:-3] + 'zip')
    #os.system('mkdir ' + f[:-4])
    os.system('unzip -o ' + f + ' -d ' + f[:-4])

#%% get names zip and kml files
zip_files = glob.glob('*.zip')
zip_files = [f for f in zip_files if np.logical_or('al' in f,'AL' in f)]
'''
#%% GOGF 3.1

df = xr.open_dataset(url_GOFS31,decode_times=False)

#%%
## Decode the GOFS3.1 time into standardized mdates datenums 

#Time window
date_ini = '2019/09/20/00/00'
date_end = '2019/09/25/09/00'

hours_since2000 = df.time
time_naut       = datetime(2000,1,1)
time31 = np.ones_like(hours_since2000)
for ind, hrs in enumerate(hours_since2000):
    time31[ind] = mdates.date2num(time_naut+datetime.timedelta(hours=int(hrs)))

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

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Reading KMZ file

os.system('cp ' + kmz_file + ' ' + kmz_file[:-3] + 'zip')
os.system('unzip -o ' + kmz_file + ' -d ' + kmz_file[:-4])
kmz = ZipFile(kmz_file[:-3]+'zip', 'r')
kml_file = kmz_file.split('/')[-1].split('_')[0] + '.kml'
kml_best_track = kmz.open(kml_file, 'r').read()

# best track coordinates
soup = BeautifulSoup(kml_best_track,'html.parser')

lon_best_track = np.empty(len(soup.find_all("point")))
lon_best_track[:] = np.nan
lat_best_track = np.empty(len(soup.find_all("point")))
lat_best_track[:] = np.nan
for i,s in enumerate(soup.find_all("point")):
    print(s.get_text("coordinates"))
    lon_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
    lat_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
           
#  get time stamp
time_best_track = []
for i,s in enumerate(soup.find_all("atcfdtg")):
    tt = datetime.datetime.strptime(s.get_text(' '),'%Y%m%d%H')
    time_best_track.append(tt)
time_best_track = np.asarray(time_best_track)    

# get type 
wind_int = []
for i,s in enumerate(soup.find_all("intensitymph")):
    wind_int.append(s.get_text(' ')) 
wind_int = np.asarray(wind_int)  
  
cat = []
for i,s in enumerate(soup.find_all("styleurl")):
    cat.append(s.get_text('#').split('#')[-1]) 
cat = np.asarray(cat)  

# get name 
for i,s in enumerate(soup.find_all("name")):
    if s.get_text('name')[0:2] == 'AL':
        name_storm = s.get_text('name').split(' ')[-1]
        
#%% Figures temperature at surface

#okd = np.where(depth31 >= 100)[0][0]
#okt = np.round(np.interp(time31[oktime31],timeg,np.arange(len(timeg)))).astype(int)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31[::2]):
    T31 = df.water_temp[ind,0,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(7,5)) 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Surface Temperature  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(20,33,14))
    #kw = dict(levels=np.linspace(20,33,6))
    #kw = dict(levels=np.linspace(28,32,14))
        
    plt.contourf(lon31g,lat31g, var, cmap=cmocean.cm.thermal,**kw)
    
    okt = np.where(mdates.date2num(time_best_track) == time31[ind])[0][0]
    
    '''
    cat_fig = ts_fig
    cat_figg = mpimg.imread(cat_fig)
    imagebox = OffsetImage(cat_figg, zoom=0.2)
    ab = AnnotationBbox(imagebox, (lon_best_track[okt], lat_best_track[okt]))
    ax.add_artist(ab)
    '''
    
    plt.plot(lon_best_track[okt],lat_best_track[okt],'or',label=name_storm + ' ,'+ cat[okt])
    plt.legend(loc='upper left',fontsize=14)
    plt.plot(lon_best_track[0:okt],lat_best_track[0:okt],'.',color='grey')
    
    plt.axis('scaled')
    cb = plt.colorbar()
    cb.set_label('($^oC$)',rotation=90, labelpad=25, fontsize=12)
    
    file = folder + '{0}_{1}.png'.format('Salt_GOFS31_VI_100m ',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
#%% Reading glider data
    
url_glider = gdata_ng278
#url_glider = gdata_sg664
#url_glider = gdata_sg668
#url_glider = gdata_ru29

#del depthg_gridded, tempg_gridded, saltg_gridded, densg_gridded

var = 'temperature'
date_ini = '2019/09/15/00' # year/month/day/hour
date_end = '2019/09/26/00' # year/month/day/hour
scatter_plot = 'yes'
kwargs = dict(date_ini=date_ini,date_end=date_end)

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
tempg = np.asarray(varg.T)   

var = 'salinity'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)         
             
saltg = np.asarray(varg.T)  
 
var = 'density'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
densg = np.asarray(varg.T)
depthg = np.asarray(depthg.T)                
  
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
        
#%% Get rid off of profiles with no data below 100 m
'''
tempg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[t,:])
    if sum(depthg_gridded[okt] > 100) > 10:
        if tempg_gridded[t,0] != tempg_gridded[t,20]:
            tempg_full.append(tempg_gridded[t,:]) 
            timeg_full.append(tt) 
       
tempg_full = np.asarray(tempg_full)
timeg_full = np.asarray(timeg_full)
'''

#%% Read GOFS 3.1 output
    
print('Retrieving coordinates from model')
model = xr.open_dataset(url_GOFS31,decode_times=False)
    
lat31 = model.lat[:]
lon31 = model.lon[:]
depth31 = model.depth[:]
tt31 = model.time
t31 = netCDF4.num2date(tt31[:],tt31.units) 

tmin = datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H')
tmax = datetime.datetime.strptime(date_end,'%Y/%m/%d/%H')

oktime31 = np.where(np.logical_and(t31 >= tmin, t31 <= tmax))
time31 = t31[oktime31]
    
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
sublon31=np.interp(tstamp_model,tstamp_glider,target_lon)
sublat31=np.interp(tstamp_model,tstamp_glider,target_lat)

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

#%% time of Karen passing closets to glider
    
#tDorian = np.tile(datetime.datetime(2019,8,28,18),len(np.arange(-1000,0))) #ng665, ng666 
#tDorian = np.tile(datetime.datetime(2019,8,28,6),len(np.arange(-1000,0))) #ng668
tKaren = np.tile(datetime.datetime(2019,9,24,4),len(np.arange(-1000,0))) #silbo    
    
#%%
color_map = cmocean.cm.thermal
       
okg = depthg_gridded <= np.max(depthg_gridded) 
okm = depth31 <= np.max(depthg_gridded) 
min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg,:]),np.nanmin(target_temp31[okm])]))
max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg,:]),np.nanmax(target_temp31[okm])]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    

# plot
fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=color_map,**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded,[26],colors='k')

cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xticklabels(' ')

plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + inst_id)
   
ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_temp31,cmap=color_map,**kw)
plt.contour(mdates.date2num(time31),-depth31,target_temp31,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + 'GOFS 3.1')  

file = folder + ' ' + 'along_track_temp ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the 
# mixed layer and td: temp at 1 meter below the mixed layer            

d10 = np.where(depthg_gridded >= 10)[0][0]
dt = 0.2

MLD_dt = np.empty(len(timeg)) 
MLD_dt[:] = np.nan
Tmean_dtemp = np.empty(len(timeg)) 
Tmean_dtemp[:] = np.nan
Td = np.empty(len(timeg)) 
Td[:] = np.nan
for t,tt in enumerate(timeg):
    T10 = tempg_gridded[d10,t]
    delta_T = T10 - tempg_gridded[:,t] 
    ok_mld = np.where(delta_T <= dt)[0]    
    if ok_mld.size == 0:
        MLD_dt[t] = np.nan
        Tmean_dtemp[t] = np.nan
        Td[t] = np.nan
    else:
        ok_mld_plus1m = np.where(depthg_gridded >= depthg_gridded[ok_mld[-1]] + 1)[0][0]
        MLD_dt[t] = depthg_gridded[ok_mld[-1]]
        Tmean_dtemp[t] = np.nanmean(tempg_gridded[ok_mld,t])
        Td[t] = tempg_gridded[ok_mld_plus1m,t]
        
#%%  Calculation of mixed layer depth based on drho

d10 = np.where(depthg_gridded >= 10)[0][0]
drho = 0.125

MLD_drho = np.empty(len(timeg)) 
MLD_drho[:] = np.nan
Tmean_drho = np.empty(len(timeg)) 
Tmean_drho[:] = np.nan
for t,tt in enumerate(timeg):
    rho10 = densg_gridded[d10,t]
    delta_rho = -(rho10 - densg_gridded[:,t]) 
    ok_mld = np.where(delta_rho <= drho)
    if ok_mld[0].size == 0:
        MLD_drho[t] = np.nan
        Tmean_drho[t] = np.nan
    else:
        MLD_drho[t] = depthg_gridded[ok_mld[0][-1]] 
        Tmean_drho[t] = np.nanmean(tempg_gridded[ok_mld,t])        
        
#%% Tmean and Td
        
fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg,Tmean_dtemp,'.-',color='grey',label='Tmean mixed layer depth, temperature criteria')
plt.plot(timeg,Tmean_drho,'.-',color='lightgreen',label='Tmean mixed layer depth, density criteria')
#plt.plot(timeg_low,Tmean_low,'-',color='grey',label='12 hours lowpass')
#plt.plot(timeg,Td,'.-g',label='Td')
#plt.plot(timeg[np.isfinite(Td)],Td_low,'-',color='lightseagreen',label='12 hours lowpass')
plt.legend()
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])
#plt.xlim([datetime(2018,9,10),datetime(2018,9,15)])
plt.ylabel('$^oC$',fontsize = 14)
#tDorian = np.tile(datetime.datetime(2019,8,28,18),len(np.arange(29.0,29.30,0.01)))# ng665,ng666
#tDorian = np.tile(datetime.datetime(2019,8,29,6),len(np.arange(29.0,29.30,0.01)))  # ng668
tKaren = np.tile(datetime.datetime(2019,9,24,4),len(np.arange(29.2,29.7,0.01))) #silbo 
plt.plot(tKaren,np.arange(29.2,29.7,0.01),'--k')
plt.title(inst_id)

file = folder + ' ' + inst_id + '_Tmean_Td'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Top 200 m salinity

color_map = cmocean.cm.thermal
       
okg = depthg_gridded <= 200
okm = depth31 <= 200 
min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp31[okm])]))
max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp31[okm])]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    
# plot
fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=color_map,**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded,[26],colors='k')
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='grey' )
plt.plot(timeg,-MLD_drho,'-',label='MLD drho',color='lightgreen' )

cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xticklabels(' ')
#tDorian = np.tile(datetime.datetime(2019,8,28,18),len(np.arange(-1000,0)))
#tDorian = np.tile(datetime(2019,8,28,6),len(np.arange(-1000,0)))
tKaren = np.tile(datetime.datetime(2019,9,24,4),len(np.arange(-1000,0))) #silbo 
plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + inst_id)
plt.legend()   

ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_temp31,cmap=color_map,**kw)
plt.contour(mdates.date2num(time31),-depth31,target_temp31,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Along Track ' + 'Temperature' + ' Profile ' + 'GOFS 3.1')  
plt.plot(tKaren,np.arange(-1000,0),'--k')

file = folder + ' ' + 'along_track_temp_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Top 200 m salinity

color_map = cmocean.cm.haline
       
okg = depthg_gridded <= 200
okm = depth31 <= 200 
min_val = np.floor(np.min([np.nanmin(saltg_gridded[okg]),np.nanmin(target_salt31[okm])]))
max_val = np.ceil(np.max([np.nanmax(saltg_gridded[okg]),np.nanmax(target_salt31[okm])]))
    
#nlevels = max_val - min_val + 1
#nlevels = (max_val - min_val + 1)*4
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))

kw = dict(levels = np.linspace(33.0,37.8,17))
    
# plot
fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,saltg_gridded,cmap=color_map,**kw)
plt.contour(timeg,-depthg_gridded,saltg_gridded,[26],colors='k')
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='grey' )
plt.plot(timeg,-MLD_drho,'-',label='MLD drho',color='lightgreen' )

cs = fig.colorbar(cs, orientation='vertical') 
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xticklabels(' ')
#tDorian = np.tile(datetime.datetime(2019,8,28,18),len(np.arange(-1000,0)))
tDorian = np.tile(datetime.datetime(2019,8,28,6),len(np.arange(-1000,0)))
plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + inst_id)
plt.legend()   

ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_salt31,cmap=color_map,**kw)
plt.contour(mdates.date2num(time31),-depth31,target_salt31,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + 'GOFS 3.1')  

file = folder + ' ' + 'along_track_salt_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Time series surface salinity

okdg = np.where(depthg_gridded >= 10)[0][0]
okd31 = np.where(depth31 >= 10)[0][0]

fig, ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg,saltg_gridded[okdg,:],'.-b',label=inst_id.split('-')[0])
plt.plot(time31,target_salt31[okd31,:],'.-r',label='GOFS 3.1')
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
tKaren = np.tile(datetime.datetime(2019,9,24,4),len(np.arange(33.25,35.0,0.01))) #silbo 
plt.plot(tKaren,np.arange(33.25,35.0,0.01),'--k')
plt.legend()
plt.title('Surface Salinity' + ' Time Series ' + inst_id,fontsize=16)

file = folder + ' ' + 'surface_salinity_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Density transect

fig,ax = plt.subplots(figsize=(15, 4))

nlevels = np.round(np.nanmax(densg_gridded)) - np.round(np.nanmin(densg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(densg_gridded)),\
                               np.round(np.nanmax(densg_gridded)),11))
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,densg_gridded,cmap=cmocean.cm.dense,**kw)
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='grey' )
plt.plot(timeg,-MLD_drho,'-',label='MLD drho',color='lightgreen' )
plt.xlim(timeg[0],timeg[-1])
plt.ylim([-200,0])
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('$kg/m^3$',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)  
plt.title('Density Transect ' + inst_id,fontsize=18) 
plt.legend() 
#tDorian = np.tile(datetime.datetime(2019,8,28,18),len(np.arange(-1000,0)))
plt.plot(tKaren,np.arange(-1000,0),'--k')

file = folder + ' ' + inst_id + '_dens_200m'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
     

#%% Salinity transect

fig,ax = plt.subplots(figsize=(15, 4))

nlevels = np.ceil(np.nanmax(saltg_gridded)) - np.floor(np.nanmin(saltg_gridded)) + 1
kw = dict(levels = np.linspace(np.floor(np.nanmin(saltg_gridded)),\
                               np.ceil(np.nanmax(saltg_gridded)),nlevels*3))
#kw = dict(levels = np.linspace(34.8,37.4,27))
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,saltg_gridded,cmap=cmocean.cm.haline,**kw)
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='grey' )
plt.plot(timeg,-MLD_drho,'-',label='MLD drho',color='lightgreen' )
plt.xlim(timeg[0],timeg[-1])
plt.ylim([-200,0])
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('  ',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)  
plt.title('Salinity Transect ' + inst_id,fontsize=18) 
plt.legend() 
#tDorian = np.tile(datetime.datetime(2019,8,28,18),len(np.arange(-1000,0)))
plt.plot(tKaren,np.arange(-1000,0),'--k')

file = folder + ' ' + inst_id + '_salt_200m'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
  
   
#%% Get time bounds

# Time bounds
min_time = '2019-09-22T00:00:00Z'
max_time = '2019-09-23T00:00:00Z'

#%% Look for datasets in IOOS glider dac

print('Looking for glider data sets')
e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time,
    'max_time': max_time,
}

search_url = e.get_search_url(response='csv', **kw)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

# Setting constraints
constraints = {
        'time>=': min_time,
        'time<=': max_time,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time'
        ]

e = ERDDAP(
        server=server,
        protocol='tabledap',
        response='nc'
        )

#%% Map of North Atlantic with glider 

col = ['red','darkcyan','gold','m','darkorange','crimson','lime',\
       'darkorchid','brown','sienna','yellow','orchid','gray',\
       'darkcyan','gold','m','darkorange','crimson','lime','red',\
       'darkorchid','brown','sienna','yellow','orchid','gray']
mark = ['o','*','p','^','D','X','o','*','p','^','D','X','o',\
        'o','*','p','^','D','X','o','*','p','^','D','X','o']
#edgc = ['k','w','k','w','k','w','k','w','k','w','k','w','k','w','k']

fig, ax = plt.subplots(figsize=(7, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap='Blues_r')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.yticks([])
plt.xticks([])
plt.axis([-70,-60,10,20])
plt.title('Active Glider Deployments on ' + min_time[0:10],fontsize=20)

okt = np.where(time_best_track == tmax)[0][0]
plt.plot(lon_best_track[okt],lat_best_track[okt],'or')
plt.plot(lon_best_track[0:okt],lat_best_track[0:okt],'.',color='grey')
plt.text(lon_best_track[okt],lat_best_track[okt],name_storm,fontsize=14)

for i,id in enumerate(gliders[2:]):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables

        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.nanmean(df['longitude (degrees_east)']),\
                np.nanmean(df['latitude (degrees_north)']),'-',color=col[i],\
                marker = mark[i],markeredgecolor = 'k',markersize=14,\
                label=id.split('-')[0])
        
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])

ax.legend(fontsize=14,loc='upper right',bbox_to_anchor = [1.2,1])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Daily_map_North_Atlantic_gliders_in_DAC_' + min_time[0:10] + '_' + max_time[0:10] + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Map of North Atlantic with glider tracks

# Look for datasets in IOOS glider dac

# Time bounds
min_time = '2019-09-20T00:00:00Z'
max_time = '2019-09-27T00:00:00Z'

lat_lim = [10,50]
lon_lim = [-100,-50]

# Reading bathymetry data
ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

print('Looking for glider data sets')
e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time,
    'max_time': max_time,
}

search_url = e.get_search_url(response='csv', **kw)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

#% get entire deployment (lat and lon) during hurricane season

# Time bounds
min_time2 = '2019-06-01T00:00:00Z'
max_time2 = '2019-11-30T00:00:00Z'

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time2,
    'max_time': max_time2,
}

search_url = e.get_search_url(response='csv', **kw)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders_all = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders_all), '\n'.join(gliders_all)))

# Setting constraints
constraints = {
        'time>=': min_time2,
        'time<=': max_time2,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time'
        ]

e = ERDDAP(
        server=server,
        protocol='tabledap',
        response='nc'
        )


col = ['red','darkcyan','gold','m','darkorange','crimson','lime',\
       'darkorchid','brown','sienna','yellow','orchid','gray',\
       'darkcyan','gold','m','darkorange','crimson','lime','red',\
       'darkorchid','brown','sienna','yellow','orchid','gray']
mark = ['o','*','p','^','D','X','o','*','p','^','D','X','o',\
        'o','*','p','^','D','X','o','*','p','^','D','X','o']
#edgc = ['k','w','k','w','k','w','k','w','k','w','k','w','k','w','k']

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 5))
#plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
#plt.colorbar()
plt.yticks([])
plt.xticks([])
#plt.axis([-70,-60,13,23])
plt.title('Active Glider Deployments '+ min_time[0:10]+'-'+max_time[0:10] ,fontsize=20)
plt.plot(lon_best_track[5:],lat_best_track[5:],'or',markersize=3)
ax.set_aspect(1)
lat_lim = [15,23]
lon_lim = [-70,-63.5]
rect = patches.Rectangle((-70,15),\
                             -63.5-(-70),23-15,\
                             linewidth=2,edgecolor='k',facecolor='none')
ax.add_patch(rect)


for i,id_all in enumerate(gliders_all):
    id = [id for id in gliders if id == id_all]
    if len(id) != 0:
        print(id[0])     
        e.dataset_id = id[0]
        e.constraints = constraints
        e.variables = variables

        df = e.to_pandas(
                index_col='time (UTC)',
                parse_dates=True,
                skiprows=(1,)  # units information can be dropped.
                ).dropna()
        
        print(len(df))
               
        timeg, ind = np.unique(df.index.values,return_index=True)
        latg = df['latitude (degrees_north)'].values[ind]
        long = df['longitude (degrees_east)'].values[ind]
            #ax.plot(np.nanmean(df['longitude (degrees_east)']),\
            #        np.nanmean(df['latitude (degrees_north)']),'-',color=col[i],\
            #        marker = mark[i],markeredgecolor = 'k',markersize=8,\
            #        label=id.split('-')[0])
        ax.plot(long,latg,'.-',color='darkorange',markersize=1)
            #ax.plot(df['longitude (degrees_east)'][len(df['longitude (degrees_east)'])-1],\
            #     df['latitude (degrees_north)'][len(df['longitude (degrees_east)'])-1],\
            #     '-',color=col[i],\
            #     marker = mark[i],markeredgecolor = 'k',markersize=6,\
            #     label=id.split('-')[0])
        
            #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])

#ax.legend(fontsize=14,bbox_to_anchor = [1,1])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Daily_map_North_Atlantic_gliders_in_DAC_' + min_time[0:10] + '_' + max_time[0:10] + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)


#%% Map detail

# Time bounds
min_time = '2019-09-20T00:00:00Z'
max_time = '2019-09-27T00:00:00Z'

lat_lim = [15,23]
lon_lim = [-70,-63.5]

# Reading bathymetry data
ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

print('Looking for glider data sets')
e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time,
    'max_time': max_time,
}

search_url = e.get_search_url(response='csv', **kw)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

#% get entire deployment (lat and lon) during hurricane season

# Time bounds
min_time2 = '2019-06-01T00:00:00Z'
max_time2 = '2019-11-30T00:00:00Z'

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time2,
    'max_time': max_time2,
}

search_url = e.get_search_url(response='csv', **kw)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders_all = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders_all), '\n'.join(gliders_all)))

# Setting constraints
constraints = {
        'time>=': min_time2,
        'time<=': max_time2,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time'
        ]

e = ERDDAP(
        server=server,
        protocol='tabledap',
        response='nc'
        )


col = ['red','darkcyan','gold','m','darkorange','crimson','lime',\
       'darkorchid','brown','sienna','yellow','orchid','gray',\
       'darkcyan','gold','m','darkorange','crimson','lime','red',\
       'darkorchid','brown','sienna','yellow','orchid','gray']
mark = ['o','*','p','^','D','X','o','*','p','^','D','X','o',\
        'o','*','p','^','D','X','o','*','p','^','D','X','o']

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 5))
#plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.yticks([])
plt.xticks([])
plt.axis([lon_lim[0],lon_lim[1],lat_lim[0],lat_lim[1]])
plt.title('Active Glider Deployments '+ min_time[0:10]+'-'+max_time[0:10] ,fontsize=20)
plt.plot(lon_best_track[17:-3],lat_best_track[17:-3],'or',markersize=6)
ax.set_aspect(1)


for i,id_all in enumerate(gliders_all):
    id = [id for id in gliders if id == id_all]
    if len(id) != 0:
        print(id[0])     
        e.dataset_id = id[0]
        e.constraints = constraints
        e.variables = variables

        df = e.to_pandas(
                index_col='time (UTC)',
                parse_dates=True,
                skiprows=(1,)  # units information can be dropped.
                ).dropna()
        
        print(len(df))
               
        timeg, ind = np.unique(df.index.values,return_index=True)
        latg = df['latitude (degrees_north)'].values[ind]
        long = df['longitude (degrees_east)'].values[ind]
            #ax.plot(np.nanmean(df['longitude (degrees_east)']),\
            #        np.nanmean(df['latitude (degrees_north)']),'-',color=col[i],\
            #        marker = mark[i],markeredgecolor = 'k',markersize=8,\
            #        label=id.split('-')[0])
        ax.plot(long,latg,'.-',color='darkorange',markersize=1)
        ax.plot(long[-1],latg[-1],'o',color='k',markersize=4)
            #ax.plot(df['longitude (degrees_east)'][len(df['longitude (degrees_east)'])-1],\
            #     df['latitude (degrees_north)'][len(df['longitude (degrees_east)'])-1],\
            #     '-',color=col[i],\
            #     marker = mark[i],markeredgecolor = 'k',markersize=6,\
            #     label=id.split('-')[0])
        
        ax.text(long[-1],latg[-1],id[0].split('-')[0],weight='bold',
                bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))

#ax.legend(fontsize=14,bbox_to_anchor = [1,1])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Daily_map_North_Atlantic_gliders_in_DAC_detail_' + min_time[0:10] + '_' + max_time[0:10] + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

'''
from matplotlib.cbook import get_sample_data

hurr_img = plt.imread(get_sample_data('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/ts_nhemi.png'))
poo_axs = [None for i in range(len(lon_best_track))]
loc = ax.transData.transform((lon_best_track, lat_best_track))
poo_axs[i] = fig.add_axes([loc[0] loc[1]], anchor='C')

fig, ax = plt.subplots()
ax.plot(lon_best_track, lat_best_track, linestyle="-")
poo_axs[0].imshow(hurr_img)

#plt.plot(lon_best_track[5:],lat_best_track[5:],'or',markersize=3)
#plt.plot(lon_best_track[0],lat_best_track[0],hurr_img[0,0,0])
'''
