#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:11:06 2019

@author: aristizabal
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

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# KMZ file
kmz_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/al052019_best_track-5.kmz'

# Hurricane category figures
ts_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/ts_nhemi.png'

# url for GOFS 3.1
url_GOFS31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# hourly precipitation file
# data from https://www.ncdc.noaa.gov/cdo-web/datasets/LCD/stations/WBAN:11640/detail
csv_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/txt_csv_files/1899543.csv'

# Argo data
# Jun1 -Jul1
Dir_Argo = '/Volumes/aristizabal/ARGO_data/DataSelection_20191014_193816_8936308'

# NAVGEN 10 m wind output
file_NAVGEN = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/erdNavgem1D10mWind_6da5_e6b3_6991.nc' 

#%%

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xarray as xr
import netCDF4
from netCDF4 import Dataset
import cmocean
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from bs4 import BeautifulSoup
from zipfile import ZipFile
import sys
from erddapy import ERDDAP
import pandas as pd
import seawater as sw
import csv
import glob
import seawater

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import read_glider_data_thredds_server
#from process_glider_data import grid_glider_data_thredd

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Precipitation data from https://www.cocorahs.org/ViewData/ListDailyPrecipReports.aspx
# for station VI-SC-8 Virgin Islands

dayl_accum = np.array([0,0,0,0,np.nan,0,0,0,0,0.08,0.06,0.02,0.05,\
                       0,0.65,2.45,0.02,0,0,0.02,0,0,0.42,np.nan,\
                       0.03,0,0,0,0.01,0,0.24,0.32,0.83,1.04,0.02,\
                       0,0.01,0.01,0,0,0.04,1.61,0.31,0,0,0.03,1.09])

time_accum = np.asarray([datetime(2019,10,9,7)-timedelta(float(n)) for n in np.arange(47)])

#%% Read hourly precipitation

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
            
hourly_precip = np.asarray(hourly_precip)
time_hourly_precip = np.asarray(time_hourly_precip)

#%% Calculate 12 hourly precipitation
nn = 12 # accumulation every 6 hours
time_accum_precip = []
accum_precip = []
for i,pp in enumerate(hourly_precip[::nn+1]):
    time_accum_precip.append(mdates.num2date(np.mean(mdates.date2num(time_hourly_precip[nn*i:nn*i+nn]))))
    accum_precip.append(np.nansum(hourly_precip[nn*i:nn*i+nn]))
    
time_accum_precip = mdates.num2date(mdates.date2num(time_accum_precip))

#%% Reading Argo data

argo_files = sorted(glob.glob(os.path.join(Dir_Argo,'*.nc')))

ncargo = Dataset(argo_files[-1])
argo_id = ncargo.variables['PLATFORM_NUMBER'][:]
argo_lat = ncargo.variables['LATITUDE'][:]
argo_lon = ncargo.variables['LONGITUDE'][:]

argo_tim = ncargo.variables['JULD']#[:]
argo_time = netCDF4.num2date(argo_tim[:],argo_tim.units) 

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
    tt = datetime.strptime(s.get_text(' '),'%Y%m%d%H')
    time_best_track.append(tt)
time_best_track = np.asarray(time_best_track)    

# get type 
wind_int = []
for i,s in enumerate(soup.find_all("intensitymph")):
    wind_int.append(s.get_text(' ')) 
wind_int = np.asarray(wind_int)
wind_int = wind_int.astype(float)  

wind_int_kt = []
for i,s in enumerate(soup.find_all("intensity")):
    wind_int_kt.append(s.get_text(' ')) 
wind_int_kt = np.asarray(wind_int_kt)
wind_int_kt = wind_int_kt.astype(float)
  
cat = []
for i,s in enumerate(soup.find_all("styleurl")):
    cat.append(s.get_text('#').split('#')[-1]) 
cat = np.asarray(cat)  
   
#%% Figures temperature at surface

#okd = np.where(depth31 >= 100)[0][0]
#okt = np.round(np.interp(time31[oktime31],timeg,np.arange(len(timeg)))).astype(int)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31[46:81][::2]):
    T31 = df.water_temp[ind,0,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(7,5)) 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Surface Temperature  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(23,32,10))
        
    plt.contourf(lon31g,lat31g, var, cmap=cmocean.cm.thermal,**kw)
    
    okt = np.where(mdates.date2num(time_best_track) == time31[ind])[0][0]
    
    plt.plot(lon_best_track[okt],lat_best_track[okt],'or',label='Dorian ,'+ cat[okt])
    plt.legend(loc='upper left',fontsize=14)
    plt.plot(lon_best_track[14:okt],lat_best_track[14:okt],'.',color='grey')
    
    plt.axis('scaled')
    cb = plt.colorbar()
    cb.set_label('($^oC$)',rotation=90, labelpad=25, fontsize=12)
    
    file = folder + '{0}_{1}.png'.format('Temp_GOFS31_Caribbean_',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Surface salinity
'''
for i,ind in enumerate(oktime31[::4][6:-7]):
    S31 = df.salinity[ind,0,botm:top,left:right]
    var = S31

    fig, ax = plt.subplots(figsize=(7,5)) 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Surface Salinity  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(33,37.2,22))
        
    plt.contourf(lon31g,lat31g, var, cmap=cmocean.cm.haline,**kw)

    okt = np.where(mdates.date2num(time_best_track) >= time31[ind])[0][0]
    
    plt.plot(lon_best_track[okt],lat_best_track[okt],'or',label='Dorian ,'+ cat[okt])
    plt.legend(loc='upper right',fontsize=14)
    plt.plot(lon_best_track[14:okt],lat_best_track[14:okt],'.',color='grey')
    #plt.plot(-77.4,27.0,'o',color='yellowgreen')
    
    plt.axis('scaled')
    cb = plt.colorbar()
    #cb.set_label('($^oC$)',rotation=90, labelpad=25, fontsize=12)
    
    file = folder + '{0}_{1}.png'.format('Salt_GOFS31_Caribbean_',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   
'''

#%% Time series at point underneath Dorian when cat 5

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = df.water_temp[oktime31,0,indy,indx]

fig,ax1 = plt.subplots(figsize=(12, 4))
plt.plot(time31[oktime31],temp31,'o-',linewidth=2)

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])
plt.ylabel('$^oC$',fontsize = 14)
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(22,30,0.1)))
plt.plot(tDorian,np.arange(22,30,0.1),'--k')
plt.title('Surface Temperature GOFS 3.1',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

file = folder + ' ' + 'surf_temp_GOFS31_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the 
# mixed layer and td: temp at 1 meter below the mixed layer
# for GOFS 3.1 output at point of maximum cooling           

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = np.asarray(df.water_temp[oktime31,:,indy,indx]).T
salt31 = np.asarray(df.salinity[oktime31,:,indy,indx]).T

d10 = np.where(depth31 >= 10)[0][0]
dt = 0.2

t31 = time31[oktime31]

MLD31 = np.empty(len(t31)) 
MLD31[:] = np.nan
Tmean31 = np.empty(len(t31)) 
Tmean31[:] = np.nan
Smean31 = np.empty(len(t31)) 
Smean31[:] = np.nan
Td31 = np.empty(len(t31)) 
Td31[:] = np.nan
for t,tt in enumerate(t31):
    print(t)
    T10 = temp31[d10,t]
    delta_T = T10 - temp31[:,t] 
    ok_mld = np.where(delta_T <= dt)[0]    
    if ok_mld.size == 0:
        MLD31[t] = np.nan
        Tmean31[t] = np.nan
        Smean31[t] = np.nan
        Td31[t] = np.nan
    else:
        ok_mld_plus1m = np.where(depth31 >= depth31[ok_mld[-1]] + 1)[0][0]
        MLD31[t] = depth31[ok_mld[-1]]
        Tmean31[t] = np.nanmean(temp31[ok_mld,t])
        Smean31[t] = np.nanmean(salt31[ok_mld,t])
        Td31[t] = temp31[ok_mld_plus1m,t] 
        
#%% Tmean ad td

t31 = time31[oktime31]     
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)

fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(t31,Tmean31,'.-r',label='Tmeanmixed layer')
plt.plot(t31,Td31,'.-b',label='Td 1m below mixed layer')
plt.legend()
xfmt = mdates.DateFormatter('%d\n-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([t31[0],t31[-1]])
plt.ylabel('$^oC/hours$',fontsize = 14)
plt.xticks(xticks)         
        
#%% Calculation of dTmean_dt 
t31 = time31[oktime31]        
dTmean31_dt = (Tmean31[1:]-Tmean31[0:-1])/(t31[1:] - t31[0:-1])
dTmean31_dt = dTmean31_dt/24 # in hours 
  
time31_mid = (t31[1:] + t31[0:-1])/2

#%% Calculate dh/dt (Tmean - Td)/h

dh_dt = (MLD31[1:] - MLD31[0:-1]) / (t31[1:] - t31[0:-1])

time31_mid = (t31[1:] + t31[0:-1])/2

Tmean31_mid = np.interp(time31_mid,t31,Tmean31)
Td31_mid = np.interp(time31_mid,t31,Td31)
MLD31_mid = np.interp(time31_mid,t31,MLD31)

dT_dt_MLD_deep = dh_dt * (Tmean31_mid - Td31_mid)/MLD31_mid 
dT_dt_MLD_deep = dT_dt_MLD_deep/24 # in hours

#%% Calculate Ekman Pumping

# Download data from NAVGEN
NAVGEN = xr.open_dataset(file_NAVGEN,decode_times=False)

navgen_t = NAVGEN.variables['time']  
navgen_time = np.asarray(netCDF4.num2date(navgen_t[:],navgen_t.attrs['units']))

navgen_lat = np.asarray(NAVGEN.variables['latitude'][:])
navgen_lon = np.asarray(NAVGEN.variables['longitude'][:])
navgen_hag = np.asarray(NAVGEN.variables['height_above_ground'][:])
navgen_uwind = np.asarray(NAVGEN.variables['wnd_ucmp_height_above_ground'][:,0,:,:])
navgen_vwind = np.asarray(NAVGEN.variables['wnd_vcmp_height_above_ground'][:,0,:,:])
#%%
indx = np.where(navgen_lon > -77.4+360)[0][0]
indy = np.where(navgen_lat < 27.0)[0][-1]

navgen_uw_x0 = navgen_uwind[:,indy,indx]
navgen_uw_x1 = navgen_uwind[:,indy,indx+1]
navgen_vw_y0 = navgen_vwind[:,indy,indx]




#%% 
lat_lim = [8,50]
lon_lim = [-84,-40]
oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

lev = np.arange(-9000,9100,100)
plt.figure()
#plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
q = plt.quiver(navgen_lon-360, navgen_lat,navgen_uwind[0,0,:,:],navgen_vwind[0,0,:,:],units='xy' ,scale=5)
plt.quiverkey(q,-90,40,10,"10 m/s",coordinates='data',color='k')

#%% dTmean/dt

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)

fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(time31_mid,np.tile(0,len(time31_mid)),'--k')
plt.plot(time31_mid,dTmean31_dt,'.-r',label='dTmean/dt')
plt.plot(time31_mid,dT_dt_MLD_deep,'.-b',label='dh/dt')
plt.legend()
xfmt = mdates.DateFormatter('%d\n-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([t31[0],t31[-1]])
plt.ylabel('$^oC/hours$',fontsize = 14)
plt.xticks(xticks)
plt.ylim([-0.7,0.7])

file = folder + 'GOFS31_Tmean_Td'
#plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% temperature Time series at point underneath Dorian when cat 5

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = np.asarray(df.water_temp[oktime31,:,indy,indx]).T

color_map = cmocean.cm.thermal

maxd = 300      
okm = depth31 <= maxd
min_val = np.floor(np.nanmin(temp31[okm]))
max_val = np.ceil(np.nanmax(temp31[okm]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    
# plot
fig, ax = plt.subplots(figsize=(12, 3))
  
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(time31[oktime31],-depth31,temp31,cmap=color_map,**kw)
plt.contour(time31[oktime31],-depth31,temp31,[26],colors='k')
plt.plot(time31[oktime31],-MLD31,'.-',color='grey')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)

ax.set_xlim(datetime(2019,8,25),datetime(2019,9,8))
ax.set_ylim(-300, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(-maxd,0)))
plt.plot(tDorian,np.arange(-maxd,0),'--k')
plt.title('Temperature' + ' Profile ' + 'GOFS 3.1')  

file = folder + ' ' + 'temp_profile_point_max_cooling'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)


#%% Detail temperature Time series at point underneath Dorian when cat 5

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = np.asarray(df.water_temp[oktime31,:,indy,indx]).T

color_map = cmocean.cm.thermal

maxd = 300      
okm = depth31 <= maxd
min_val = np.floor(np.nanmin(temp31[okm]))
max_val = np.ceil(np.nanmax(temp31[okm]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    
# plot
fig, ax = plt.subplots(figsize=(4, 4))
  
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(time31[oktime31],-depth31,temp31,cmap=color_map,**kw)
plt.contour(time31[oktime31],-depth31,temp31,[26],colors='k')
plt.plot(time31[oktime31],-MLD31,'.-',color='grey')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)

ax.set_xlim(datetime(2019,8,25),datetime(2019,9,8))
ax.set_ylim(-300, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(-maxd,0)))
plt.plot(tDorian,np.arange(-maxd,0),'--k')
plt.title('Temperature' + ' Profile ' + 'GOFS 3.1') 

color_cat = ['deepskyblue','cyan','lemonchiffon','gold','orange',\
             'darkorange','red'] 

for tt,ca in enumerate(cat):
    if ca == 'cat2':
        plt.plot(time_best_track[tt],-10,'.',color=color_cat[3],\
                 markersize=15,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat3':
        plt.plot(time_best_track[tt],-10,'.',color=color_cat[4],\
                 markersize=15,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat4':
        plt.plot(time_best_track[tt],-10,'.',color=color_cat[5],\
                 markersize=15,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat5':
        plt.plot(time_best_track[tt],-10,'.',color=color_cat[6],\
                 markersize=15,markeredgecolor='k',markeredgewidth=2)
        
plt.xlim([datetime(2019,9,1),datetime(2019,9,4)])        

file = folder + ' ' + 'temp_profile_point_max_cooling_detail'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% salinity Time series at point underneath Dorian when cat 5

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

salt31 = np.asarray(df.salinity[oktime31,:,indy,indx]).T

color_map = cmocean.cm.haline

maxd = 300      
okm = depth31 <= maxd
min_val = np.floor(np.nanmin(salt31[okm]))
max_val = np.ceil(np.nanmax(salt31[okm]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    

# plot
fig, ax = plt.subplots(figsize=(12, 3))
  
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(time31[oktime31],-depth31,salt31,cmap=color_map)#,**kw)
plt.contour(time31[oktime31],-depth31,salt31,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)

ax.set_xlim(datetime(2019,8,25),datetime(2019,9,8))
ax.set_ylim(-300, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(-maxd,0)))
plt.plot(tDorian,np.arange(-maxd,0),'--k')
plt.title('Salinity Profile ' + 'GOFS 3.1')  

file = folder + ' ' + 'salt_profile_point_max_cooling'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
 

#%% Time series at point underneath Dorian when cat 5 
# With Dorian category

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = df.water_temp[oktime31,0,indy,indx]

fig,ax1 = plt.subplots(figsize=(12, 4))

plt.plot(time31[oktime31],temp31,'o-',linewidth=2,color='steelblue')

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])

plt.ylabel('$^oC$',fontsize = 14,color='steelblue')
ax1.tick_params('y', colors='steelblue')
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(22,31,0.1)))
plt.plot(tDorian,np.arange(22,31,0.1),'--k')
plt.title('Surface Temperature GOFS 3.1',fontsize=16,color='steelblue')
#plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

ax2 = ax1.twinx()
ax2.plot(time_best_track,wind_int,'.-k')
color_cat = ['deepskyblue','cyan','lemonchiffon','gold','orange',\
             'darkorange','red']

yticks = [34,73,95,110,129,156]
plt.yticks(yticks)

for tt,int in enumerate(wind_int):
    #print(tt,int)
    if int <= yticks[0]:
        print(int)
        ax2.plot(time_best_track[tt],wind_int[tt],'.',color=color_cat[0],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if np.logical_and(int > yticks[0],int<=yticks[1]):
        print(int)
        ax2.plot(time_best_track[tt],wind_int[tt],'.',color=color_cat[1],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if np.logical_and(int>yticks[1],int <= yticks[2]):
        ax2.plot(time_best_track[tt],wind_int[tt],'.',color=color_cat[2],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if np.logical_and(int>yticks[2],int <= yticks[3]):
        ax2.plot(time_best_track[tt],wind_int[tt],'.',color=color_cat[3],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if np.logical_and(int>yticks[3],int <= yticks[4]):
        ax2.plot(time_best_track[tt],wind_int[tt],'.',color=color_cat[4],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if np.logical_and(int>yticks[4],int <= yticks[5]):
        ax2.plot(time_best_track[tt],wind_int[tt],'.',color=color_cat[5],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if int > yticks[5]:
        ax2.plot(time_best_track[tt],wind_int[tt],'.',color=color_cat[6],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
       
ax2.set_ylabel('Wind Speed (mph)',color = 'black',fontsize=14)
ax2.tick_params('y', colors='black')

ts = np.tile(34,len(xticks))
plt.plot(xticks,ts,'--',color='silver',zorder=0)
ts = np.tile(73,len(xticks))
plt.plot(xticks,ts,'--',color='silver',zorder=0)
plt.text(datetime(2019,8,25,6),yticks[1]-10,'Ts',fontsize=12)
ts = np.tile(95,len(xticks))
plt.plot(xticks,ts,'--',color='silver',zorder=0)
plt.text(datetime(2019,8,25,6),yticks[2]-10,'Cat 1',fontsize=12)
ts = np.tile(110,len(xticks))
plt.plot(xticks,ts,'--',color='silver',zorder=0)
plt.text(datetime(2019,8,25,6),yticks[3]-10,'Cat 2',fontsize=12)
ts = np.tile(129,len(xticks))
plt.plot(xticks,ts,'--',color='silver',zorder=0)
plt.text(datetime(2019,8,25,6),yticks[4]-10,'Cat 3',fontsize=12)
ts = np.tile(156,len(xticks))
plt.plot(xticks,ts,'--',color='silver',zorder=0)
plt.text(datetime(2019,8,25,6),yticks[5]-15,'Cat 4',fontsize=12)
plt.text(datetime(2019,8,25,6),yticks[5]+15,'Cat 5',fontsize=12)

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)
plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])

file = folder + ' ' + 'surf_temp_GOFS31_Dorian_cat'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#rect = patches.Rectangle((xticks[0],yticks[0]),\
#                             xticks[-1]-xticks[0],yticks[1]-yticks[0],\
#                             linewidth=1,edgecolor='k',facecolor='cyan',\
#                             alpha=0.2,linestyle='--')
#ax2.add_patch(rect)

#%% Time series at point underneath Dorian when cat 5 
# With Dorian distance to point of max cooling in GOFS

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = df.water_temp[oktime31,0,indy,indx]

distt = []
for pos,lat in enumerate(lat_best_track):
    vec_y = [lat_best_track[pos],27.0]
    vec_x = [lon_best_track[pos],-77.4]   
    distt.append(seawater.dist(vec_y,vec_x,units='km')[0][0])
    
dist_point_max_cool = np.asarray(distt)#/1.609

fig,ax1 = plt.subplots(figsize=(12, 4))

plt.plot(time31[oktime31],temp31,'o-',linewidth=2,color='steelblue')

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])

plt.ylabel('$^oC$',fontsize = 14,color='steelblue')
ax1.tick_params('y', colors='steelblue')
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(22,31,0.1)))
plt.plot(tDorian,np.arange(22,31,0.1),'--k')
plt.title('Surface Temperature GOFS 3.1',fontsize=16,color='steelblue')
#plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

ax2 = ax1.twinx()
ax2.plot(time_best_track,dist_point_max_cool,'.-k')

color_cat = ['deepskyblue','cyan','lemonchiffon','gold','orange',\
             'darkorange','red']

for tt,ca in enumerate(cat):
    if ca == 'td':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[0],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'ts':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[1],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat1':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[2],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat2':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[3],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat3':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[4],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat4':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[5],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat5':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[6],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'ex':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color='yellowgreen',\
                 markersize=20,markeredgecolor='yellowgreen',markeredgewidth=2)
       
ax2.set_ylabel('Distance (Km)',color = 'black',fontsize=14)
ax2.tick_params('y', colors='black')

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)
plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])
plt.grid(True)

file = folder + ' ' + 'surf_temp_GOFS31_Dorian_dist_point_max_cool'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Detail Time series at point underneath Dorian when cat 5
# With Dorian distance to point of max cooling in GOFS

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = df.water_temp[oktime31,0,indy,indx]

distt = []
for pos,lat in enumerate(lat_best_track):
    vec_y = [lat_best_track[pos],27.0]
    vec_x = [lon_best_track[pos],-77.4]   
    distt.append(seawater.dist(vec_y,vec_x,units='km')[0][0])
    
dist_point_max_cool = np.asarray(distt)#/1.609

fig,ax1 = plt.subplots(figsize=(4, 4))

plt.plot(time31[oktime31],temp31,'o-',linewidth=2,color='steelblue')

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])

plt.ylabel('$^oC$',fontsize = 14,color='steelblue')
ax1.tick_params('y', colors='steelblue')
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(22,31,0.1)))
plt.plot(tDorian,np.arange(22,31,0.1),'--k')
plt.title('Surface Temperature GOFS 3.1',fontsize=16,color='steelblue')
#plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

ax2 = ax1.twinx()
ax2.plot(time_best_track,dist_point_max_cool,'.-k')

color_cat = ['deepskyblue','cyan','lemonchiffon','gold','orange',\
             'darkorange','red']

for tt,ca in enumerate(cat):
    if ca == 'td':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[0],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'ts':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[1],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat1':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[2],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat2':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[3],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat3':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[4],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat4':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[5],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'cat5':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color=color_cat[6],\
                 markersize=20,markeredgecolor='k',markeredgewidth=2)
    if ca == 'ex':
        ax2.plot(time_best_track[tt],dist_point_max_cool[tt],'.',color='yellowgreen',\
                 markersize=20,markeredgecolor='yellowgreen',markeredgewidth=2)
       
ax2.set_ylabel('Distance (Km)',color = 'black',fontsize=14)
ax2.tick_params('y', colors='black')

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)
#plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])
plt.xlim([datetime(2019,9,1),datetime(2019,9,4)])
plt.ylim([0,200])
plt.grid(True)

file = folder + ' ' + 'surf_temp_GOFS31_Dorian_dist_point_max_cool_detail'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% temperature Time series following Dorian track

okt = np.where(np.logical_and(time_best_track >= datetime(2019,8,30,18),\
                              time_best_track <= datetime(2019,9,4,18)))[0]

temp31_prof_track = np.empty([len(depth31),len(okt)])
temp31_prof_track[:] = np.nan
surf_temp31_track = []
for t,ind in enumerate(okt):
    oklon = int(np.round(np.interp(lon_best_track[ind]+360,df.lon,np.arange(len(df.lon)))))
    oklat = int(np.round(np.interp(lat_best_track[ind],df.lat,np.arange(len(df.lat)))))
    oktt = np.where(time31 == mdates.date2num(time_best_track[ind]))[0][0]

    temp31_prof_track[:,t] = np.asarray(df.water_temp[oktt,:,oklat,oklon]).T
    surf_temp31_track.append(np.asarray(df.water_temp[oktt,0,oklat,oklon]))

surf_temp31_track = np.asarray(surf_temp31_track)

color_map = cmocean.cm.thermal
maxd = 300      
okm = depth31 <= maxd

'''
min_val = np.floor(np.nanmin(temp31[okm]))
max_val = np.ceil(np.nanmax(temp31[okm]))    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
'''
   
kw = dict(levels = np.linspace(17,31,15)) 
# plot
fig, ax = plt.subplots(figsize=(12, 3))
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(time_best_track[okt],-depth31,temp31_prof_track,cmap=color_map,**kw)
plt.contour(time_best_track[okt],-depth31,temp31_prof_track,[26],colors='k')
#plt.plot(time_best_track[okt],-MLD31,'.-',color='grey')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
'''
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
'''
#ax.set_xlim(datetime(2019,8,25),datetime(2019,9,8))
ax.set_ylim(-300, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
#tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(-maxd,0)))
#plt.plot(tDorian,np.arange(-maxd,0),'--k')
plt.title('Temperature' + ' Profile ' + 'GOFS 3.1')  

file = folder + ' ' + 'temp_profile_following_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Time series temper1ature following Dorian track

indx = np.where(df.lon > -77.4+360)[0][0]
indy = np.where(df.lat > 27.0)[0][0]

temp31 = df.water_temp[oktime31,0,indy,indx]

fig,ax1 = plt.subplots(figsize=(12, 4))

plt.plot(time_best_track[okt],surf_temp31_track,'o-',linewidth=2,color='steelblue')

#t0 = datetime(2019,8,25)
#deltat= timedelta(1)
#xticks = [t0+nday*deltat for nday in np.arange(15)]
#xticks = np.asarray(xticks)
#plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
#plt.xlim([time31[oktime31[0]],time31[oktime31[-1]]])

plt.ylabel('$^oC$',fontsize = 14,color='steelblue')
ax1.tick_params('y', colors='steelblue')
#tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(22,31,0.1)))
#plt.plot(tDorian,np.arange(22,31,0.1),'--k')
plt.title('Surface Temperature GOFS 3.1 along Dorian Track',fontsize=16,color='steelblue')
#plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

#%% Reading glider data
    
url_glider = gdata_ng665
#url_glider = gdata_ng666
#url_glider = gdata_ng668
#url_glider = gdata_silbo

#del depthg_gridded, tempg_gridded, saltg_gridded, densg_gridded

var = 'temperature'
#Time window
#date_ini = '2019/08/25/00'
#date_end = '2019/09/08/00'
scatter_plot = 'no'
kwargs = dict(date_ini=date_ini[0:-3],date_end=date_end[0:-3])

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
#tempg = np.asarray(varg.T)
tempg = varg  

var = 'salinity'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)         
             
#saltg = np.asarray(varg.T)  
saltg = varg
 
var = 'density'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
#densg = np.asarray(varg.T)
#depthg = np.asarray(depthg.T)                
             
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

#%% Calculate density for GOFS 3.1 

target_dens31 = sw.dens(target_salt31,target_temp31,np.tile(depth31,(len(time31),1)).T)

#%% time of Dorian passing closets to glider
    
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0))) #ng665, ng666 
#tDorian = np.tile(datetime(2019,8,28,6),len(np.arange(-1000,0))) #ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(-1000,0))) #silbo    
    
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

plt.plot(tDorian,np.arange(-1000,0),'--k')
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
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + 'GOFS 3.1')  

file = folder + ' ' + 'along_track_temp ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the 
# mixed layer and td: temp at 1 meter below the mixed layer            
# for glider data

d10 = np.where(depthg_gridded >= 10)[0][0]
dt = 0.2

MLD_dt = np.empty(len(timeg)) 
MLD_dt[:] = np.nan
Tmean_dtemp = np.empty(len(timeg)) 
Tmean_dtemp[:] = np.nan
Smean_dtemp = np.empty(len(timeg)) 
Smean_dtemp[:] = np.nan
Td = np.empty(len(timeg)) 
Td[:] = np.nan
for t,tt in enumerate(timeg):
    T10 = tempg_gridded[d10,t]
    delta_T = T10 - tempg_gridded[:,t] 
    ok_mld = np.where(delta_T <= dt)[0]    
    if ok_mld.size == 0:
        MLD_dt[t] = np.nan
        Tmean_dtemp[t] = np.nan
        Smean_dtemp[t] = np.nan
        Td[t] = np.nan
    else:
        ok_mld_plus1m = np.where(depthg_gridded >= depthg_gridded[ok_mld[-1]] + 1)[0][0]
        MLD_dt[t] = depthg_gridded[ok_mld[-1]]
        Tmean_dtemp[t] = np.nanmean(tempg_gridded[ok_mld,t])
        Smean_dtemp[t] = np.nanmean(saltg_gridded[ok_mld,t])
        Td[t] = tempg_gridded[ok_mld_plus1m,t]
        
#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the 
# mixed layer and td: temp at 1 meter below the mixed layer
# for GOFS 3.1 output            

d10 = np.where(depth31 >= 10)[0][0]
dt = 0.2

MLD_dt31 = np.empty(len(time31)) 
MLD_dt31[:] = np.nan
Tmean_dtemp31 = np.empty(len(time31)) 
Tmean_dtemp31[:] = np.nan
Smean_dtemp31 = np.empty(len(time31)) 
Smean_dtemp31[:] = np.nan
Td31 = np.empty(len(time31)) 
Td31[:] = np.nan
for t,tt in enumerate(time31):
    T10 = target_temp31[d10,t]
    delta_T = T10 - target_temp31[:,t] 
    ok_mld = np.where(delta_T <= dt)[0]    
    if ok_mld.size == 0:
        MLD_dt31[t] = np.nan
        Tmean_dtemp31[t] = np.nan
        Smean_dtemp31[t] = np.nan
        Td31[t] = np.nan
    else:
        ok_mld_plus1m = np.where(depth31 >= depth31[ok_mld[-1]] + 1)[0][0]
        MLD_dt31[t] = depth31[ok_mld[-1]]
        Tmean_dtemp31[t] = np.nanmean(target_temp31[ok_mld,t])
        Smean_dtemp31[t] = np.nanmean(target_salt31[ok_mld,t])
        Td31[t] = target_temp31[ok_mld_plus1m,t]        
        
#%%  Calculation of mixed layer depth based on drho
# for glider data        
        
d10 = np.where(depthg_gridded >= 10)[0][0]
drho = 0.125

MLD_drho = np.empty(len(timeg)) 
MLD_drho[:] = np.nan
Tmean_drho = np.empty(len(timeg)) 
Tmean_drho[:] = np.nan
Smean_drho = np.empty(len(timeg)) 
Smean_drho[:] = np.nan
for t,tt in enumerate(timeg):
    rho10 = densg_gridded[d10,t]
    delta_rho = -(rho10 - densg_gridded[:,t]) 
    ok_mld = np.where(delta_rho <= drho)
    if ok_mld[0].size == 0:
        MLD_drho[t] = np.nan
        Tmean_drho[t] = np.nan
        Smean_drho[t] = np.nan
    else:
        MLD_drho[t] = depthg_gridded[ok_mld[0][-1]] 
        Tmean_drho[t] = np.nanmean(tempg_gridded[ok_mld,t]) 
        Smean_drho[t] = np.nanmean(saltg_gridded[ok_mld,t]) 
        
#%%  Calculation of mixed layer depth based on drho
# for GOFS 3.1 output        
        
d10 = np.where(depth31 >= 10)[0][0]
drho = 0.125

MLD_drho31 = np.empty(len(time31)) 
MLD_drho31[:] = np.nan
Tmean_drho31 = np.empty(len(time31)) 
Tmean_drho31[:] = np.nan
Smean_drho31 = np.empty(len(time31)) 
Smean_drho31[:] = np.nan
for t,tt in enumerate(time31):
    rho10 = target_dens31[d10,t]
    delta_rho31 = -(rho10 - target_dens31[:,t]) 
    ok_mld = np.where(delta_rho31 <= drho)
    if ok_mld[0].size == 0:
        MLD_drho31[t] = np.nan
        Tmean_drho31[t] = np.nan
        Smean_drho31[t] = np.nan
    else:
        MLD_drho31[t] = depth31[ok_mld[0][-1]] 
        Tmean_drho31[t] = np.nanmean(target_temp31[ok_mld,t]) 
        Smean_drho31[t] = np.nanmean(target_salt31[ok_mld,t]) 
        
#%% Tmean and Td
        
Tmean_dtemp_pom_oper = np.array([28.4089653 , 28.36802826, 28.32124467, 28.36863823, 28.3147935 ,
       28.28294667, 28.27456093, 28.30914376, 28.31633434, 28.26532156,
       28.26961396, 28.39389628, 28.42460771, 28.38040543, 28.31607368,
       28.38089908, 28.43803444, 28.43900433, 28.41603296, 28.42313125,
       28.41012833])
    
Tmean_drho_pom_oper = np.array([28.4390192 , 28.372166  , 28.33453342, 28.37648419, 28.31623023,
       28.28398252, 28.27493429, 28.34033421, 28.3637139 , 28.2818954 ,
       28.27561273, 28.41571469, 28.48051453, 28.42306868, 28.35106564,
       28.46577517, 28.56110115, 28.5258316 , 28.49490395, 28.55054436,
       28.55966415])
    
Tmean_dtemp_pom_exp = np.array([29.28252983, 29.22755661, 29.16900558, 29.17637482, 29.10806592,
       29.04672941, 28.9209938 , 28.88021628, 28.87556966, 28.91069762,
       28.94633516, 28.99495284, 28.95316772, 28.86668434, 28.83317539,
       28.89052909, 28.8785677 , 28.70841135, 28.54443632, 28.56767137,
       28.61995179])
    
Tmean_drho_pom_exp = np.array([29.31585932, 29.25217724, 29.19769192, 29.19869423, 29.1336834 ,
       29.05274773, 28.92811279, 28.89937363, 28.89851618, 28.9182476 ,
       28.96085472, 29.01779938, 28.97569799, 28.87948418, 28.8425045 ,
       28.93228607, 28.90347824, 28.73153114, 28.5518713 , 28.61049271,
       28.66873217])
    
# time POM
t0 = datetime(2019,8,28)
time_pom = [t0 + timedelta(hours=int(hrs)) for hrs in np.arange(0,126,6)]
        
fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg[0],Tmean_dtemp[0],'o-',color='royalblue',label=inst_id.split('-')[0],linewidth=3)
plt.plot(timeg,Tmean_drho,'s-',color='royalblue',linewidth=3)
plt.plot(time31,Tmean_dtemp31,'--o',color='indianred',label='GOFS 3.1')
plt.plot(time31,Tmean_drho31,'--s',color='indianred')
plt.plot(time_pom,Tmean_dtemp_pom_oper,'-o',color='seagreen',label='POM Oper')
plt.plot(time_pom,Tmean_drho_pom_oper,'-s',color='seagreen')
plt.plot(time_pom,Tmean_dtemp_pom_exp,'-o',color='darkorchid',label='POM Exp')
plt.plot(time_pom,Tmean_drho_pom_exp,'-s',color='darkorchid')
plt.plot(timeg[0],Tmean_dtemp[0],'o',color='k',label='Temperature criteria',linewidth=3)
plt.plot(timeg[0],Tmean_drho[0],'s',color='k',label='Density Criteria',linewidth=3)
#plt.plot(timeg_low,Tmean_low,'-',color='grey',label='12 hours lowpass')
#plt.plot(timeg,Td,'.-g',label='Td')
#plt.plot(timeg[np.isfinite(Td)],Td_low,'-',color='lightseagreen',label='12 hours lowpass')
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])
plt.ylabel('$^oC$',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(28.2,29.4,0.01)))# ng665,ng666
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.0,29.30,0.01)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.2,29.7,0.01)))  # ng668
plt.plot(tDorian,np.arange(28.2,29.4,0.01),'--k')
plt.title('Mixed Layer Temperature',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))
plt.xlim([time_pom[0],time_pom[-1]])

file = folder + ' ' + inst_id + '_Tmean_Td'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Tmean and Td and wind speed
        
fig,ax1 = plt.subplots(figsize=(12, 4))
plt.plot(timeg,Tmean_dtemp,'-',color='indianred',label='Tmean mixed layer, temp criteria',linewidth=3)
plt.plot(timeg,Tmean_drho,'-',color='seagreen',label='Tmean mixed layer, dens criteria',linewidth=3)
plt.plot(time31,Tmean_dtemp31,'--o',color='lightcoral',label='Tmean mixed layer GOFS 3.1, temp criteria')
plt.plot(time31,Tmean_drho31,'--o',color='lightgreen',label='Tmean mixed layer GOFS 3.1, dens criteria')
#plt.plot(timeg_low,Tmean_low,'-',color='grey',label='12 hours lowpass')
#plt.plot(timeg,Td,'.-g',label='Td')
#plt.plot(timeg[np.isfinite(Td)],Td_low,'-',color='lightseagreen',label='12 hours lowpass')
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])
plt.ylabel('$^oC$',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(29.0,29.30,0.01)))# ng665,ng666
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.0,29.30,0.01)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.2,29.7,0.01)))  # ng668
plt.plot(tDorian,np.arange(29.0,29.30,0.01),'--k')
plt.title(inst_id.split('T')[0],fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

ax2 = ax1.twinx()
ax2.plot(time_NDBC,wspd_NDBC,'.-',color='steelblue',alpha=0.5)
ax2.set_ylabel('Wind Speed (m/s)',color = 'steelblue',fontsize=14)
ax2.tick_params('y', colors='steelblue')
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + inst_id + '_Tmean_Td_wspd'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Smean
        
fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg,Smean_dtemp,'-',color='indianred',label='Smean mixed layer, temp criteria',linewidth=3)
plt.plot(timeg,Smean_drho,'-',color='seagreen',label='Smean mixed layer, dens criteria',linewidth=3)
plt.plot(time31,Smean_dtemp31,'--o',color='lightcoral',label='Smean mixed layer GOFS 3.1, temp criteria')
plt.plot(time31,Smean_drho31,'--o',color='lightgreen',label='Smean mixed layer GOFS 3.1, dens criteria')
#plt.plot(timeg_low,Tmean_low,'-',color='grey',label='12 hours lowpass')
#plt.plot(timeg,Td,'.-g',label='Td')
#plt.plot(timeg[np.isfinite(Td)],Td_low,'-',color='lightseagreen',label='12 hours lowpass')
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])
#plt.ylabel('$^oC$',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(35.5,36.5,0.1)))# ng665
#tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(36.0,36.6,0.1)))# ng665,ng666
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(34.0,36.0,0.1)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.2,29.7,0.01)))  # ng668
plt.plot(tDorian,np.arange(35.5,36.5,0.1),'--k')
plt.title(inst_id.split('T')[0],fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

file = folder + ' ' + inst_id + '_Smean'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Smean and daily rain daily accumulation 
        
fig,ax1 = plt.subplots(figsize=(12, 4))
plt.plot(timeg,Smean_dtemp,'-',color='indianred',label='Smean mixed layer, temp criteria',linewidth=3)
plt.plot(timeg,Smean_drho,'-',color='seagreen',label='Smean mixed layer, dens criteria',linewidth=3)
plt.plot(time31,Smean_dtemp31,'--o',color='lightcoral',label='Smean mixed layer GOFS 3.1, temp criteria')
plt.plot(time31,Smean_drho31,'--o',color='lightgreen',label='Smean mixed layer GOFS 3.1, dens criteria')
#plt.plot(timeg_low,Tmean_low,'-',color='grey',label='12 hours lowpass')
#plt.plot(timeg,Td,'.-g',label='Td')
#plt.plot(timeg[np.isfinite(Td)],Td_low,'-',color='lightseagreen',label='12 hours lowpass')
t0 = datetime(2019,9,17)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])
#plt.ylabel('$^oC$',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(35.5,36.5,0.1)))
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.0,29.30,0.01)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.2,29.7,0.01)))  # ng668
plt.plot(tDorian,np.arange(35.5,36.5,0.1),'--k')
plt.title(inst_id.split('T')[0],fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

ax2 = ax1.twinx()
ax2.plot(time_accum,dayl_accum,'o-',color='steelblue')
ax2.set_ylabel('Daily Accumulation (in)',color = 'steelblue',fontsize=14)
ax2.tick_params('y', colors='steelblue')
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + inst_id + '_Smean_precip1'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Smean and daily rain daily accumulation 
        
fig,ax1 = plt.subplots(figsize=(12, 4))
plt.plot(timeg,Smean_dtemp,'-',color='indianred',label='Smean mixed layer, temp criteria',linewidth=3)
plt.plot(timeg,Smean_drho,'-',color='seagreen',label='Smean mixed layer, dens criteria',linewidth=3)
plt.plot(time31,Smean_dtemp31,'--o',color='lightcoral',label='Smean mixed layer GOFS 3.1, temp criteria')
plt.plot(time31,Smean_drho31,'--o',color='lightgreen',label='Smean mixed layer GOFS 3.1, dens criteria')
#plt.plot(timeg_low,Tmean_low,'-',color='grey',label='12 hours lowpass')
#plt.plot(timeg,Td,'.-g',label='Td')
#plt.plot(timeg[np.isfinite(Td)],Td_low,'-',color='lightseagreen',label='12 hours lowpass')
t0 = datetime(2019,9,17)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])
#plt.ylabel('$^oC$',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(35.5,36.5,0.1)))
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.0,29.30,0.01)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.2,29.7,0.01)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.0,29.30,0.01)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.2,29.7,0.01)))  # ng668
plt.plot(tDorian,np.arange(35.5,36.5,0.1),'--k')
plt.title(inst_id.split('T')[0],fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

ax2 = ax1.twinx()
#ax2.plot(time_hourly_precip,hourly_precip,'o-',color='steelblue')
#ax2.set_ylabel('Hourly Precipitation (in)',color = 'steelblue',fontsize=14)
ax2.plot(mdates.date2num(time_accum_precip),accum_precip,'o-',color='steelblue')
#ax2.plot(time_accum,dayl_accum,'o-',color='darkcyan')
ax2.set_ylabel('6 hourly Precipitation (in)',color = 'steelblue',fontsize=14)
ax2.tick_params('y', colors='steelblue')
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + inst_id + '_Smean_precip2'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   


#%% mld 

MLD_dt_pom_oper = np.array([-36.35997355, -36.35997355, -43.63196736, -36.35997355,
       -43.63196736, -43.63196736, -43.63196736, -43.63196736,
       -36.35997355, -43.63196736, -43.63196736, -43.63196736,
       -43.63196736, -36.35997355, -43.63196736, -43.63196736,
       -36.35997355, -36.35997355, -43.63196736, -43.63196736,
       -43.63196736])
    
MLD_drho_pom_oper = np.array([-15.64580736, -15.64580736, -19.83271244, -19.83271244,
       -19.83271244, -24.68070981, -24.68070981, -19.83271244,
       -15.64580736, -24.68070981, -30.1897961 , -30.1897961 ,
       -19.83271244, -15.64580736, -15.64580736, -15.64580736,
       -11.89962746, -11.89962746, -11.89962746, -11.89962746,
       -11.89962746])
    
MLD_dt_pom_exp = np.array([-26.99999884, -26.99999884, -26.99999884, -26.99999884,
       -35.50000046, -35.50000046, -35.50000046, -35.50000046,
       -35.50000046, -35.50000046, -35.50000046, -35.50000046,
       -26.99999884, -26.99999884, -44.99999806, -44.99999806,
       -35.50000046, -44.99999806, -44.99999806, -44.99999806,
       -44.99999806])
    
MLD_drho_pom_exp = np.array([-19.49999959, -19.49999959, -19.49999959, -19.49999959,
       -26.99999884, -26.99999884, -26.99999884, -26.99999884,
       -19.49999959, -26.99999884, -26.99999884, -26.99999884,
       -19.49999959, -19.49999959, -26.99999884, -26.99999884,
       -26.99999884, -26.99999884, -19.49999959, -19.49999959,
       -19.49999959])
    
# time POM
t0 = datetime(2019,8,28)
time_pom = [t0 + timedelta(hours=int(hrs)) for hrs in np.arange(0,126,6)]
        
fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg,MLD_dt,'-o',color='royalblue',label=inst_id.split('-')[0],linewidth=3)
plt.plot(timeg,MLD_drho,'-s',color='royalblue',linewidth=3)
plt.plot(time31,MLD_dt31,'--o',color='indianred',label='GOFS 3.1')
plt.plot(time31,MLD_drho31,'--s',color='indianred')
plt.plot(time_pom,-MLD_dt_pom_oper,'-o',color='seagreen',label='POM Oper')
plt.plot(time_pom,-MLD_drho_pom_oper,'-s',color='seagreen')
plt.plot(time_pom,-MLD_dt_pom_exp,'-o',color='darkorchid',label='POM Exp')
plt.plot(time_pom,-MLD_drho_pom_exp,'-s',color='darkorchid')
plt.plot(timeg[0],MLD_dt[0],'o',color='k',label='Temperature Criteria',linewidth=3)
plt.plot(timeg[0],MLD_drho[0],'s',color='k',label='Density Criteria',linewidth=3)
#plt.plot(timeg_low,Tmean_low,'-',color='grey',label='12 hours lowpass')
#plt.plot(timeg,Td,'.-g',label='Td')
#plt.plot(timeg[np.isfinite(Td)],Td_low,'-',color='lightseagreen',label='12 hours lowpass')
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([time_pom[0],time_pom[-1]])
plt.ylabel('Depth (m)',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(10,50)))# ng665,ng666
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.0,29.30,0.01)))  # ng668
#tDorian = np.tile(datetime(2019,8,29,6),len(np.arange(29.2,29.7,0.01)))  # ng668
plt.plot(tDorian,np.arange(10,50),'--k')
plt.title('Mixed Layer Depth',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder + ' ' + inst_id + '_mld_dt_drho'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Top 200 m temperature

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
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='indianred',linewidth=2 )
plt.plot(timeg,-MLD_drho,'-',label='MLD drho',color='seagreen',linewidth=2 )

cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticklabels(' ')
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0))) #ng665
#tDorian = np.tile(datetime(2019,8,28,6),len(np.arange(-1000,0)))
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + inst_id)
plt.legend()   

ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_temp31,cmap=color_map,**kw)
plt.contour(mdates.date2num(time31),-depth31,target_temp31,[26],colors='k')
plt.plot(time31,-MLD_dt31,'--.',label='MLD dt',color='indianred' )
plt.plot(time31,-MLD_drho31,'--.',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + 'GOFS 3.1')  
plt.legend()   

file = folder + ' ' + 'along_track_temp_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Top 200 m glider temperature from 2019/08/28/00

color_map = cmocean.cm.thermal
       
okg = depthg_gridded <= 200
okm = depth31 <= 200 
min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp31[okm])]))
max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp31[okm])]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    

# plot
fig, ax = plt.subplots(figsize=(12, 2))
      
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=color_map,**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded,[26],colors='k')
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='indianred',linewidth=2 )
plt.plot(timeg,-MLD_drho,'-',label='MLD drho',color='seagreen',linewidth=2 )

cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
#ax.set_xticklabels(' ')
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0))) #ng665
#tDorian = np.tile(datetime(2019,8,28,6),len(np.arange(-1000,0)))
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + inst_id)
plt.legend()  
ax.set_xlim(datetime(2019,8,28), datetime(2019,9,2)) 

file = folder + ' ' + 'along_track_temp_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Top 200 m GOFS 3.1 temperature from 2019/08/28/00

color_map = cmocean.cm.thermal
       
okg = depthg_gridded <= 200
okm = depth31 <= 200 
min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp31[okm])]))
max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp31[okm])]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))

fig, ax = plt.subplots(figsize=(12, 2))     
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_temp31,cmap=color_map,**kw)
plt.contour(mdates.date2num(time31),-depth31,target_temp31,[26],colors='k')
plt.plot(time31,-MLD_dt31,'--.',label='MLD dt',color='indianred' )
plt.plot(time31,-MLD_drho31,'--.',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + 'GOFS 3.1')  
plt.legend()  
ax.set_xlim(datetime(2019,8,28), datetime(2019,9,2))  

file = folder + ' ' + 'along_track_temp_top200_GOFS_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Vertican profile sg665, GOFS 3.1, POM

tDorian = datetime(2019,8,28,18)
okg = np.where(timeg < tDorian)[0][-1]
ok31 = np.where(time31 == tDorian )[0][0] #tdorian

temp_prof_pom_oper = np.array([28.40577126, 28.39975929, 28.39424515, 28.37770271, 28.34392929,
       28.3406601 , 28.37332153, 28.38916588, 28.36230278, 28.29952431,
       28.1592865 , 27.86792564, 27.4238739 , 26.77181053, 25.9107666 ,
       24.8875103 , 23.86647797, 22.95853615, 22.22094536, 21.50520134,
       20.73034096, 19.815382  , 18.74596214, 17.55633545, 16.22701836,
       14.7481432 , 13.208251  , 11.61562538, 10.006423  ,  8.48377323,
        7.16531754,  6.10278702,  5.30196524,  4.82748652,  4.52641249,
        4.12195826,  3.68905044,  3.10000157,  1.76830471,      np.nan])
    
depth_prof_pom_oper = np.array([-1.10181741e+00, -3.30545217e+00, -5.72945057e+00, -8.59417558e+00,
       -1.18996275e+01, -1.56458074e+01, -1.98327124e+01, -2.46807098e+01,
       -3.01897961e+01, -3.63599736e+01, -4.36319674e+01, -5.20057798e+01,
       -6.12610471e+01, -7.18384967e+01, -8.39584857e+01, -9.76210229e+01,
       -1.13266822e+02, -1.31336634e+02, -1.51830431e+02, -1.75188963e+02,
       -2.01852944e+02, -2.32042745e+02, -2.66639811e+02, -3.06084865e+02,
       -3.50818667e+02, -4.01942996e+02, -4.60339335e+02, -5.26889057e+02,
       -6.02694124e+02, -6.89076613e+02, -7.87579081e+02, -8.99964494e+02,
       -1.02821600e+03, -1.17409658e+03, -1.34047099e+03, -1.53020403e+03,
       -1.74638066e+03, -1.99296733e+03, -2.27415112e+03, -2.42399829e+03])
    
temp_prof_pom_exp = np.array([29.19390869, 29.19709778, 29.19932365, 29.20444679, 29.08709717,
       28.76905441, 28.17053032, 27.37616539, 26.56984138, 25.8953495 ,
       25.28985214, 24.07573128, 23.02121162, 21.90457916, 20.6519928 ,
       19.54855728, 18.48551941, 17.23768616, 15.703269  , 13.64709091,
       11.16824913,  8.85765171,  7.64004326,  6.96062756,  6.01803637,
        5.20464754,  4.70668983,  4.45792246,  4.34634113,  4.21899176,
        4.00993824,  3.82045794,  3.47413278,  3.18805742,  3.02687716,
        2.72020507,  2.32690835,  1.92075431,  1.18043661,         np.nan])
    
depth_prof_pom_exp = np.array([-2.50000002e+00, -7.49999989e+00, -1.30000002e+01, -1.94999996e+01,
       -2.69999988e+01, -3.55000005e+01, -4.49999981e+01, -5.59999999e+01,
       -6.84999982e+01, -8.24999982e+01, -9.89999957e+01, -1.17999996e+02,
       -1.38999999e+02, -1.63000004e+02, -1.90499999e+02, -2.21500002e+02,
       -2.56999984e+02, -2.97999999e+02, -3.44499983e+02, -3.97499990e+02,
       -4.57999989e+02, -5.26499998e+02, -6.04999997e+02, -6.94499977e+02,
       -7.96000011e+02, -9.12000015e+02, -1.04450005e+03, -1.19549994e+03,
       -1.36750001e+03, -1.56350002e+03, -1.78700000e+03, -2.04200009e+03,
       -2.33300000e+03, -2.66399989e+03, -3.04149985e+03, -3.47200003e+03,
       -3.96250015e+03, -4.52200001e+03, -5.16000000e+03, -5.50000000e+03])
    
plt.figure(figsize=(4,7))
plt.plot(tempg[:,okg],-depthg[:,okg],'-',color='royalblue',linewidth=4,label='sg665') 
plt.plot(target_temp31[:,ok31],-depth31,'o-',color='indianred',linewidth=2,label='GOFS 3.1')
plt.plot(temp_prof_pom_oper,depth_prof_pom_oper,'^-',color='seagreen',linewidth=2,label='POM Oper')
plt.plot(temp_prof_pom_exp,depth_prof_pom_exp,'s-',color='darkorchid',linewidth=2,label='POM Exp')
plt.ylim([-200,0])
plt.xlim([20,30])
plt.title('Temperature ($^oC$)',fontsize=16)
plt.ylabel('Depth (m)')
plt.legend()

file = folder + ' ' + 'temp_profile_sg665_GOFS_POM' + inst_id
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

#kw = dict(levels = np.linspace(35.5,37.3,19)) #ng665
kw = dict(levels = np.linspace(34.0,37.6,19)) #ng668
    
# plot
fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,saltg_gridded,cmap=color_map,**kw)
plt.contour(timeg,-depthg_gridded,saltg_gridded,[26],colors='k')
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='indianred',linewidth=2 )
plt.plot(timeg,-MLD_drho,'-*',label='MLD drho',color='seagreen',linewidth=2 )

cs = fig.colorbar(cs, orientation='vertical') 
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticklabels(' ')
#tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0)))
tDorian = np.tile(datetime(2019,8,28,6),len(np.arange(-1000,0)))
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + inst_id)
plt.legend()   

ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_salt31,cmap=color_map,**kw)
plt.contour(mdates.date2num(time31),-depth31,target_salt31,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
plt.plot(time31,-MLD_dt31,'--.',label='MLD dt',color='indianred' )
plt.plot(time31,-MLD_drho31,'--.',label='MLD drho',color='seagreen' )

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.legend()
plt.title('Along Track ' + 'Salinity' + ' Profile ' + 'GOFS 3.1')  

file = folder + ' ' + 'along_track_salt_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Density transect

fig,ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211) 
nlevels = np.round(np.nanmax(densg_gridded)) - np.round(np.nanmin(densg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(densg_gridded)),\
                               np.round(np.nanmax(densg_gridded)),11))
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,densg_gridded,cmap=cmocean.cm.dense,**kw)
plt.plot(timeg,-MLD_dt,'-',label='MLD dt',color='indianred',linewidth=2 )
plt.plot(timeg,-MLD_drho,'-',label='MLD drho',color='seagreen',linewidth=2 )
plt.xlim(timeg[0],timeg[-1])
plt.ylim([-200,0])
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticklabels(' ')
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('$kg/m^3$',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)  
plt.title('Density Transect ' + inst_id,fontsize=18) 
plt.legend() 
#tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0)))
plt.plot(tDorian,np.arange(-1000,0),'--k')

ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_dens31,cmap=cmocean.cm.dense,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
plt.plot(time31,-MLD_dt31,'--.',label='MLD dt',color='indianred' )
plt.plot(time31,-MLD_drho31,'--.',label='MLD drho',color='seagreen' )

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xticks = [t0+nday*deltat for nday in np.arange(14)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(tDorian,np.arange(-1000,0),'--k')
plt.legend()
plt.title('Along Track ' + 'Salinity' + ' Profile ' + 'GOFS 3.1') 

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
#tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0)))
plt.plot(tDorian,np.arange(-1000,0),'--k')

file = folder + ' ' + inst_id + '_salt_200m'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
  
#%% Map of North Atlantic with glider tracks

# Look for datasets in IOOS glider dac

# Time bounds
min_time = '2019-08-24T00:00:00Z'
max_time = '2019-09-07T00:00:00Z'

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
        ax.plot(long,latg,'.-',color='darkorange',markersize=1)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Daily_map_North_Atlantic_gliders_in_DAC_' + min_time[0:10] + '_' + max_time[0:10] + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Map detail

# Time bounds
min_time = '2019-08-24T00:00:00Z'
max_time = '2019-09-07T00:00:00Z'

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

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 5))
#plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.yticks([])
plt.xticks([])
plt.axis([lon_lim[0],lon_lim[1],lat_lim[0],lat_lim[1]])
plt.title('Active Glider Deployments '+ min_time[0:10]+'-'+max_time[0:10] ,fontsize=20)
plt.plot(lon_best_track[14:-1],lat_best_track[14:-1],'or',markersize=6)
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

#%% Map of North Atlantic with glider tracks and ARGO floats

# Look for datasets in IOOS glider dac

# Time bounds
min_time = '2019-08-24T00:00:00Z'
max_time = '2019-09-07T00:00:00Z'

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

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.yticks([])
plt.xticks([])
plt.title('Active Glider Deployments '+ min_time[0:10]+'-'+max_time[0:10] ,fontsize=20)
plt.plot(lon_best_track[5:-2],lat_best_track[5:-2],'or',markersize=3)
ax.set_aspect(1)

lat_lim2 = [24,31]
lon_lim2 = [-81,-74]
rect = patches.Rectangle((lon_lim2[0],lat_lim2[0]),\
                             lon_lim2[1]-lon_lim2[0],lat_lim2[1]-lat_lim2[0],\
                             linewidth=2,edgecolor='k',facecolor='none')
ax.add_patch(rect)

for i,id in enumerate(gliders):
   e.dataset_id = id
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
   ax.plot(long,latg,'.-',color='darkorange',markersize=1)
        
for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:]
    argo_lon = ncargo.variables['LONGITUDE'][:]

    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

plt.xlim([lon_lim[0],lon_lim[1]])
plt.ylim([lat_lim[0],lat_lim[1]])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Daily_map_North_Atlantic_gliders_in_DAC_ARGO' + min_time[0:10] + '_' + max_time[0:10] + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Map of North Atlantic with glider tracks and ARGO floats detail

# Time bounds
min_time = '2019-08-24T00:00:00Z'
max_time = '2019-09-07T00:00:00Z'

lat_lim = [24,31]
lon_lim = [-81,-74]

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

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 5))
#plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.yticks([])
plt.xticks([])
plt.axis([lon_lim[0],lon_lim[1],lat_lim[0],lat_lim[1]])
#plt.title('Active Glider Deployments '+ min_time[0:10]+'-'+max_time[0:10] ,fontsize=20)
plt.plot(lon_best_track[14:-1],lat_best_track[14:-1],'or',markersize=6)
ax.set_aspect(1)

for id in gliders:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    df = e.to_pandas()
    if len(df) !=0:
        df = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            skiprows=(1,)  # units information can be dropped.
            ).dropna()
        
        print(len(df))
               
        timeg, ind = np.unique(df.index.values,return_index=True)
        latg = df['latitude (degrees_north)'].values[ind]
        long = df['longitude (degrees_east)'].values[ind]
        ax.plot(long,latg,'.-',color='darkorange',markersize=1)
        ax.plot(long[-1],latg[-1],'o',color='k',markersize=4)
        
        ax.text(long[-1],latg[-1],id.split('-')[0],weight='bold',
                bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))
        
for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:]
    argo_lon = ncargo.variables['LONGITUDE'][:]

    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

#plt.xlim([lon_lim[0],lon_lim[1]])
#plt.ylim([lat_lim[0],lat_lim[1]])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Daily_map_North_Atlantic_gliders_in_DAC_ARGO_detail' + min_time[0:10] + '_' + max_time[0:10] + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Find closest Argo floats to Dorian's path

lat_lim = [26,28]
lon_lim = [-78,-75]

l_Dorian = []
for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:]
    argo_lon = ncargo.variables['LONGITUDE'][:]
    
    oklatargo = np.logical_and(argo_lat >= lat_lim[0],argo_lat <= lat_lim[-1])
    oklonargo = np.logical_and(argo_lon >= lon_lim[0],argo_lon <= lon_lim[-1])
    argo_latsub = argo_lat[oklatargo]
    argo_lonsub = argo_lon[oklonargo]
    if np.logical_and(len(argo_latsub)!= 0,len(argo_lonsub)!= 0):
        l_Dorian.append(l)
        print(l)
        print(argo_lat)
        print(argo_lon)
        
#%%
l_Dorian = '/Volumes/aristizabal/ARGO_data/DataSelection_20191014_193816_8936308/argo-profiles-4902110.nc'
ncargo = Dataset(l_Dorian)

argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
argo_pres = np.asarray(ncargo.variables['PRES'][:])
argo_temp = np.asarray(ncargo.variables['TEMP'][:])
argo_psal = np.asarray(ncargo.variables['PSAL'][:])

argo_time = ncargo.variables['JULD']
argo_juld = ncargo.variables['JULD'][:]
argo_time = netCDF4.num2date(argo_juld,argo_time.units)

# Get rid off fill values
argo_pres[argo_pres == 99999.0] = np.nan
argo_temp[argo_temp == 99999.0] = np.nan
argo_psal[argo_psal == 99999.0] = np.nan

#%% Comparing Argo temperature with GOFS 3.1 and POM

# POM before from forecasting cycle 2019082512
tpom_bef = np.datetime64('2019-08-25T12:00:00.000000000')

z_matrix_pom_bef = np.array([[-6.91818186e-01, -2.07545452e+00, -3.59745459e+00,
        -5.39618170e+00, -7.47163604e+00, -9.82381831e+00,
        -1.24527267e+01, -1.54967272e+01, -1.89558177e+01,
        -2.28299995e+01, -2.73959988e+01, -3.26538171e+01,
        -3.84650905e+01, -4.51065467e+01, -5.27165451e+01,
        -6.12950914e+01, -7.11189048e+01, -8.24647269e+01,
        -9.53325407e+01, -1.09999088e+02, -1.26741088e+02,
        -1.45696908e+02, -1.67419999e+02, -1.92187085e+02,
        -2.20274912e+02, -2.52375277e+02, -2.89041651e+02,
        -3.30827438e+02, -3.78424547e+02, -4.32663096e+02,
        -4.94511636e+02, -5.65077114e+02, -6.45604728e+02,
        -7.37201423e+02, -8.41665959e+02, -9.60797100e+02,
        -1.09653186e+03, -1.25136073e+03, -1.42791273e+03,
        -1.52200000e+03]])
    
temp_pom_bef = np.array([28.486917 , 28.486572 , 28.485542 , 28.48426  , 28.483156 ,
       28.482365 , 28.479004 , 28.470148 , 28.447077 , 28.40281  ,
       28.315058 , 28.138271 , 27.850958 , 27.415342 , 26.794386 ,
       25.978376 , 25.024775 , 24.097149 , 23.349989 , 22.80228  ,
       22.40235  , 21.980844 , 21.439383 , 20.78513  , 20.065117 ,
       19.305069 , 18.516937 , 17.76371  , 17.039299 , 16.18957  ,
       15.114508 , 13.755231 , 11.903246 ,  9.670144 ,  7.437302 ,
        5.481609 ,  4.130128 ,  3.5499206,  3.624825 ,        np.nan])

salt_pom_bef = np.array([36.312824, 36.313595, 36.315495, 36.319057, 36.32393 , 36.33038 ,
       36.338696, 36.349174, 36.361782, 36.376316, 36.393936, 36.414734,
       36.438427, 36.466465, 36.498577, 36.534252, 36.573963, 36.61706 ,
       36.659332, 36.6948  , 36.717422, 36.726357, 36.72498 , 36.71915 ,
       36.707985, 36.689487, 36.66004 , 36.606373, 36.51512 , 36.370632,
       36.163357, 35.891327, 35.575665, 35.257458, 34.98247 , 34.794872,
       34.7132  , 34.72321 , 34.79677 ,       np.nan])
    
# POM after from forecasting cycle 2019090412
tpom_aft = np.datetime64('2019-09-04T12:00:00.000000000')    
    
z_matrix_pom_aft = np.array([[-4.55909066e-01, -1.36772717e+00, -2.37072716e+00,
        -3.55609062e+00, -4.92381767e+00, -6.47390878e+00,
        -8.20636278e+00, -1.02123630e+01, -1.24919080e+01,
        -1.50449987e+01, -1.80539981e+01, -2.15189071e+01,
        -2.53485437e+01, -2.97252717e+01, -3.47402704e+01,
        -4.03935433e+01, -4.68674488e+01, -5.43443601e+01,
        -6.28242658e+01, -7.24895392e+01, -8.35225384e+01,
        -9.60144483e+01, -1.10329993e+02, -1.26651534e+02,
        -1.45161448e+02, -1.66315629e+02, -1.90478816e+02,
        -2.18015703e+02, -2.49382259e+02, -2.85125531e+02,
        -3.25883798e+02, -3.72386538e+02, -4.25454339e+02,
        -4.85816677e+02, -5.54658939e+02, -6.33166513e+02,
        -7.22615893e+02, -8.24648316e+02, -9.40996306e+02,
        -1.00299994e+03]])

temp_pom_aft = np.array([26.549149 , 26.549137 , 26.54913  , 26.54912  , 26.54911  ,
       26.549103 , 26.549026 , 26.549122 , 26.5491   , 26.549147 ,
       26.5485   , 26.545742 , 26.53994  , 26.531277 , 26.51631  ,
       26.42541  , 26.073051 , 25.338528 , 24.333183 , 23.548782 ,
       22.882275 , 22.260403 , 21.876997 , 21.643265 , 21.311485 ,
       20.865627 , 20.185505 , 19.503124 , 18.905855 , 18.251097 ,
       17.552067 , 16.787643 , 15.946483 , 14.919392 , 13.65732  ,
       12.044389 , 10.214731 ,  8.642639 ,  7.4338336,        np.nan])

salt_pom_aft = np.array([36.415367, 36.41536 , 36.415394, 36.4154  , 36.415417, 36.41544 ,
       36.41546 , 36.4155  , 36.415543, 36.41559 , 36.415806, 36.41635 ,
       36.41732 , 36.419098, 36.42333 , 36.437866, 36.47229 , 36.5248  ,
       36.580963, 36.615986, 36.64782 , 36.68931 , 36.70738 , 36.72591 ,
       36.731716, 36.72427 , 36.71356 , 36.698833, 36.678135, 36.644382,
       36.580257, 36.480103, 36.329998, 36.127525, 35.885815, 35.636875,
       35.417183, 35.28746 , 35.310886,       np.nan])    

indx = np.where(df.lon > argo_lon[0]+360)[0][0]
indy = np.where(df.lat > argo_lat[0])[0][0]
indt = np.where(time31 < mdates.date2num(argo_time[0]))[0][-1]
temp31 = df.water_temp[indt,:,indy,indx]

plt.figure(figsize=(5,8))
plt.plot(argo_temp[0,:],-argo_pres[0,:],'.-',color='indianred',\
         linewidth=2,label='Argo '+str(argo_time[0])[0:16])
plt.plot(temp31,-depth31,'o-',color='lightcoral',linewidth=2,\
         label='GOFS '+str(mdates.num2date(time31[indt]))[0:16])
plt.plot(temp_pom_bef,z_matrix_pom_bef[0,:],'^-',color='salmon',linewidth=2,\
         label='POM '+str(tpom_bef)[0:16])

indx = np.where(df.lon > argo_lon[2]+360)[0][0]
indy = np.where(df.lat > argo_lat[2])[0][0]
indt = np.where(time31 > mdates.date2num(argo_time[2]))[0][0]
temp31 = df.water_temp[indt,:,indy,indx]

plt.plot(argo_temp[2,:],-argo_pres[2,:],'.-',color='royalblue',\
         linewidth=2,label='Argo '+str(argo_time[2])[0:16])
plt.plot(temp31,-depth31,'o-',color='skyblue',linewidth=2,\
         label='GOFS '+str(mdates.num2date(time31[indt]))[0:16])
plt.plot(temp_pom_aft,z_matrix_pom_aft[0,:],'^-',color='steelblue',linewidth=2,\
         label='POM '+str(tpom_aft)[0:16])
plt.ylim([-200,0])
plt.xlim([20,30])
plt.grid(True)
plt.legend()
plt.ylabel('Depth (m)',fontsize=16)
plt.xlabel('Temperature ($^oC$)',fontsize=16)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Argo_GOFS_POM_temp_profile_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Comparing Argo salinity with GOFS 3.1

indx = np.where(df.lon > argo_lon[0]+360)[0][0]
indy = np.where(df.lat > argo_lat[0])[0][0]
indt = np.where(time31 < mdates.date2num(argo_time[0]))[0][-1]
salt31 = df.salinity[indt,:,indy,indx]

plt.figure(figsize=(5,8))
plt.plot(argo_psal[0,:],-argo_pres[0,:],'.-',color='indianred',\
         linewidth=2,label='Argo '+str(argo_time[0])[0:16])
plt.plot(salt31,-depth31,'o-',color='lightcoral',linewidth=2,\
         label='GOFS '+str(mdates.num2date(time31[indt]))[0:16])
plt.plot(salt_pom_bef,z_matrix_pom_bef[0,:],'^-',color='salmon',linewidth=2,\
         label='POM '+str(tpom_bef)[0:16])

indx = np.where(df.lon > argo_lon[2]+360)[0][0]
indy = np.where(df.lat > argo_lat[2])[0][0]
indt = np.where(time31 > mdates.date2num(argo_time[2]))[0][0]
salt31 = df.salinity[indt,:,indy,indx]

plt.plot(argo_psal[2,:],-argo_pres[2,:],'.-',color='royalblue',\
         linewidth=2,label='Argo '+str(argo_time[2])[0:16])
plt.plot(salt31,-depth31,'o-',color='skyblue',linewidth=2,\
         label='GOFS '+str(mdates.num2date(time31[indt]))[0:16])
plt.plot(salt_pom_aft,z_matrix_pom_bef[0,:],'^-',color='steelblue',linewidth=2,\
         label='POM '+str(tpom_bef)[0:16])
plt.ylim([-200,0])
plt.xlim([36.2,37])
plt.grid(True)
plt.legend(loc='lower left')
plt.ylabel('Depth (m)',fontsize=16)
plt.xlabel('Salinity',fontsize=16)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'Argo_GOFS_POM_salt_profile_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)