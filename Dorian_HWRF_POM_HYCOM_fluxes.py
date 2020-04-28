#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:18:28 2020

@author: root
"""

#%% User input

lon_lim = [-98.5,-60.0]
lat_lim = [10.0,45.0]

# Server erddap url IOOS glider dap
server = 'https://data.ioos.us/gliders/erddap'

#gliders sg666, sg665, sg668, silbo
gdata_sg665 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG665-20190718T1155/SG665-20190718T1155.nc3.nc'
gdata_sg666 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG666-20190718T1206/SG666-20190718T1206.nc3.nc'
gdata_sg668 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'

gdata = gdata_sg666

cycle = '2019082800'

delta_lon = 0 # delta longitude around hurricane track to calculate
               # statistics
Nini = 0 # 0 is the start of forecating cycle (2019082800)
      # 1 is 6 hours of forecasting cycle   (2019082806)
      # 2 is 12 hours ...... 20 is 120 hours 

Nend = 22 # indicates how far in the hurricabe track you want
          # include in the analysis. This is helpful if for ex:
          # you onl want to analyse the portion of the track
          # where the storm intensifies
          # 22 corresponds to all the hurricane track forecasted in a cycle
#Nend = 13

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# KMZ file best track Dorian
kmz_file_Dorian = '/home/aristizabal/KMZ_files/al052019_best_track-5.kmz'

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# figures
folder_fig = '/home/aristizabal/Figures/'

# folder nc files POM
folder_pom19 =  '/home/aristizabal/HWRF2019_POM_Dorian/'
folder_pom20 =  '/home/aristizabal/HWRF2020_POM_Dorian/'

# folde HWRF2020_HYCOM
folder_hycom20 = '/home/aristizabal/HWRF2020_HYCOM_Dorian/'

###################

# folder nc files POM
folder_pom_oper = folder_pom19 + 'HWRF2019_POM_dorian05l.' + cycle + '_pom_files_oper/'
folder_pom_exp = folder_pom20 + 'HWRF2020_POM_dorian05l.'  + cycle + '_pom_files_exp/'
prefix_pom = 'dorian05l.' + cycle + '.pom.00'

pom_grid_oper = folder_pom_oper + 'dorian05l.' + cycle + '.pom.grid.nc'
pom_grid_exp = folder_pom_exp + 'dorian05l.' + cycle + '.pom.grid.nc'

# Dorian track files
hwrf_pom_track_oper = folder_pom_oper + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'
hwrf_pom_track_exp = folder_pom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

# folder nc files hwrf
folder_hwrf_pom19_oper = folder_pom19 + 'HWRF2019_POM_dorian05l.' + cycle + '_grb2_to_nc_oper/'
folder_hwrf_pom20_exp = folder_pom20 + 'HWRF2020_POM_dorian05l.' + cycle + '_grb2_to_nc_exp/'

##################
# folder ab files HYCOM
folder_hycom_exp = folder_hycom20 + 'HWRF2020_HYCOM_dorian05l.' + cycle + '_hycom_files_exp/'
prefix_hycom = 'dorian05l.' + cycle + '.hwrf_rtofs_hat10_3z'

#Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HMON-HYCOM_Michael/'
Dir_HMON_HYCOM = '/home/aristizabal/ncep_model/HWRF-Hycom-WW3_exp_Michael/'
# RTOFS grid file name
hycom_grid_exp = Dir_HMON_HYCOM + 'hwrf_rtofs_hat10.basin.regional.grid'

# Dorian track files
hwrf_hycom_track_exp = folder_hycom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

# folder nc files hwrf
folder_hwrf_hycom20_exp = folder_hycom20 + 'HWRF2020_HYCOM_dorian05l.' + cycle + '_grb2_to_nc_exp/'

#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import os.path
import glob
import netCDF4
from bs4 import BeautifulSoup
from zipfile import ZipFile
#import cmocean

import sys
sys.path.append('/home/aristizabal/glider_model_comparisons_Python')
from read_glider_data import read_glider_data_thredds_server

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Function Grid glider variables according to depth

def varsg_gridded(depth,time,temp,salt,dens,delta_z):
             
    depthg_gridded = np.arange(0,np.nanmax(depth),delta_z)
    tempg_gridded = np.empty((len(depthg_gridded),len(time)))
    tempg_gridded[:] = np.nan
    saltg_gridded = np.empty((len(depthg_gridded),len(time)))
    saltg_gridded[:] = np.nan
    densg_gridded = np.empty((len(depthg_gridded),len(time)))
    densg_gridded[:] = np.nan

    for t,tt in enumerate(time):
        depthu,oku = np.unique(depth[:,t],return_index=True)
        tempu = temp[oku,t]
        saltu = salt[oku,t]
        densu = dens[oku,t]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        tempf = tempu[okdd]
        saltf = saltu[okdd]
        densf = densu[okdd]
 
        okt = np.isfinite(tempf)
        if np.sum(okt) < 3:
            temp[:,t] = np.nan
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
        
    return depthg_gridded, tempg_gridded, saltg_gridded, densg_gridded


#%%
def MLD_temp_and_dens_criteria(dt,drho,time,depth,temp,salt,dens):

    MLD_temp_crit = np.empty(temp.shape[1]) 
    MLD_temp_crit[:] = np.nan
    Tmean_temp_crit = np.empty(temp.shape[1]) 
    Tmean_temp_crit[:] = np.nan
    Smean_temp_crit = np.empty(temp.shape[1]) 
    Smean_temp_crit[:] = np.nan
    Td_temp_crit = np.empty(temp.shape[1]) 
    Td_temp_crit[:] = np.nan
    MLD_dens_crit = np.empty(temp.shape[1])
    MLD_dens_crit[:] = np.nan
    Tmean_dens_crit = np.empty(temp.shape[1])
    Tmean_dens_crit[:] = np.nan
    Smean_dens_crit = np.empty(temp.shape[1]) 
    Smean_dens_crit[:] = np.nan
    Td_dens_crit = np.empty(temp.shape[1]) 
    Td_dens_crit[:] = np.nan
    for t in np.arange(temp.shape[1]):
        if depth.ndim == 1:
            d10 = np.where(depth >= 10)[0][0]
        if depth.ndim == 2:
            d10 = np.where(depth[:,t] >= -10)[0][-1]
        T10 = temp[d10,t]
        delta_T = T10 - temp[:,t] 
        ok_mld_temp = np.where(delta_T <= dt)[0]
        rho10 = dens[d10,t]
        delta_rho = -(rho10 - dens[:,t])
        ok_mld_rho = np.where(delta_rho <= drho)[0]
        
        if ok_mld_temp.size == 0:
            MLD_temp_crit[t] = np.nan
            Td_temp_crit[t] = np.nan
            Tmean_temp_crit[t] = np.nan
            Smean_temp_crit[t] = np.nan            
        else:                             
            if depth.ndim == 1:
                MLD_temp_crit[t] = depth[ok_mld_temp[-1]]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_temp[-1]] + 1)[0][0]                 
            if depth.ndim == 2:
                MLD_temp_crit[t] = depth[ok_mld_temp[-1],t]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_temp[-1],t] + 1)[0][0]
            Td_temp_crit[t] = temp[ok_mld_plus1m,t]
            Tmean_temp_crit[t] = np.nanmean(temp[ok_mld_temp,t])
            Smean_temp_crit[t] = np.nanmean(salt[ok_mld_temp,t])
                
        if ok_mld_rho.size == 0:
            MLD_dens_crit[t] = np.nan
            Td_dens_crit[t] = np.nan
            Tmean_dens_crit[t] = np.nan
            Smean_dens_crit[t] = np.nan           
        else:
            if depth.ndim == 1:
                MLD_dens_crit[t] = depth[ok_mld_rho[-1]]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_rho[-1]] + 1)[0][0] 
            if depth.ndim == 2:
                MLD_dens_crit[t] = depth[ok_mld_rho[-1],t]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_rho[-1],t] + 1)[0][0] 
            Td_dens_crit[t] = temp[ok_mld_plus1m,t]        
            Tmean_dens_crit[t] = np.nanmean(temp[ok_mld_rho,t])
            Smean_dens_crit[t] = np.nanmean(salt[ok_mld_rho,t]) 

    return MLD_temp_crit,Tmean_temp_crit,Smean_temp_crit,Td_temp_crit,\
           MLD_dens_crit,Tmean_dens_crit,Smean_dens_crit,Td_dens_crit
           
#%% Get latent and sensible heat fluxes at glider position and time 
def get_sensible_latent_heat_fluxes_HWRF(ncfile_list):

    shtfl_hwrf = []
    lhtfl_hwrf = []
    time_hwrf = []
    
    for N,file in enumerate(ncfile_list):
        print(N)
        HWRF = xr.open_dataset(file)
        lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
        lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
        t_hwrf = np.asarray(HWRF.variables['time'][:])
        SHTFL_hwrf = np.asarray(HWRF.variables['SHTFL_surface'][0,:,:])
        LHTFL_hwrf = np.asarray(HWRF.variables['LHTFL_surface'][0,:,:])
        
        # Changing times to timestamp
        tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
        tstamp_hwrf = [mdates.date2num(t_hwrf[i]) for i in np.arange(len(t_hwrf))]
        
        # interpolating glider lon and lat to lat and lon on model time
        sublon_hwrf = np.interp(tstamp_hwrf,tstamp_glider,long)
        sublat_hwrf = np.interp(tstamp_hwrf,tstamp_glider,latg)
        
        # getting the model grid positions for sublonm and sublatm
        oklon_hwrf = np.round(np.interp(sublon_hwrf,lon_hwrf,np.arange(len(lon_hwrf)))).astype(int)
        oklat_hwrf = np.round(np.interp(sublat_hwrf,lat_hwrf,np.arange(len(lat_hwrf)))).astype(int)
            
        shtfl_hwrf.append(SHTFL_hwrf[oklat_hwrf,oklon_hwrf][0])
        lhtfl_hwrf.append(LHTFL_hwrf[oklat_hwrf,oklon_hwrf][0])
        time_hwrf.append(t_hwrf[0])
        
    shtfl_hwrf = np.asarray(shtfl_hwrf)
    lhtfl_hwrf = np.asarray(lhtfl_hwrf)
    time_hwrf = np.asarray(time_hwrf)
    
    return shtfl_hwrf, lhtfl_hwrf, time_hwrf

#%% Read best storm track from kmz file
    
def read_kmz_file_storm_best_track(kmz_file):
    
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
        lon_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
        lat_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
             
    #  get time stamp
    time_best_track = []
    for i,s in enumerate(soup.find_all("atcfdtg")):
        tt = datetime.strptime(s.get_text(' '),'%Y%m%d%H')
        time_best_track.append(tt)
    time_best_track = np.asarray(time_best_track)    
    
    # get type 
    wind_int_mph = []
    for i,s in enumerate(soup.find_all("intensitymph")):
        wind_int_mph.append(s.get_text(' ')) 
    wind_int_mph = np.asarray(wind_int_mph)
    wind_int_mph = wind_int_mph.astype(float)  
    
    wind_int_kt = []
    for i,s in enumerate(soup.find_all("intensity")):
        wind_int_kt.append(s.get_text(' ')) 
    wind_int_kt = np.asarray(wind_int_kt)
    wind_int_kt = wind_int_kt.astype(float)
      
    cat = []
    for i,s in enumerate(soup.find_all("styleurl")):
        cat.append(s.get_text('#').split('#')[-1]) 
    cat = np.asarray(cat)
    
    mslp = []
    for i,s in enumerate(soup.find_all("minsealevelpres")):
        mslp.append(s.get_text('#').split('#')[-1]) 
    mslp = np.asarray(mslp)
    
    return lon_best_track, lat_best_track, time_best_track, wind_int_mph, wind_int_kt, cat, mslp 

#%% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Get Dorian best track and winds

lon_best_track, lat_best_track, time_best_track, wind_int_mph, _, _, mslp = \
read_kmz_file_storm_best_track(kmz_file_Dorian)

wind_int_ms = 0.447 * wind_int_mph

#%% Reading glider data
#Time window
date_ini = cycle[0:4]+'/'+cycle[4:6]+'/'+cycle[6:8]+'/'+cycle[8:]+'/00/00'
tini = datetime.strptime(date_ini,'%Y/%m/%d/%H/%M/%S')
tend = tini + timedelta(hours=120)
date_end = tend.strftime('%Y/%m/%d/%H/%M/%S')
    
url_glider = gdata

var = 'temperature'
scatter_plot = 'no'
kwargs = dict(date_ini=date_ini[0:-6],date_end=date_end[0:-6])

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
             
#%% Grid glider variables according to depth

delta_z = 0.5
depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded = \
varsg_gridded(depthg,timeg,tempg,saltg,densg,delta_z)  

#%% Calculation of mixed layer depth based on temperature and density critria
# Tmean: mean temp within the mixed layer and 
# td: temp at 1 meter below the mixed layer            

dt = 0.2
drho = 0.125

# for glider data
MLD_temp_crit_glid, _, _, _, MLD_dens_crit_glid, Tmean_dens_crit_glid, Smean_dens_crit_glid, _ = \
MLD_temp_and_dens_criteria(dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)             

#%% Get list HWRF files
HWRF_POM_oper = sorted(glob.glob(os.path.join(folder_hwrf_pom19_oper,'*.nc')))
HWRF_POM_exp = sorted(glob.glob(os.path.join(folder_hwrf_pom20_exp,'*.nc')))
HWRF_HYCOM_exp = sorted(glob.glob(os.path.join(folder_hwrf_hycom20_exp,'*.nc')))

#%% Get storm intensity 
max_wind_10m_hwrf_pom_oper = []
max_wind_10m_hwrf_pom_exp = []
max_wind_10m_hwrf_hycom_exp = []
time_hwrf = []

for fl in HWRF_POM_oper:
        HWRF = xr.open_dataset(fl)
        t_hwrf = np.asarray(HWRF.variables['time'][:])
        UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
        VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
        max_wind_10m_hwrf_pom_oper.append(np.max(np.sqrt(UGRD_hwrf**2 + VGRD_hwrf**2)))
        time_hwrf.append(t_hwrf)

for fl in HWRF_POM_exp:
        HWRF = xr.open_dataset(fl)
        UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
        VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
        max_wind_10m_hwrf_pom_exp.append(np.max(np.sqrt(UGRD_hwrf**2 + VGRD_hwrf**2)))

for fl in HWRF_HYCOM_exp:
        HWRF = xr.open_dataset(fl)
        t_hwrf = np.asarray(HWRF.variables['time'][:])
        UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
        VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
        max_wind_10m_hwrf_hycom_exp.append(np.max(np.sqrt(UGRD_hwrf**2 + VGRD_hwrf**2)))
        time_hwrf.append(t_hwrf)

# wind speed in knots
max_wind_10m_hwrf_pom_oper = 1.94384 * np.asarray(max_wind_10m_hwrf_pom_oper)
max_wind_10m_hwrf_pom_exp = 1.94384 * np.asarray(max_wind_10m_hwrf_pom_exp)
max_wind_10m_hwrf_hycom_exp = 1.94384 * np.asarray(max_wind_10m_hwrf_hycom_exp)

#%% Read HWRF nc files
shtfl_maxwind = []
lhtfl_maxwind = []
time_hwrf = []
for fl in HWRF_POM_oper:
        print(fl)
        HWRF = xr.open_dataset(fl)
        lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
        lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
        t_hwrf = np.asarray(HWRF.variables['time'][:])
        UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
        VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
        SHTFL_hwrf = np.asarray(HWRF.variables['SHTFL_surface'][0,:,:])
        LHTFL_hwrf = np.asarray(HWRF.variables['LHTFL_surface'][0,:,:])
        DSWRD_hwrf = np.asarray(HWRF.variables['DSWRF_surface'][0,:,:])
        USWRD_hwrf = np.asarray(HWRF.variables['USWRF_surface'][0,:,:])
        DLWRD_hwrf = np.asarray(HWRF.variables['DLWRF_surface'][0,:,:])
        ULWRD_hwrf = np.asarray(HWRF.variables['ULWRF_surface'][0,:,:])
        WTMP_hwrf = np.asarray(HWRF.variables['WTMP_surface'][0,:,:])
        SWRD_hwrf = DSWRD_hwrf - USWRD_hwrf
        LWRD_hwrf = DLWRD_hwrf - ULWRD_hwrf
         
        wind_10m = np.sqrt(UGRD_hwrf**2 + VGRD_hwrf**2)
        ok_maxwind = np.where(wind_10m == np.max(wind_10m))
        time_hwrf.append(t_hwrf)
        shtfl_maxwind.append(SHTFL_hwrf[ok_maxwind[0][0],ok_maxwind[1][0]])
        lhtfl_maxwind.append(LHTFL_hwrf[ok_maxwind[0][0],ok_maxwind[1][0]])
        

#%% Get latent and sensible heat fluxes at glider position and time 
  
shtfl_hwrf19_pom, lhtfl_hwrf19_pom, time_hwrf19_pom = \
           get_sensible_latent_heat_fluxes_HWRF(HWRF_POM_oper)
           
shtfl_hwrf20_pom, lhtfl_hwrf20_pom, time_hwrf20_pom = \
           get_sensible_latent_heat_fluxes_HWRF(HWRF_POM_exp)
         
shtfl_hwrf20_hycom, lhtfl_hwrf20_hycom,time_hwrf20_hycom = \
          get_sensible_latent_heat_fluxes_HWRF(HWRF_HYCOM_exp)

#%% Estimate bulk heat fluxes during glider deployments
          
url_NDBC = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/41043/41043h2019.nc'

wind_NDBC = xr.open_dataset(url_NDBC,decode_times=False)

tt = wind_NDBC['time']
time_NDBC = netCDF4.num2date(tt[:],tt.units)
okt = np.logical_and(time_NDBC >= tini, time_NDBC < tend)

t_NDBC = time_NDBC[okt]
wspd_NDBC = np.asarray(wind_NDBC['wind_spd'][:])[okt,0,0] # winds measured at 3.8 meters height
wdir_NDBC = np.asarray(wind_NDBC['wind_dir'][:])[okt,0,0] 
Tair_NDBC = np.asarray(wind_NDBC['air_temperature'][:])[okt,0,0]
Tsea_NDBC = np.asarray(wind_NDBC['sea_surface_temperature'][:])[okt,0,0]

# Simple estimate from S. A. Hsu et al. (1994) (https://www.ndbc.noaa.gov/adjust_wind.shtml)
# Uz2 = Uz1 (z2/z1)^(0.11)
wspd_NDBC_10m = wspd_NDBC *(10/3.8)**(0.11)

# Sensible heat flux NDBC
rho = 1.184 # kg/m^3
cp = 1004 # J/(kg K)
cs = 1 * 10**(-3)
Tair_NDBC_to_glid = np.interp(mdates.date2num(timeg),mdates.date2num(t_NDBC),Tair_NDBC)
del_t = Tmean_dens_crit_glid - Tair_NDBC_to_glid
oktg = np.logical_and(timeg >= datetime(2019,8,29,6),timeg <= datetime(2019,8,29,6,30))
oktt = time_best_track == datetime(2019,8,29,6)
Qs_glider = rho * cp * cs * wind_int_ms[oktt] * del_t[oktg]
#Qs_glider2 = 111.3 #SG665
Qs_glider2 = 42.1  #SG666

# Latent heat flux NDBC
rho = 1.184 # kg/m^3
Le = 2.5 * 10**(6) # J/(kg K)
cl = 1.2 * 10**(-3)
qs = 1
rh = 0.97
Ql_glider = rho * Le * cl * wind_int_ms[oktt] * qs *(1 - rh)   
#Ql_glider2 = 419.7 #SG665
Ql_glider2 = 184.9 #SG666


#%% Wind speed NDBC
fig,ax = plt.subplots(figsize=(7,5))
plt.plot(t_NDBC,wspd_NDBC_10m,'.-',color='indianred',label='Wind speed NDBC adjusted at 10m')
plt.plot(t_NDBC,wspd_NDBC,'.-',color='brown',label='Wind speed NDBC at 3.8')
plt.legend()
plt.ylabel('$m/s$',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(np.tile(datetime(2019,8,29,0),len(np.arange(13))),np.arange(13),'--k')

#%% Tair_NDBC, Tsea_NDBC, ML temp from glider
fig,ax = plt.subplots(figsize=(7,5))
plt.plot(t_NDBC,Tsea_NDBC,'.-',color='indianred',label='Tsea_NDBC')
plt.plot(t_NDBC,Tair_NDBC,'.--',color='indianred',label='Tair_NDBC')
plt.plot(timeg,Tmean_dens_crit_glid,'.-',color='royalblue',label='ML Temp ' +inst_id.split('-')[0])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
#plt.plot(np.tile(datetime(2019,8,29,0),len(np.arange(24,30,0.01))),np.arange(24,30,0.01),'--k')
plt.plot(np.tile(datetime(2019,8,29,6),len(np.arange(24,30,0.01))),np.arange(24,30,0.01),'--k')
plt.legend()
plt.ylabel('$^oC$',fontsize=14)

#%% Tsea_NDBC - Tair_NDBC, ML temp from glider - Tair_NDBC
fig,ax = plt.subplots(figsize=(7,5))
plt.plot(t_NDBC,Tsea_NDBC - Tair_NDBC,'.-',color='indianred',label='Tsea NDBC - Tair NDBC')
plt.plot(timeg,Tmean_dens_crit_glid - Tair_NDBC_to_glid ,'.-',color='royalblue',label='ML Temp ' +inst_id.split('-')[0]+ ' - Tair NDBC')
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(np.tile(datetime(2019,8,28,18),len(np.arange(6))),np.arange(6),'--k')
plt.legend()
plt.ylabel('$^oC$',fontsize=14)
         
#%% Heat fluxes
fig,ax = plt.subplots(figsize=(10,5))
plt.plot(time_hwrf19_pom,shtfl_hwrf19_pom,'.--',color='mediumorchid',label='Sensible Heat Flux',linewidth=2)
plt.plot(time_hwrf19_pom,lhtfl_hwrf19_pom,'.-',color='mediumorchid',label='Latent Heat Flux',linewidth=2)
plt.plot(time_hwrf19_pom,lhtfl_hwrf19_pom,'.-',color='mediumorchid',label='HWRF2019-POM Oper',linewidth=2)
plt.plot(time_hwrf19_pom,shtfl_hwrf19_pom,'.--',color='mediumorchid',linewidth=2)
plt.plot(time_hwrf20_pom,shtfl_hwrf20_pom,'.--',color='teal',linewidth=2)
plt.plot(time_hwrf20_pom,lhtfl_hwrf20_pom,'.-',color='teal',label='HWRF2020-POM Exp',linewidth=2)
plt.plot(time_hwrf20_hycom,shtfl_hwrf20_hycom,'.--',color='darkorange',linewidth=2)
plt.plot(time_hwrf20_hycom,lhtfl_hwrf20_hycom,'.-',color='darkorange',label='HWRF2020-HYCOM Exp')
plt.plot(timeg[oktg],Qs_glider2,'s',color='royalblue',label='Sensible '+inst_id.split('-')[0],markeredgecolor='k',markersize=7)
plt.plot(timeg[oktg],Ql_glider2,'o',color='royalblue',label='Latent '+inst_id.split('-')[0],markeredgecolor='k',markersize=7)
plt.legend()
plt.ylabel('($W/m^2$)',fontsize = 14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(np.tile(datetime(2019,8,29,0),len(np.arange(600))),np.arange(600),'--k')
#plt.plot(np.tile(datetime(2019,8,28,15),len(np.arange(600))),np.arange(600),'--k')

#%% Heat fluxes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

lead_time = np.arange(0,129,3)

fig,ax = plt.subplots(figsize=(10,5))
plt.plot(lead_time,shtfl_hwrf19_pom,'.--',color='mediumorchid',label='Sensible Heat Flux',linewidth=2)
plt.plot(lead_time,lhtfl_hwrf19_pom,'.-',color='mediumorchid',label='Latent Heat Flux',linewidth=2)
plt.plot(lead_time,lhtfl_hwrf19_pom,'.-',color='mediumorchid',label='HWRF2019-POM Oper',linewidth=2)
plt.plot(lead_time,shtfl_hwrf19_pom,'.--',color='mediumorchid',linewidth=2)
plt.plot(lead_time,shtfl_hwrf20_pom,'.--',color='teal',linewidth=2)
plt.plot(lead_time,lhtfl_hwrf20_pom,'.-',color='teal',label='HWRF2020-POM Exp',linewidth=2)
plt.plot(lead_time,shtfl_hwrf20_hycom,'.--',color='darkorange',linewidth=2)
plt.plot(lead_time,lhtfl_hwrf20_hycom,'.-',color='darkorange',label='HWRF2020-HYCOM Exp')
#plt.plot(24,Qs_glider2,'s',color='royalblue',label='Sensible '+inst_id.split('-')[0],markeredgecolor='k',markersize=7)
#plt.plot(24,Ql_glider2,'o',color='royalblue',label='Latent '+inst_id.split('-')[0],markeredgecolor='k',markersize=7)
plt.legend()
plt.ylabel('($W/m^2$)',fontsize = 14)
plt.xlim([0,126])
ax.xaxis.set_major_locator(MultipleLocator(12))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(MultipleLocator(3))
#plt.plot(np.tile(24,len(np.arange(600))),np.arange(600),'--k')
plt.plot(np.tile(15,len(np.arange(600))),np.arange(600),'--k')
plt.xlabel('Forecast Lead Time (Hr)',fontsize=14)

#%% Sensible heat flux
fig,ax = plt.subplots(figsize=(7,5))
plt.plot(time_hwrf19_pom,shtfl_hwrf19_pom,'.--',color='purple',label='HWRF2019-POM Oper')
plt.plot(time_hwrf20_pom,shtfl_hwrf20_pom,'.--',color='teal',label='HWRF2020-POM Exp')
plt.plot(time_hwrf20_hycom,shtfl_hwrf20_hycom,'.--',color='orange',label='HWRF2020-HYCOM Exp')
plt.plot(timeg[oktg],Qs_glider2,'o',color='royalblue',label=inst_id.split('-')[0])
plt.ylabel('($W/m^2$)',fontsize = 14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(np.tile(datetime(2019,8,28,18),len(np.arange(110))),np.arange(110),'--k')
plt.legend()
plt.title('Sensible Heat Flux',fontsize=14)
      
#%% map wind vectors HWRP2019-POM operational
N = 6
file = HWRF_POM_oper[N]
HWRF = xr.open_dataset(file)
lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
t_hwrf = np.asarray(HWRF.variables['time'][:])
UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
SHTFL_hwrf = np.asarray(HWRF.variables['SHTFL_surface'][0,:,:])

kw = dict(levels=np.linspace(-200,200,11))

fig,ax = plt.subplots()    
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.contourf(lon_hwrf,lat_hwrf,SHTFL_hwrf,cmap=plt.cm.coolwarm,**kw)
c = plt.colorbar()
q = plt.quiver(lon_hwrf[::30], lat_hwrf[::30],UGRD_hwrf[::30,::30],VGRD_hwrf[::30,::30]) #,units='xy' ,scale=0.01)
plt.quiverkey(q,np.max(lon_hwrf)+3,np.max(lat_hwrf)+1.5,30,"30 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.plot(-77.4,27.0,'*r',markersize=7)
plt.title('HWRF19-POM Sensible Heat Flux on '+str(t_hwrf[0])[0:13],fontsize=14)
c.set_label('$W/m^2$',rotation=90, labelpad=1, fontsize=16)
c.ax.tick_params(labelsize=14)
plt.ylim([np.min(lat_hwrf),np.max(lat_hwrf)])
plt.xlim([np.min(lon_hwrf),np.max(lon_hwrf)])

file_name = folder_fig + 'Dorian_HWRF19_POM_SHTFL_' + str(t_hwrf[0])[2:18]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1) 

#%% map wind vectors HWRP2020-POM experimental
N = 6
file = HWRF_POM_oper[N]
HWRF = xr.open_dataset(file)
lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
t_hwrf = np.asarray(HWRF.variables['time'][:])
UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
SHTFL_hwrf = np.asarray(HWRF.variables['SHTFL_surface'][0,:,:])

kw = dict(levels=np.linspace(-200,200,11))

fig,ax = plt.subplots()    
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.contourf(lon_hwrf,lat_hwrf,SHTFL_hwrf,cmap=plt.cm.coolwarm,**kw)
c = plt.colorbar()
q = plt.quiver(lon_hwrf[::30], lat_hwrf[::30],UGRD_hwrf[::30,::30],VGRD_hwrf[::30,::30]) #,units='xy' ,scale=0.01)
plt.quiverkey(q,np.max(lon_hwrf)+3,np.max(lat_hwrf)+1.5,30,"30 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.plot(-77.4,27.0,'*r',markersize=7)
plt.title('HWRF20-POM Sensible Heat Flux on '+str(t_hwrf[0])[0:13],fontsize=14)
c.set_label('$W/m^2$',rotation=90, labelpad=1, fontsize=16)
c.ax.tick_params(labelsize=14)
plt.ylim([np.min(lat_hwrf),np.max(lat_hwrf)])
plt.xlim([np.min(lon_hwrf),np.max(lon_hwrf)])

file_name = folder_fig + 'Dorian_HWRF20_POM_SHTFL_' + str(t_hwrf[0])[2:18]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1) 

#%% map wind vectors HWRP2020-HYCOM experimental
N = 6
file = HWRF_HYCOM_exp[N]
HWRF = xr.open_dataset(file)
lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
t_hwrf = np.asarray(HWRF.variables['time'][:])
UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
SHTFL_hwrf = np.asarray(HWRF.variables['SHTFL_surface'][0,:,:])

kw = dict(levels=np.linspace(-200,200,11))

fig,ax = plt.subplots()    
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.contourf(lon_hwrf,lat_hwrf,SHTFL_hwrf,cmap=plt.cm.coolwarm,**kw)
c = plt.colorbar()
q = plt.quiver(lon_hwrf[::30], lat_hwrf[::30],UGRD_hwrf[::30,::30],VGRD_hwrf[::30,::30]) #,units='xy' ,scale=0.01)
plt.quiverkey(q,np.max(lon_hwrf)+3,np.max(lat_hwrf)+1.5,30,"30 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.plot(-77.4,27.0,'*r',markersize=7)
plt.title('HWRF20-HYCOM Sensible Heat Flux on '+str(t_hwrf[0])[0:13],fontsize=14)
c.set_label('$W/m^2$',rotation=90, labelpad=1, fontsize=16)
c.ax.tick_params(labelsize=14)
plt.ylim([np.min(lat_hwrf),np.max(lat_hwrf)])
plt.xlim([np.min(lon_hwrf),np.max(lon_hwrf)])

file_name = folder_fig + 'Dorian_HWRF20_HYCOM_SHTFL_' + str(t_hwrf[0])[2:18]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1) 

#%% map wind vectors HWRP2019-POM operational
N = 6
file = HWRF_POM_oper[N]
HWRF = xr.open_dataset(file)
lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
t_hwrf = np.asarray(HWRF.variables['time'][:])
UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
LHTFL_hwrf = np.asarray(HWRF.variables['LHTFL_surface'][0,:,:])

kw = dict(levels=np.linspace(-600,600,13))

fig,ax = plt.subplots()    
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.contourf(lon_hwrf,lat_hwrf,LHTFL_hwrf,cmap=plt.cm.coolwarm,**kw)
c = plt.colorbar()
q = plt.quiver(lon_hwrf[::30], lat_hwrf[::30],UGRD_hwrf[::30,::30],VGRD_hwrf[::30,::30]) #,units='xy' ,scale=0.01)
plt.quiverkey(q,np.max(lon_hwrf)+3,np.max(lat_hwrf)+1.5,30,"30 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.plot(-77.4,27.0,'*r',markersize=7)
plt.title('HWRF19-POM Latent Heat Flux on '+str(t_hwrf[0])[0:13],fontsize=14)
c.set_label('$W/m^2$',rotation=90, labelpad=1, fontsize=16)
c.ax.tick_params(labelsize=14)
plt.ylim([np.min(lat_hwrf),np.max(lat_hwrf)])
plt.xlim([np.min(lon_hwrf),np.max(lon_hwrf)])

file_name = folder_fig + 'Dorian_HWRF19_POM_LHTFL_' + str(t_hwrf[0])[2:18]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1) 

#%% map wind vectors HWRP2020-POM operational
N = 6
file = HWRF_POM_exp[N]
HWRF = xr.open_dataset(file)
lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
t_hwrf = np.asarray(HWRF.variables['time'][:])
UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
LHTFL_hwrf = np.asarray(HWRF.variables['LHTFL_surface'][0,:,:])

kw = dict(levels=np.linspace(-600,600,13))

fig,ax = plt.subplots()    
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.contourf(lon_hwrf,lat_hwrf,LHTFL_hwrf,cmap=plt.cm.coolwarm,**kw)
c = plt.colorbar()
q = plt.quiver(lon_hwrf[::30], lat_hwrf[::30],UGRD_hwrf[::30,::30],VGRD_hwrf[::30,::30]) #,units='xy' ,scale=0.01)
plt.quiverkey(q,np.max(lon_hwrf)+3,np.max(lat_hwrf)+1.5,30,"30 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.plot(-77.4,27.0,'*r',markersize=7)
plt.title('HWRF20-POM Latent Heat Flux on '+str(t_hwrf[0])[0:13],fontsize=14)
c.set_label('$W/m^2$',rotation=90, labelpad=1, fontsize=16)
c.ax.tick_params(labelsize=14)
plt.ylim([np.min(lat_hwrf),np.max(lat_hwrf)])
plt.xlim([np.min(lon_hwrf),np.max(lon_hwrf)])

file_name = folder_fig + 'Dorian_HWRF20_POM_LHTFL_' + str(t_hwrf[0])[2:18]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1) 

#%% map wind vectors HWRP2020-HYCOM operational
N = 6
file = HWRF_HYCOM_exp[N]
HWRF = xr.open_dataset(file)
lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
t_hwrf = np.asarray(HWRF.variables['time'][:])
UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
LHTFL_hwrf = np.asarray(HWRF.variables['LHTFL_surface'][0,:,:])

kw = dict(levels=np.linspace(-600,600,13))

fig,ax = plt.subplots()    
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.contourf(lon_hwrf,lat_hwrf,LHTFL_hwrf,cmap=plt.cm.coolwarm,**kw)
c = plt.colorbar()
q = plt.quiver(lon_hwrf[::30], lat_hwrf[::30],UGRD_hwrf[::30,::30],VGRD_hwrf[::30,::30]) #,units='xy' ,scale=0.01)
plt.quiverkey(q,np.max(lon_hwrf)+3,np.max(lat_hwrf)+1.5,30,"30 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.plot(-77.4,27.0,'*r',markersize=7)
plt.title('HWRF20-HYCOM Latent Heat Flux on '+str(t_hwrf[0])[0:13],fontsize=14)
c.set_label('$W/m^2$',rotation=90, labelpad=1, fontsize=16)
c.ax.tick_params(labelsize=14)
plt.ylim([np.min(lat_hwrf),np.max(lat_hwrf)])
plt.xlim([np.min(lon_hwrf),np.max(lon_hwrf)])

file_name = folder_fig + 'Dorian_HWRF20_HYCOM_LHTFL_' + str(t_hwrf[0])[2:18]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1) 


#%% map SST 
'''
m = Basemap(projection='merc',llcrnrlat=15,urcrnrlat=35,llcrnrlon=-80,urcrnrlon=-60,resolution='l')
x, y = m(*np.meshgrid(lon_hwrf,lat_hwrf))
plt.figure()
plt.ion()
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
kw = dict(levels=np.linspace(24,33,10))
plt.contourf(x,y,WTMP_hwrf-273.15,cmap=cmocean.cm.thermal,**kw)
c = plt.colorbar()
q = plt.quiver(x[::30,::30], y[::30,::30],UGRD_hwrf[::30,::30],VGRD_hwrf[::30,::30])
xq,yq = m(-78,12.5)
plt.quiverkey(q,xq,yq,30,"30 m/s",coordinates='data',color='k',fontproperties={'size': 14})
xc, yc = m(-77.4,27.0)
plt.plot(xc,yc,'*r',markersize=7)
plt.title('HWRF SST on '+str(time_hwrf)[2:18],fontsize=16)
c.set_label('$^oC$',rotation=90, labelpad=15, fontsize=16)
c.ax.tick_params(labelsize=14)
'''
#%%
'''
shtfl_maxwind = np.asarray(shtfl_maxwind)
lhtfl_maxwind = np.asarray(lhtfl_maxwind)

fig,ax = plt.subplots()
plt.plot(time_hwrf,shtfl_maxwind,'.-',label='Sensible Heat Flux')
plt.plot(time_hwrf,lhtfl_maxwind,'.-',label='Latent Heat Flux')
plt.plot(time_hwrf,lhtfl_maxwind+shtfl_maxwind,'.-',label='Enthalpy')
plt.legend()
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
'''
        
