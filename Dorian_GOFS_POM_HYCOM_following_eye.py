#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:07:48 2020

@author: root
"""

#%% User input

lon_lim = [-98.5,-60.0]
lat_lim = [10.0,45.0]

#cycle = '2019082800'
#cycle = '2019082918'
cycle = '2019083018'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc' 

# KMZ file best track Dorian
kmz_file_Dorian = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/al052019_best_track-5.kmz'  

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# figures
folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# folder nc files POM
folder_pom19 =  '/Volumes/aristizabal/HWRF2019_POM_Dorian/'
folder_pom20 =  '/Volumes/aristizabal/HWRF2020_POM_Dorian/'

# folde HWRF2020_HYCOM
folder_hycom20 = '/Volumes/aristizabal/HWRF2020_HYCOM_Dorian/'

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

##################
# folder ab files HYCOM
folder_hycom_exp = folder_hycom20 + 'HWRF2020_HYCOM_dorian05l.' + cycle + '_hycom_files_exp/'
prefix_hycom = 'dorian05l.' + cycle + '.hwrf_rtofs_hat10_3z'

#Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HMON-HYCOM_Michael/'
Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HWRF-Hycom-WW3_exp_Michael/'
# RTOFS grid file name
hycom_grid_exp = Dir_HMON_HYCOM + 'hwrf_rtofs_hat10.basin.regional.grid'

# Dorian track files
hwrf_hycom_track_exp = folder_hycom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os
import os.path
from bs4 import BeautifulSoup
from zipfile import ZipFile
import glob
import seawater as sw
import cmocean

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%%
def MLD_temp_and_dens_criteria(dt,drho,depth,temp,salt,dens):

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
           
#%% Function Ocean Heat Content

def OHC_surface(temp,depth,dens):
    cp = 3985 #Heat capacity in J/(kg K)

    OHC = np.empty(temp.shape[1])
    OHC[:] = np.nan
    for t in np.arange(temp.shape[1]):
        ok26 = temp[:,t] >= 26
        if len(depth[ok26]) != 0:
            if np.nanmin(depth[ok26])>10:
                OHC[t] = np.nan  
            else:
                rho0 = np.nanmean(dens[ok26,t])
                if depth.ndim == 1:
                    OHC[t] = np.abs(cp * rho0 * np.trapz(temp[ok26,t]-26,depth[ok26]))
                if depth.ndim == 2:
                    OHC[t] = np.abs(cp * rho0 * np.trapz(temp[ok26,t]-26,depth[ok26,t]))
        else:    
            OHC[t] = np.nan
            
    return OHC

#%% Get storm track from HWRF/POM output

def get_storm_track_POM(file_track):

    ff = open(file_track,'r')
    f = ff.readlines()
    
    latt = []
    lont = []
    lead_time = []
    for l in f:
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
        lont.append(lon)
        lead_time.append(int(l.split(',')[5][1:4]))
    
    latt = np.asarray(latt)
    lont = np.asarray(lont)
    lead_time, ind = np.unique(lead_time,return_index=True)
    lat_track = latt[ind]
    lon_track = lont[ind]  

    return lon_track, lat_track, lead_time

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
    
    return lon_best_track, lat_best_track, time_best_track, wind_int_mph, wind_int_kt, cat

#%% Function Getting glider transect from GOFS
    
def get_glider_transect_from_GOFS(depth_GOFS,oktime_GOFS,oklat_GOFS,oklon_GOFS):
    
    print('Getting glider transect from GOFS')
    target_temp_GOFS = np.empty((len(depth_GOFS),len(oktime_GOFS[0])))
    target_temp_GOFS[:] = np.nan
    target_salt_GOFS = np.empty((len(depth_GOFS),len(oktime_GOFS[0])))
    target_salt_GOFS[:] = np.nan
    for i in range(len(oktime_GOFS[0])):
        print(len(oktime_GOFS[0]),' ',i)
        target_temp_GOFS[:,i] = GOFS.variables['water_temp'][oktime_GOFS[0][i],:,oklat_GOFS[i],oklon_GOFS[i]]
        target_salt_GOFS[:,i] = GOFS.variables['salinity'][oktime_GOFS[0][i],:,oklat_GOFS[i],oklon_GOFS[i]]

    return target_temp_GOFS,target_salt_GOFS

#%%
def get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_pom,\
                                 lon_pom,lat_pom,tstamp_glider,long,latg):

    ncfiles = sorted(glob.glob(os.path.join(folder_pom,prefix+'*.nc')))

    target_temp_POM = np.empty((len(zlev),len(ncfiles)))
    target_temp_POM[:] = np.nan
    target_salt_POM = np.empty((len(zlev),len(ncfiles)))
    target_salt_POM[:] = np.nan
    target_dens_POM = np.empty((len(zlev),len(ncfiles)))
    target_dens_POM[:] = np.nan
    target_depth_POM = np.empty((len(zlev),len(ncfiles)))
    target_depth_POM[:] = np.nan
    time_POM = []
    
    for x,file in enumerate(ncfiles):
        print(x)
        pom = xr.open_dataset(file)
        
        tpom = pom['time'][:]
        timestamp_pom = mdates.date2num(tpom)[0]
        time_POM.append(mdates.num2date(timestamp_pom))
        
        # Interpolating latg and longlider into RTOFS grid
        sublonpom = np.interp(timestamp_pom,tstamp_glider,long)
        sublatpom = np.interp(timestamp_pom,tstamp_glider,latg)
        oklonpom = np.int(np.round(np.interp(sublonpom,lon_pom[0,:],np.arange(len(lon_pom[0,:])))))
        oklatpom = np.int(np.round(np.interp(sublatpom,lat_pom[:,0],np.arange(len(lat_pom[:,0])))))
        
        target_temp_POM[:,x] = np.asarray(pom['t'][0,:,oklatpom,oklonpom])
        target_salt_POM[:,x] = np.asarray(pom['s'][0,:,oklatpom,oklonpom])
        target_rho_pom = np.asarray(pom['rho'][0,:,oklatpom,oklonpom])
        target_dens_POM[:,x] = target_rho_pom * 1000 + 1000
        target_depth_POM[:,x] = zmatrix_pom[oklatpom,oklonpom,:].T
        
    target_temp_POM[target_temp_POM==0] = np.nan
    target_salt_POM[target_salt_POM==0] = np.nan
    target_dens_POM[target_dens_POM==1000.0] = np.nan
            
    return time_POM, target_temp_POM, target_salt_POM, target_dens_POM, target_depth_POM

#%%
    
def get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,lon_hycom,lat_hycom,var,timestamp_glider,lon_glider,lat_glider):

    afiles = sorted(glob.glob(os.path.join(folder_hycom,prefix+'*.a')))    
        
    target_var_hycom = np.empty((nz,len(afiles)))
    target_var_hycom[:] = np.nan
    time_hycom = []
    for x, file in enumerate(afiles):
        print(x)
        #lines=[line.rstrip() for line in open(file[:-2]+'.b')]
    
        #Reading time stamp
        year = int(file.split('/')[-1].split('.')[1][0:4])
        month = int(file.split('/')[-1].split('.')[1][4:6])
        day = int(file.split('/')[-1].split('.')[1][6:8])
        hour = int(file.split('/')[-1].split('.')[1][8:10])
        dt = int(file.split('/')[-1].split('.')[3][1:])
        timestamp_hycom = mdates.date2num(datetime(year,month,day,hour)) + dt/24
        time_hycom.append(mdates.num2date(timestamp_hycom))
        
        # Interpolating latg and longlider into HYCOM grid
        sublon_hycom = np.interp(timestamp_hycom,timestamp_glider,lon_glider)
        sublat_hycom = np.interp(timestamp_hycom,timestamp_glider,lat_glider)
        oklon_hycom = np.int(np.round(np.interp(sublon_hycom,lon_hycom[0,:],np.arange(len(lon_hycom[0,:])))))
        oklat_hycom = np.int(np.round(np.interp(sublat_hycom,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))))
        
        # Reading 3D variable from binary file 
        var_hycom = readBinz(file[:-2],'3z',var)
        #ts=readBin(afile,'archive','temp')
        target_var_hycom[:,x] = var_hycom[oklat_hycom,oklon_hycom,:]
        
    time_hycom = np.asarray(time_hycom)
    #timestamp_hycom = mdates.date2num(time_hycom)
    
    return target_var_hycom, time_hycom

#%%    
def depth_aver_top_100(depth,var):

    varmean100 = np.empty(var.shape[1])
    varmean100[:] = np.nan
    if depth.ndim == 1:
        okd = np.abs(depth) <= 100
        if len(depth[okd]) != 0:
            for t in np.arange(var.shape[1]):
                if len(np.where(np.isnan(var[okd,t]))[0])>10:
                    varmean100[t] = np.nan
                else:
                    varmean100[t] = np.nanmean(var[okd,t],0)
    else:
        for t in np.arange(depth.shape[1]):
            okd = np.abs(depth[:,t]) <= 100
            if len(depth[okd,t]) != 0:
                if len(np.where(np.isnan(var[okd,t]))[0])>10:
                    varmean100[t] = np.nan
                else:
                    varmean100[t] = np.nanmean(var[okd,t])
            else:
                varmean100[t] = np.nan
    
    return varmean100  

#%% Function Conversion from glider longitude and latitude to GOFS convention

def glider_coor_to_GOFS_coord(long,latg):
    
    target_lon = np.empty((len(long),))
    target_lon[:] = np.nan
    for i,ii in enumerate(long):
        if ii < 0: 
            target_lon[i] = 360 + ii
        else:
            target_lon[i] = ii
    target_lat = latg
    
    return target_lon, target_lat

#%%  Function Conversion from GOFS convention to glider longitude and latitude
    
def GOFS_coor_to_glider_coord(lon_GOFS,lat_GOFS):
    
    lon_GOFSg = np.empty((len(lon_GOFS),))
    lon_GOFSg[:] = np.nan
    for i in range(len(lon_GOFS)):
        if lon_GOFS[i] > 180: 
            lon_GOFSg[i] = lon_GOFS[i] - 360 
        else:
            lon_GOFSg[i] = lon_GOFS[i]
    lat_GOFSg = lat_GOFS
    
    return lon_GOFSg, lat_GOFSg
        
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

#%% Read POM grid

print('Retrieving coordinates from POM')
POM_grid_oper = xr.open_dataset(pom_grid_oper,decode_times=False)
lon_POM_oper = np.asarray(POM_grid_oper['east_e'][:])
lat_POM_oper = np.asarray(POM_grid_oper['north_e'][:])
zlev_POM_oper = np.asarray(POM_grid_oper['zz'][:])
hpom_oper = np.asarray(POM_grid_oper['h'][:])
zmatrix = np.dot(hpom_oper.reshape(-1,1),zlev_POM_oper.reshape(1,-1))
zmatrix_POM_oper = zmatrix.reshape(hpom_oper.shape[0],hpom_oper.shape[1],zlev_POM_oper.shape[0])

POM_grid_exp = xr.open_dataset(pom_grid_exp,decode_times=False)
lon_POM_exp = np.asarray(POM_grid_exp['east_e'][:])
lat_POM_exp = np.asarray(POM_grid_exp['north_e'][:])
zlev_POM_exp = np.asarray(POM_grid_exp['zz'][:])
hpom_exp = np.asarray(POM_grid_exp['h'][:])
zmatrix = np.dot(hpom_exp.reshape(-1,1),zlev_POM_exp.reshape(1,-1))
zmatrix_POM_exp = zmatrix.reshape(hpom_exp.shape[0],hpom_exp.shape[1],zlev_POM_exp.shape[0])

#%% Reading HYCOM grid

# Reading lat and lon
lines_grid = [line.rstrip() for line in open(hycom_grid_exp+'.b')]
lon_HYCOM = np.array(readgrids(hycom_grid_exp,'plon:',[0]))
lat_HYCOM = np.array(readgrids(hycom_grid_exp,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles = sorted(glob.glob(os.path.join(folder_hycom_exp,prefix_hycom+'*.a')))

# Reading depths
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]=='temp':
        #print(line.split()[1])
        z.append(float(line.split()[1]))
depth_HYCOM_exp = np.asarray(z) 

nz = len(depth_HYCOM_exp) 

#%% Read GOFS 3.1 grid

print('Retrieving coordinates from GOFS')
GOFS = xr.open_dataset(url_GOFS,decode_times=False) 

tt_G = GOFS.time
t_G = netCDF4.num2date(tt_G[:],tt_G.units)
lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:]) 
depth_GOFS = np.asarray(GOFS.depth[:])

#%% Get Dorian track from POM

lon_forec_track_pom_oper, lat_forec_track_pom_oper, lead_time_pom_oper = get_storm_track_POM(hwrf_pom_track_oper)

lon_forec_track_pom_exp, lat_forec_track_pom_exp, lead_time_pom_exp = get_storm_track_POM(hwrf_pom_track_exp)

lon_forec_track_hycom_exp, lat_forec_track_hycom_exp, lead_time_hycom_exp = get_storm_track_POM(hwrf_hycom_track_exp)

#%% Build time vector for forecasted track

t0 = datetime(int(cycle[0:4]),int(cycle[4:6]),int(cycle[6:8]),int(cycle[8:10]))
time_track = [t0 + timedelta(days=n/24) for n in lead_time_pom_oper]
timestamp_track = mdates.date2num(time_track)

#%% Get Dorian best track 

lon_best_track, lat_best_track, time_best_track, _, _, _ = \
read_kmz_file_storm_best_track(kmz_file_Dorian)

#%% Get glider transect from GOFS 3.1

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=126)
date_end = str(tend)

tmin = datetime.strptime(date_ini[0:-6],'%Y-%m-%d %H')
tmax = datetime.strptime(date_end[0:-6],'%Y-%m-%d %H')
oktime_GOFS = np.where(np.logical_and(t_G >= tmin, t_G <= tmax)) 
time_GOFS = np.asarray(t_G[oktime_GOFS])
timestamp_GOFS = mdates.date2num(time_GOFS)

tstamp_glider = timestamp_track
long = lon_forec_track_pom_oper
latg = lat_forec_track_pom_oper

# Conversion from glider longitude and latitude to GOFS convention
target_lon, target_lat = glider_coor_to_GOFS_coord(long,latg)

# interpolating glider lon and lat to lat and lon on model time
sublon_GOFS = np.interp(timestamp_GOFS,timestamp_track,target_lon)
sublat_GOFS = np.interp(timestamp_GOFS,timestamp_track,target_lat)

# getting the model grid positions for sublonm and sublatm
oklon_GOFS = np.round(np.interp(sublon_GOFS,lon_G,np.arange(len(lon_G)))).astype(int)
oklat_GOFS = np.round(np.interp(sublat_GOFS,lat_G,np.arange(len(lat_G)))).astype(int)

# Conversion from GOFS convention to glider longitude and latitude
sublon_GOFSg,sublat_GOFSg = GOFS_coor_to_glider_coord(sublon_GOFS,sublat_GOFS)
    
# Getting glider transect from model
target_temp_eye_GOFS, target_salt_eye_GOFS = \
                          get_glider_transect_from_GOFS(depth_GOFS,oktime_GOFS,oklat_GOFS,oklon_GOFS)
    
#%% Calculate density for GOFS

target_dens_eye_GOFS = sw.dens(target_salt_eye_GOFS,target_temp_eye_GOFS,np.tile(depth_GOFS,(len(time_GOFS),1)).T) 

#%% Reading POM operational temperature and salinity for firts time step in forecast cycle 2018082800
# following a band around the forecasted storm track by HWRF/POM

folder_pom = folder_pom_oper
prefix = prefix_pom
zlev = zlev_POM_oper
zmatrix_POM = zmatrix_POM_oper
lon_POM = lon_POM_oper
lat_POM = lat_POM_oper
tstamp_glider = timestamp_track
long = lon_forec_track_pom_oper
latg = lat_forec_track_pom_oper

time_eye_POM_oper, target_temp_eye_POM_oper, target_salt_eye_POM_oper, \
    target_dens_eye_POM_oper, target_depth_eye_POM_oper = \
    get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_POM,\
                                 lon_POM,lat_POM,tstamp_glider,long,latg)
    
#%% Reading POM experimental temperature and salinity for firts time step in forecast cycle 2018082800
# following a band around the forecasted storm track by HWRF/POM

folder_POM = folder_pom_exp
prefix = prefix_pom
zlev = zlev_POM_exp
zmatrix_POM = zmatrix_POM_exp
lon_POM = lon_POM_exp
lat_POM = lat_POM_exp
tstamp_glider = timestamp_track
long = lon_forec_track_pom_oper
latg = lat_forec_track_pom_oper

time_eye_POM_exp, target_temp_eye_POM_exp, target_salt_eye_POM_exp, \
    target_dens_eye_POM_exp, target_depth_eye_POM_exp = \
    get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_POM,\
                                 lon_POM,lat_POM,tstamp_glider,long,latg)

#%% Get glider transect from HYCOM

folder_hycom = folder_hycom_exp
prefix = prefix_hycom
tstamp_glider = timestamp_track
long = lon_forec_track_pom_oper
latg = lat_forec_track_pom_oper
    
# Conversion from glider longitude and latitude to GOFS convention
target_lonG, target_latG = glider_coor_to_GOFS_coord(long,latg)

lon_track = target_lonG 
lat_track = target_latG

var = 'temp'
target_temp_eye_HYCOM_exp, time_eye_HYCOM_exp = \
    get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,\
    lon_HYCOM,lat_HYCOM,var,timestamp_track,lon_track,lat_track)

var = 'salinity'
target_salt_eye_HYCOM_exp, _ = \
    get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,\
      lon_HYCOM,lat_HYCOM,var,timestamp_track,lon_track,lat_track)

#%% Calculate density for HYCOM

nx = target_temp_eye_HYCOM_exp.shape[1]
target_dens_eye_HYCOM_exp = sw.dens(target_salt_eye_HYCOM_exp,target_temp_eye_HYCOM_exp,np.tile(depth_HYCOM_exp,(nx,1)).T) 

#%% Calculation of mixed layer depth based on temperature and density critria
# Tmean: mean temp within the mixed layer and 
# td: temp at 1 meter below the mixed layer            

dt = 0.2
drho = 0.125

# for GOFS 3.1 output 
MLD_temp_crit_GOFS, _, _, _, MLD_dens_crit_GOFS, Tmean_dens_crit_GOFS, Smean_dens_crit_GOFS, _ = \
MLD_temp_and_dens_criteria(dt,drho,depth_GOFS,target_temp_eye_GOFS,target_salt_eye_GOFS,target_dens_eye_GOFS)          

# for POM operational
MLD_temp_crit_POM_oper, _, _, _, MLD_dens_crit_POM_oper, Tmean_dens_crit_POM_oper, \
Smean_dens_crit_POM_oper, _ = \
MLD_temp_and_dens_criteria(dt,drho,target_depth_eye_POM_oper,target_temp_eye_POM_oper,\
                           target_salt_eye_POM_oper,target_dens_eye_POM_oper)

# for POM experimental
MLD_temp_crit_POM_exp, _, _, _, MLD_dens_crit_POM_exp, Tmean_dens_crit_POM_exp, \
Smean_dens_crit_POM_exp, _ = \
MLD_temp_and_dens_criteria(dt,drho,target_depth_eye_POM_exp,target_temp_eye_POM_exp,\
                           target_salt_eye_POM_exp,target_dens_eye_POM_exp)
    
# for HYCOM experimental 
depth=depth_HYCOM_exp
temp=np.asarray(target_temp_eye_HYCOM_exp)
salt=np.asarray(target_salt_eye_HYCOM_exp)
dens=np.asarray(target_dens_eye_HYCOM_exp) 
temp[temp>100]=np.nan
salt[salt>100]=np.nan
dens[dens<1000]=np.nan
#dens[dens>10**4]=np.nan

MLD_temp_crit_HYCOM_exp, _, _, _, MLD_dens_crit_HYCOM_exp, Tmean_dens_crit_HYCOM_exp, \
Smean_dens_crit_HYCOM_exp, _ = \
MLD_temp_and_dens_criteria(dt,drho,depth,temp,salt,dens) 
    
#%% Surface Ocean Heat Content

# GOFS
OHC_GOFS = OHC_surface(target_temp_eye_GOFS,depth_GOFS,target_dens_eye_GOFS)

# POM operational    
OHC_POM_oper = OHC_surface(target_temp_eye_POM_oper,target_depth_eye_POM_oper,\
                           target_dens_eye_POM_oper)

# POM experimental
OHC_POM_exp = OHC_surface(target_temp_eye_POM_exp,target_depth_eye_POM_exp,\
                           target_dens_eye_POM_exp)
    
# HYCOM experimental
OHC_HYCOM_exp = OHC_surface(target_temp_eye_HYCOM_exp,depth_HYCOM_exp,\
                           target_dens_eye_HYCOM_exp)  

#%% Calculate T100

T100_GOFS = depth_aver_top_100(depth_GOFS,target_temp_eye_GOFS)

T100_POM_oper = depth_aver_top_100(target_depth_eye_POM_oper,target_temp_eye_POM_oper) 

T100_POM_exp = depth_aver_top_100(target_depth_eye_POM_exp,target_temp_eye_POM_exp)

T100_HYCOM_exp = depth_aver_top_100(depth_HYCOM_exp,target_temp_eye_HYCOM_exp)    

#%% Figure forecasted track operational POM

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=float(lead_time_pom_oper[-1]))
date_end = str(tend)

okt = np.logical_and(time_best_track >= tini,time_best_track <= tend)

# time forecasted track_exp
time_forec_track_oper = np.asarray([tini + timedelta(hours = float(t)) for t in lead_time_pom_oper])
oktt = [np.where(t == time_forec_track_oper)[0][0] for t in time_best_track[okt]]

lev = np.arange(-9000,9100,100)
    
plt.figure()
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.plot(lon_forec_track_pom_oper[oktt], lat_forec_track_pom_oper[oktt],'X-',color='mediumorchid',\
         markeredgecolor='k',label='POM Oper',markersize=7)
plt.plot(lon_forec_track_pom_exp[oktt], lat_forec_track_pom_exp[oktt],'^-',color='teal',\
         markeredgecolor='k',label='POM Exp',markersize=7)
plt.plot(lon_forec_track_hycom_exp[oktt], lat_forec_track_hycom_exp[oktt],'H-',color='orange',\
         markeredgecolor='k',label='HYCOM Exp',markersize=7)
plt.plot(lon_best_track[okt], lat_best_track[okt],'o-',color='k',label='Best Track')   
plt.legend()
plt.title('Track Forecast Dorian cycle='+ cycle,fontsize=18)
plt.xlim([np.min(lon_forec_track_pom_oper[oktt])-0.5,np.max(lon_forec_track_pom_oper[oktt])+0.5])
plt.ylim([np.min(lat_forec_track_pom_oper[oktt])-0.5,np.max(lat_forec_track_pom_oper[oktt])+0.5])

file = folder_fig + 'best_track_vs_forec_track_POM_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Top 200 m GOFS 3.1 temperature below eye forecasted Dorian track

color_map = cmocean.cm.thermal
       
okm = depth_GOFS <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(time_GOFS,-depth_GOFS,target_temp_eye_GOFS,cmap=color_map,**kw)
plt.contour(time_GOFS,-depth_GOFS,target_temp_eye_GOFS,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Temperature Below Eye ' + 'GOFS 3.1 on ' + ' (cycle= ' + cycle +')',fontsize=14)  

file = folder_fig + ' ' + 'temp_eye_top200_GOFS_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Top 200 m POM oper temperature below eye forecasted Dorian track

color_map = cmocean.cm.thermal
time_matrix = np.tile(time_eye_POM_oper,(target_depth_eye_POM_oper.shape[0],1))
       
okm = depth_GOFS <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(time_matrix,target_depth_eye_POM_oper,target_temp_eye_POM_oper,cmap=color_map,**kw)
plt.contour(time_matrix,target_depth_eye_POM_oper,target_temp_eye_POM_oper,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14) 
#ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Temperature Below Eye '  + 'POM Operational' + ' (cycle= ' + cycle +')',fontsize=14)  
 

file = folder_fig + ' ' + 'folow_eye_temp_top200_POM_oper_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Top 200 m POM exp temperature along forecasted Dorian track

color_map = cmocean.cm.thermal
time_matrix = np.tile(time_eye_POM_exp,(target_depth_eye_POM_exp.shape[0],1))
       
okm = depth_GOFS <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(time_matrix,target_depth_eye_POM_exp,target_temp_eye_POM_exp,cmap=color_map,**kw)
plt.contour(time_matrix,target_depth_eye_POM_exp,target_temp_eye_POM_exp,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14) 
#ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Temperature Below Eye '  + 'POM Experimental' + ' (cycle= ' + cycle +')',fontsize=14)  
 

file = folder_fig + ' ' + 'folow_eye_temp_top200_POM_exp_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Top 200 m HYCOM exp temperature along forecasted Dorian track

color_map = cmocean.cm.thermal
#lat_matrix = np.tile(lat_bnd,(zmatrix_POM_band_exp.shape[0],1))
       
okm = depth_HYCOM_exp <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(time_eye_HYCOM_exp,-depth_HYCOM_exp,target_temp_eye_HYCOM_exp,cmap=color_map,**kw)
plt.contour(time_eye_HYCOM_exp,-depth_HYCOM_exp,target_temp_eye_HYCOM_exp,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
#ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Temperature Below Eye ' + 'HYCOM Experimental ' + ' (cycle= ' + cycle +')',fontsize=14)  
 

file = folder_fig + ' ' + 'temp_below_eye_top200_HYCOM_exp_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% time series temp ML

fig,ax = plt.subplots(figsize=(12, 2))
plt.plot(time_GOFS,Tmean_dens_crit_GOFS,'--o',color='indianred',label='GOFS 3.1')
plt.plot(time_eye_POM_oper,Tmean_dens_crit_POM_oper,'-X',color='mediumpurple',label='POM Oper')
plt.plot(time_eye_POM_exp,Tmean_dens_crit_POM_exp,'-^',color='teal',label='POM Exp')
plt.plot(time_eye_HYCOM_exp,Tmean_dens_crit_HYCOM_exp,'-H',color='orange',label='HYCOM Exp')
plt.ylabel('($^oC$)',fontsize = 14)
#plt.xlabel('Distance Along Track (km)',fontsize = 14)
plt.title('Mixed Layer Temperature Below Dorian Eye ' + ' (cycle= ' + cycle +')',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + 'temp_ml_below_Dorian_eye' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% time series temp OHC

fig,ax = plt.subplots(figsize=(12, 2))
plt.plot(time_GOFS,OHC_GOFS/10**7,'--o',color='indianred',label='GOFS 3.1')
plt.plot(time_eye_POM_oper,OHC_POM_oper/10**7,'-X',color='mediumpurple',label='POM Oper')
plt.plot(time_eye_POM_exp,OHC_POM_exp/10**7,'-^',color='teal',label='POM Exp')
plt.plot(time_eye_HYCOM_exp,OHC_HYCOM_exp/10**7,'-H',color='orange',label='HYCOM Exp')
plt.ylabel('($KJ/cm^2$)',fontsize = 14)
#plt.xlabel('Distance Along Track (km)',fontsize = 14)
plt.title('Ocean Heat Content Below Dorian Eye ' + ' (cycle= ' + cycle +')',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + 'OHC_below_Dorian_eye ' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% time series temp T100

fig,ax = plt.subplots(figsize=(12, 2))
plt.plot(time_GOFS,T100_GOFS,'--o',color='indianred',label='GOFS 3.1')
plt.plot(time_eye_POM_oper,T100_POM_oper,'-X',color='mediumpurple',label='POM Oper')
plt.plot(time_eye_POM_exp,T100_POM_exp,'-^',color='teal',label='POM Exp')
plt.plot(time_eye_HYCOM_exp,T100_HYCOM_exp,'-H',color='orange',label='HYCOM Exp')
plt.ylabel('($^oC$)',fontsize = 14)
#plt.xlabel('Distance Along Track (km)',fontsize = 14)
plt.title('T100 Below Dorian Eye ' + ' (cycle= ' + cycle +')',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + 'T100_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

