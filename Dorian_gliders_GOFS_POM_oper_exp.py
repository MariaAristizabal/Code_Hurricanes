#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:40:28 2020

@author: root
"""

#%% User input

#lon_lim = [-100.0,-55.0]
#lat_lim = [10.0,45.0]

lon_lim = [-85.0,-60.0]
lat_lim = [15.0,35.0]

# Server erddap url IOOS glider dap
server = 'https://data.ioos.us/gliders/erddap'

#gliders sg666, sg665, sg668, silbo
gdata_sg665 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG665-20190718T1155/SG665-20190718T1155.nc3.nc'
gdata_sg666 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG666-20190718T1206/SG666-20190718T1206.nc3.nc'
gdata_sg668 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'

#gliders sg666, sg665, sg668, silbo
url_aoml = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/'

gdata_sg665 = url_aoml+'SG665-20190718T1155/SG665-20190718T1155.nc3.nc'
gdata_sg666 = url_aoml+'SG666-20190718T1206/SG666-20190718T1206.nc3.nc'
gdata_sg668 = url_aoml+'SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
gdata_sg664 = url_aoml+'SG664-20190716T1218/SG664-20190716T1218.nc3.nc'
gdata_sg663 = url_aoml+'SG663-20190716T1159/SG663-20190716T1159.nc3.nc'
gdata_sg667 = url_aoml+'SG667-20190815T1247/SG667-20190815T1247.nc3.nc'

gdata = gdata_sg666

# forecasting cycle to be used
cycle = '2019082800'

#Time window
#date_ini = '2019/08/28/00/00'
#date_end = '2019/09/02/06/00'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# KMZ file
kmz_file_Dorian = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/al052019_best_track-5.kmz'

# url for GOFS 
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Folder where to save figure
folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# folder nc files POM
folder_pom =  '/Volumes/aristizabal/POM_Dorian/'

# folde HWRF2010_HYCOM
folder_hycom = '/Volumes/aristizabal/HWRF2020_HYCOM_Dorian/'

###################

# folder nc files POM
folder_pom_oper = folder_pom + 'POM_Dorian_' + cycle + '_nc_files_oper/'
folder_pom_exp = folder_pom + 'HWRF2020_POM_Dorian/' + 'HWRF2020_POM_Dorian_' + cycle + '_nc_files_exp/'
prefix_pom = 'dorian05l.' + cycle + '.pom.00'

pom_grid_oper = folder_pom_oper + 'dorian05l.' + cycle + '.pom.grid.nc'
pom_grid_exp = folder_pom_exp + 'dorian05l.' + cycle + '.pom.grid.nc'

# Dorian track files
hwrf_pom_track_oper = folder_pom_oper + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'
hwrf_pom_track_exp = folder_pom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'


##################
# folder ab files HYCOM
folder_hycom_exp = folder_hycom + 'HWRF2020_HYCOM_Dorian_' + cycle + '_ab_files_exp/'
prefix_hycom = 'dorian05l.' + cycle + '.hwrf_rtofs_hat10_3z'

#Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HMON-HYCOM_Michael/'
Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HWRF-Hycom-WW3_exp_Michael/'
# RTOFS grid file name
hycom_grid_exp = Dir_HMON_HYCOM + 'hwrf_rtofs_hat10.basin.regional.grid'

# Dorian track files
hwrf_hycom_track_exp = folder_hycom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

'''
# POM output
folder_pom = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/POM_Dorian_npz_files/'    
folder_pom_grid = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/'
pom_grid_oper = folder_pom_grid + 'dorian05l.2019082800.pom.grid.oper.nc'
pom_grid_exp = folder_pom_grid + 'dorian05l.2019082800.pom.grid.exp.nc'
'''

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from netCDF4 import Dataset
import cmocean
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
import seawater as sw
import os
import os.path
import glob 
from bs4 import BeautifulSoup
from zipfile import ZipFile

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')
from read_glider_data import read_glider_data_thredds_server
#from process_glider_data import grid_glider_data_thredd

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

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

#%% Function Getting glider transect from GOFS
    
def get_glider_transect_from_GOFS(depth_GOFS,oktime_GOFS):
    
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

#%%  Function Getting glider transect from POM

def get_glider_transect_from_POM(folder,prefix,zlev,zmatrix_pom,lon_pom,lat_pom,tstamp_glider,long,latg):

    ncfiles = sorted(glob.glob(os.path.join(folder,prefix+'*.nc')))

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
        year = int(file.split('.')[1][0:4])
        month = int(file.split('.')[1][4:6])
        day = int(file.split('.')[1][6:8])
        hour = int(file.split('.')[1][8:10])
        dt = int(file.split('.')[3][1:])
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

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the 
# mixed layer and td: temp at 1 meter below the mixed layer          

def MLD_temp_and_dens_criteria(dt,drho,time,depth,temp,salt,dens):

    MLD_temp_crit = np.empty(len(time)) 
    MLD_temp_crit[:] = np.nan
    Tmean_temp_crit = np.empty(len(time)) 
    Tmean_temp_crit[:] = np.nan
    Smean_temp_crit = np.empty(len(time)) 
    Smean_temp_crit[:] = np.nan
    Td_temp_crit = np.empty(len(time)) 
    Td_temp_crit[:] = np.nan
    MLD_dens_crit = np.empty(len(time)) 
    MLD_dens_crit[:] = np.nan
    Tmean_dens_crit = np.empty(len(time)) 
    Tmean_dens_crit[:] = np.nan
    Smean_dens_crit = np.empty(len(time)) 
    Smean_dens_crit[:] = np.nan
    Td_dens_crit = np.empty(len(time)) 
    Td_dens_crit[:] = np.nan
    for t,tt in enumerate(time):
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

def OHC_surface(time,temp,depth,dens):
    cp = 3985 #Heat capacity in J/(kg K)

    OHC = np.empty((len(time)))
    OHC[:] = np.nan
    for t,tt in enumerate(time):
        ok26 = temp[:,t] >= 26
        if len(depth[ok26]) != 0:
            if np.nanmin(np.abs(depth[ok26]))>10:
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

#%% Read POM grid

print('Retrieving coordinates from POM')
POM_grid_oper = xr.open_dataset(pom_grid_oper,decode_times=False)
lon_pom_oper = np.asarray(POM_grid_oper['east_e'][:])
lat_pom_oper = np.asarray(POM_grid_oper['north_e'][:])
zlev_pom_oper = np.asarray(POM_grid_oper['zz'][:])
hpom_oper = np.asarray(POM_grid_oper['h'][:])
zmatrix = np.dot(hpom_oper.reshape(-1,1),zlev_pom_oper.reshape(1,-1))
zmatrix_pom_oper = zmatrix.reshape(hpom_oper.shape[0],hpom_oper.shape[1],zlev_pom_oper.shape[0])

POM_grid_exp = xr.open_dataset(pom_grid_exp,decode_times=False)
lon_pom_exp = np.asarray(POM_grid_exp['east_e'][:])
lat_pom_exp = np.asarray(POM_grid_exp['north_e'][:])
zlev_pom_exp = np.asarray(POM_grid_exp['zz'][:])
hpom_exp = np.asarray(POM_grid_exp['h'][:])
zmatrix = np.dot(hpom_exp.reshape(-1,1),zlev_pom_exp.reshape(1,-1))
zmatrix_pom_exp = zmatrix.reshape(hpom_exp.shape[0],hpom_exp.shape[1],zlev_pom_exp.shape[0])   

#%% Read GOFS 3.1 grid

#Time window
date_ini = cycle[0:4]+'/'+cycle[4:6]+'/'+cycle[6:8]+'/'+cycle[8:]+'/00/00'
tini = datetime.strptime(date_ini,'%Y/%m/%d/%H/%M/%S')
tend = tini + timedelta(hours=120)
date_end = tend.strftime('%Y/%m/%d/%H/%M/%S')

print('Retrieving coordinates from GOFS')
GOFS = xr.open_dataset(url_GOFS,decode_times=False) 

tt_G = GOFS.time
t_G = netCDF4.num2date(tt_G[:],tt_G.units) 

tmin = datetime.strptime(date_ini[0:-6],'%Y/%m/%d/%H')
tmax = datetime.strptime(date_end[0:-6],'%Y/%m/%d/%H')
oktime_GOFS = np.where(np.logical_and(t_G >= tmin, t_G <= tmax)) 
time_GOFS = np.asarray(t_G[oktime_GOFS])
timestamp_GOFS = mdates.date2num(time_GOFS)

lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:])

# Conversion from glider longitude and latitude to GOFS convention
lon_limG, lat_limG = glider_coor_to_GOFS_coord(lon_lim,lat_lim)

oklat_GOFS = np.where(np.logical_and(lat_G >= lat_limG[0], lat_G <= lat_limG[1])) 
oklon_GOFS = np.where(np.logical_and(lon_G >= lon_limG[0], lon_G <= lon_limG[1])) 

lat_GOFS = lat_G[oklat_GOFS]
lon_GOFS = lon_G[oklon_GOFS]

depth_GOFS = np.asarray(GOFS.depth[:])

# Conversion from GOFS longitude and latitude to glider convention
lon_GOFSg, lat_GOFSg = GOFS_coor_to_glider_coord(lon_GOFS,lat_GOFS)

#%% Reading HYCOM grid

# Reading lat and lon
lines_grid = [line.rstrip() for line in open(hycom_grid_exp+'.b')]
lon_hycom = np.array(readgrids(hycom_grid_exp,'plon:',[0]))
lat_hycom = np.array(readgrids(hycom_grid_exp,'plat:',[0]))

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

#%% Reading glider data
    
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

#%% Getting glider transect from GOFS 

# Conversion from glider longitude and latitude to GOFS convention
target_lon, target_lat = glider_coor_to_GOFS_coord(long,latg)

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_model = [mdates.date2num(time_GOFS[i]) for i in np.arange(len(time_GOFS))]

# interpolating glider lon and lat to lat and lon on model time
sublon_GOFS = np.interp(tstamp_model,tstamp_glider,target_lon)
sublat_GOFS = np.interp(tstamp_model,tstamp_glider,target_lat)

# Conversion from GOFS convention to glider longitude and latitude
sublon_GOFSg,sublat_GOFSg = GOFS_coor_to_glider_coord(sublon_GOFS,sublat_GOFS)

# getting the model grid positions for sublonm and sublatm
oklon_GOFS = np.round(np.interp(sublon_GOFS,lon_G,np.arange(len(lon_G)))).astype(int)
oklat_GOFS = np.round(np.interp(sublat_GOFS,lat_G,np.arange(len(lat_G)))).astype(int)
    
# Getting glider transect from model
target_temp_GOFS, target_salt_GOFS = \
                          get_glider_transect_from_GOFS(depth_GOFS,oktime_GOFS) 
                          
#%% Calculate density for GOFS

target_dens_GOFS = sw.dens(target_salt_GOFS,target_temp_GOFS,np.tile(depth_GOFS,(len(time_GOFS),1)).T) 
    
#%% Retrieve glider transect from POM operational

tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]             

folder_pom = folder_pom_oper
prefix = prefix_pom
zlev = zlev_pom_oper
zmatrix_pom = zmatrix_pom_oper
lon_pom = lon_pom_oper
lat_pom = lat_pom_oper
tstamp_glider = tstamp_glider
long = long
latg = latg

time_POM_oper, target_temp_POM_oper, target_salt_POM_oper, \
    target_dens_POM_oper, target_depth_POM_oper = \
    get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_pom,lon_pom,lat_pom,\
                                 tstamp_glider,long,latg)
        
timestamp_POM_oper = mdates.date2num(time_POM_oper)

#%% Retrieve glider transect from POM experimental

tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]             

folder_pom = folder_pom_exp
prefix = prefix_pom
zlev = zlev_pom_exp
zmatrix_pom = zmatrix_pom_exp
lon_pom = lon_pom_exp
lat_pom = lat_pom_exp
tstamp_glider = tstamp_glider
long = long
latg = latg
   
time_POM_exp, target_temp_POM_exp, target_salt_POM_exp,\
    target_dens_POM_exp, target_depth_POM_exp = \
    get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_pom,lon_pom,lat_pom,\
                                 tstamp_glider,long,latg)
timestamp_POM_exp = mdates.date2num(time_POM_exp)

#%% Get glider transect from HYCOM

folder_hycom = folder_hycom_exp
prefix = prefix_hycom
    
# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]

# Conversion from glider longitude and latitude to GOFS convention
target_lonG, target_latG = glider_coor_to_GOFS_coord(long,latg)

lon_glider = target_lonG 
lat_glider = target_latG

var = 'temp'
target_temp_HYCOM_exp, time_HYCOM_exp = \
    get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,\
    lon_hycom,lat_hycom,var,tstamp_glider,lon_glider,lat_glider)

var = 'salinity'
target_salt_HYCOM_exp, _ = \
    get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,\
      lon_hycom,lat_hycom,var,tstamp_glider,lon_glider,lat_glider)
        
#%% Calculate density for HYCOM

target_dens_HYCOM_exp = sw.dens(target_salt_HYCOM_exp,target_temp_HYCOM_exp,np.tile(depth_HYCOM_exp,(len(time_HYCOM_exp),1)).T) 
    
#%% Calculation of mixed layer depth based on temperature and density critria
# Tmean: mean temp within the mixed layer and 
# td: temp at 1 meter below the mixed layer            

dt = 0.2
drho = 0.125

# for glider data
MLD_temp_crit_glid, _, _, _, MLD_dens_crit_glid, Tmean_dens_crit_glid, Smean_dens_crit_glid, _ = \
MLD_temp_and_dens_criteria(dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)

# for GOFS 3.1 output 
MLD_temp_crit_GOFS, _, _, _, MLD_dens_crit_GOFS, Tmean_dens_crit_GOFS, Smean_dens_crit_GOFS, _ = \
MLD_temp_and_dens_criteria(dt,drho,time_GOFS,depth_GOFS,target_temp_GOFS,target_salt_GOFS,target_dens_GOFS)          

# for POM operational
MLD_temp_crit_POM_oper, _, _, _, MLD_dens_crit_POM_oper, Tmean_dens_crit_POM_oper, Smean_dens_crit_POM_oper, _ = \
MLD_temp_and_dens_criteria(dt,drho,timestamp_POM_oper,target_depth_POM_oper,target_temp_POM_oper,target_salt_POM_oper,target_dens_POM_oper)

# for POM experimental
MLD_temp_crit_POM_exp, _, _, _, MLD_dens_crit_POM_exp, Tmean_dens_crit_POM_exp, Smean_dens_crit_POM_exp, _ = \
MLD_temp_and_dens_criteria(dt,drho,timestamp_POM_exp,target_depth_POM_exp,target_temp_POM_exp,target_salt_POM_exp,target_dens_POM_exp)    

# for HYCOM experimental
timestamp_HYCOM_exp = mdates.date2num(time_HYCOM_exp)
MLD_temp_crit_HYCOM_exp, _, _, _, MLD_dens_crit_HYCOM_exp, Tmean_dens_crit_HYCOM_exp, Smean_dens_crit_HYCOM_exp, _ = \
MLD_temp_and_dens_criteria(dt,drho,timestamp_HYCOM_exp,depth_HYCOM_exp,target_temp_HYCOM_exp,target_salt_HYCOM_exp,target_dens_HYCOM_exp)

#%% Surface Ocean Heat Content

# glider
OHC_glid = OHC_surface(timeg,tempg_gridded,depthg_gridded,densg_gridded)

# GOFS
OHC_GOFS = OHC_surface(time_GOFS,target_temp_GOFS,depth_GOFS,target_dens_GOFS)

# POM operational    
OHC_POM_oper = OHC_surface(timestamp_POM_oper,target_temp_POM_oper,target_depth_POM_oper,target_dens_POM_oper)

# POM experimental
OHC_POM_exp = OHC_surface(timestamp_POM_exp,target_temp_POM_exp,target_depth_POM_exp,target_dens_POM_exp)

# HYCOM experimental
OHC_HYCOM_exp = OHC_surface(timestamp_HYCOM_exp,target_temp_HYCOM_exp,depth_HYCOM_exp,target_dens_HYCOM_exp)

#%% Calculate T100

# glider
T100_glid = depth_aver_top_100(depthg_gridded,tempg_gridded)

# GOFS
T100_GOFS = depth_aver_top_100(depth_GOFS,target_temp_GOFS)

# POM operational
T100_POM_oper = depth_aver_top_100(target_depth_POM_oper,target_temp_POM_oper) 

# POM experimental
T100_POM_exp = depth_aver_top_100(target_depth_POM_exp,target_temp_POM_exp)  

# HYCOM experimental
T100_HYCOM_exp = depth_aver_top_100(depth_HYCOM_exp,target_temp_HYCOM_exp)

#%% Get Dorian track from POM

lon_forec_track_pom_oper, lat_forec_track_pom_oper, lead_time_pom_oper = get_storm_track_POM(hwrf_pom_track_oper)

lon_forec_track_pom_exp, lat_forec_track_pom_exp, lead_time_pom_exp = get_storm_track_POM(hwrf_pom_track_exp)

lon_forec_track_hycom_exp, lat_forec_track_hycom_exp, lead_time_hycom_exp = get_storm_track_POM(hwrf_hycom_track_exp)

#%% Get Dorian best track 

lon_best_track, lat_best_track, time_best_track, _, _, _ = \
read_kmz_file_storm_best_track(kmz_file_Dorian)

#%% Figure transets

def figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                    time,tini,tend,depth,max_depth,color_map):

    if depth.ndim == 1:
        time_matrix = time
    else:
        time_matrix = np.tile(time,(depth.shape[0],1))
    
    kw = dict(levels = np.linspace(min_var1,max_var1,nlevels))
     
    # plot
    fig, ax = plt.subplots(figsize=(12, 2))      
    cs = plt.contourf(time_matrix,depth,var1,cmap=color_map,**kw)
    plt.contour(time_matrix,depth,var1,[26],colors='k')
    plt.plot(time,var2,'-',label='MLD dt',color='indianred',linewidth=2 )
    plt.plot(time,var3,'-',label='MLD drho',color='darkgreen',marker='.',\
             markeredgecolor='k',linewidth=2 )
    
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
            
    ax.set_ylim(-200, 0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    xticks = [tini+nday*timedelta(1) for nday in np.arange(14)]
    xticks = np.asarray(xticks)
    plt.xticks(xticks)
    xfmt = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_formatter(xfmt)
    plt.legend()  
    ax.set_xlim(tini,tend)

#%% Top 200 m glider temperature from 2019/08/28/00

color_map = cmocean.cm.thermal

var1 = tempg_gridded
var2 = -MLD_temp_crit_glid
var3 = -MLD_dens_crit_glid
time = timeg
depth = -depthg_gridded
max_depth = -200
min_var1 = 19
max_var1 = 31
nlevels = max_var1 - min_var1 + 1
tini = tini
tend = tend
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + inst_id,fontsize=14)
file = folder_fig + ' ' + 'along_track_temp_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m GOFS 3.1 temperature from 2019/08/28/00

color_map = cmocean.cm.thermal

var1 = target_temp_GOFS
var2 = -MLD_temp_crit_GOFS
var3 = -MLD_dens_crit_GOFS
time = mdates.date2num(time_GOFS)
depth = -depth_GOFS
max_depth = -200
min_var1 = 19
max_var2 = 31
nlevels = max_var1 - min_var1 + 1
tini = tini
tend = tend

figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + 'GOFS 3.1',fontsize=14)
file = folder_fig + ' ' + 'along_track_temp_top200_GOFS_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m POM operational temperature from 2019/08/28/00

color_map = cmocean.cm.thermal

var1 = target_temp_POM_oper
var2 = MLD_temp_crit_POM_oper
var3 = MLD_dens_crit_POM_oper
time = mdates.date2num(time_POM_oper)
depth = target_depth_POM_oper
max_depth = -200
min_var1 = 19
max_var1 = 31
nlevels = max_var1 - min_var1 + 1
tini = tini
tend = tend

figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track Temperature Profile HWRF-POM Operational',fontsize=14)
file = folder_fig + ' ' + 'along_track_temp_top200_POM_oper_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m POM experimental temperature from 2019/08/28/00

color_map = cmocean.cm.thermal

var1 = target_temp_POM_exp
var2 = MLD_temp_crit_POM_exp
var3 = MLD_dens_crit_POM_exp
time = mdates.date2num(time_POM_exp)
depth = target_depth_POM_exp
max_depth = -200
min_var1 = 19
max_var1 = 31
nlevels = max_var1 - min_var1 + 1
tini = tini
tend = tend

figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track Temperature Profile HWRF-POM Experimental',fontsize=14)
file = folder_fig + ' ' + 'along_track_temp_top200_POM_exp_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m HYCOM experimental temperature from 2019/08/28/00

color_map = cmocean.cm.thermal

var1 = target_temp_HYCOM_exp
var2 = -MLD_temp_crit_HYCOM_exp
var3 = -MLD_dens_crit_HYCOM_exp
time = mdates.date2num(time_HYCOM_exp)
depth = -depth_HYCOM_exp
max_depth = -200
min_var1 = 19
max_var1 = 31
nlevels = max_var1 - min_var1 + 1
tini = tini
tend = tend

figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track Temperature Profile HWRF-HYCOM Experimental',fontsize=14)
file = folder_fig + ' ' + 'along_track_temp_top200_HYCOM_exp_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m glider salinity from 2019/08/28/00

color_map = cmocean.cm.haline

var1 = saltg_gridded
var2 = -MLD_temp_crit_glid
var3 = -MLD_dens_crit_glid
time = timeg
depth = -depthg_gridded
max_depth = -200
min_var1 = 35.5
max_var1 = 37.3
nlevels = 19 #np.round((max_var1 - min_var1)*10+1)
tini = tini
tend = tend
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + inst_id,fontsize=14)
file = folder_fig + ' ' + 'along_track_salt_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m GOFS salinity from 2019/08/28/00

color_map = cmocean.cm.haline

var1 = target_salt_GOFS
var2 = -MLD_temp_crit_GOFS
var3 = -MLD_dens_crit_GOFS
time = mdates.date2num(time_GOFS)
depth = -depth_GOFS
max_depth = -200
min_var1 = 35.5
max_var1 = 37.3
nlevels = 19
tini = tini
tend = tend
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + 'GOFS 3.1',fontsize=14)
file = folder_fig + ' ' + 'along_track_salt_top200_GOFS_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m POM oper salinity from 2019/08/28/00

color_map = cmocean.cm.haline

var1 = target_salt_POM_oper
var2 = MLD_temp_crit_POM_oper
var3 = MLD_dens_crit_POM_oper
time = mdates.date2num(time_POM_oper)
depth = target_depth_POM_oper
max_depth = -200
min_var1 = 35.5
max_var1 = 37.3
nlevels = 19
tini = tini
tend = tend
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track Salinity Profile HWRF-POM Operational',fontsize=14)
file = folder_fig + ' ' + 'along_track_salt_top200_POM_oper_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m POM exp salinity from 2019/08/28/00

color_map = cmocean.cm.haline

var1 = target_salt_POM_exp
var2 = MLD_temp_crit_POM_exp
var3 = MLD_dens_crit_POM_exp
time = mdates.date2num(time_POM_exp)
depth = target_depth_POM_exp
max_depth = -200
min_var1 = 35.5
max_var1 = 37.3
nlevels = 19
tini = tini
tend = tend
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track Salinity Profile HWRF-POM Experimental',fontsize=14)
file = folder_fig + ' ' + 'along_track_salt_top200_POM_exp_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m HYCOM exp salinity from 2019/08/28/00

color_map = cmocean.cm.haline

var1 = target_salt_HYCOM_exp
var2 = -MLD_temp_crit_HYCOM_exp
var3 = -MLD_dens_crit_HYCOM_exp
time = mdates.date2num(time_HYCOM_exp)
depth = -depth_HYCOM_exp
max_depth = -200
min_var1 = 35.5
max_var1 = 37.3
nlevels = 19
tini = tini
tend = tend
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track Salinity Profile HWRF-HYCOM Experimental',fontsize=14)
file = folder_fig + ' ' + 'along_track_salt_top200_HYCOM_exp_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m HYCOM exp salinity from 2019/08/28/00

color_map = cmocean.cm.haline

var1 = target_salt_HYCOM_exp
var2 = MLD_temp_crit_HYCOM_exp
var3 = MLD_dens_crit_HYCOM_exp
time = mdates.date2num(time_HYCOM_exp)
depth = depth_HYCOM_exp
max_depth = -200
min_var1 = 35.5
max_var1 = 37.3
nlevels = 19
tini = tini
tend = tend
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,\
                time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track Salinity Profile HWRF-HYCOM Experimental',fontsize=14)
file = folder_fig + ' ' + 'along_track_salt_top200_POM_exp_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Vertican profile glider, GOFS 3.1, POM , HYCOM During Dorian

tDorian = datetime(2019,8,28,18)
okg = np.where(timeg < tDorian)[0][-1]
okgofs = np.where(time_GOFS == tDorian )[0][0] #tdorian
okpom_oper = np.where(timestamp_POM_oper == mdates.date2num(tDorian))[0][0] #tdorian
okpom_exp = np.where(timestamp_POM_exp == mdates.date2num(tDorian))[0][0]
okhycom_exp = np.where(timestamp_HYCOM_exp == mdates.date2num(tDorian))[0][0]
    
plt.figure(figsize=(4,7))
plt.plot(tempg[:,okg],-depthg[:,okg],'-',color='royalblue',linewidth=4,label=inst_id.split('-')[0]) 
plt.plot(target_temp_GOFS[:,okgofs],-depth_GOFS,'s-',color='indianred',linewidth=2,label='GOFS 3.1')
plt.plot(target_temp_POM_oper[:,okpom_oper],target_depth_POM_oper[:,okpom_oper],'X-',color='mediumpurple',linewidth=2,label='POM Oper')
plt.plot(target_temp_POM_exp[:,okpom_exp],target_depth_POM_exp[:,okpom_exp],'^-',color='teal',linewidth=2,label='POM Exp')
plt.plot(target_temp_HYCOM_exp[:,okhycom_exp],-depth_HYCOM_exp,'H-',color='darkorange',linewidth=2,label='HYCOM Exp')
plt.ylim([-200,0])
plt.xlim([20,30])
plt.title('Temperature on '+str(timeg[okg])[0:13],fontsize=16)
plt.ylabel('Depth (m)')
plt.xlabel('($^oC$)')
plt.legend()

file = folder_fig + ' ' + 'temp_profile_glider_GOFS_POM_' + inst_id.split('-')[0]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% OHC figure
    
oktimeg_gofs = np.round(np.interp(tstamp_model,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)
#OHCg_to31 = OHCg[oktimeg_gofs]

oktimeg_pom_oper = np.round(np.interp(timestamp_POM_oper,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)    
#OHCg_to_pom_oper = OHCg[oktimeg_pom_oper] 

fig,ax = plt.subplots(figsize=(12, 2.8))
plt.plot(timeg,OHC_glid*10**-7,'-o',color='royalblue',label=inst_id.split('-')[0],linewidth=3)
plt.plot(time_GOFS,OHC_GOFS*10**-7,'--s',color='indianred',label='GOFS 3.1')
plt.plot(timestamp_POM_oper,OHC_POM_oper*10**-7,'-X',color='mediumpurple',label='POM Oper')
plt.plot(timestamp_POM_exp,OHC_POM_exp*10**-7,'-^',color='teal',label='POM Exp')
plt.plot(timestamp_HYCOM_exp,OHC_HYCOM_exp*10**-7,'-H',color='darkorange',label='HYCOM Exp')
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([time_POM_oper[0],time_POM_oper[-1]])
plt.ylabel('($KJ/cm^2$)',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(55,90)))# ng665,ng666
plt.plot(tDorian,np.arange(55,90),'--k')
plt.title('Ocean Heat Content',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + ' ' + inst_id + '_OHC'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Salt ML
    
oktimeg_gofs = np.round(np.interp(tstamp_model,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)
#OHCg_to31 = OHCg[oktimeg_gofs]

oktimeg_pom_oper = np.round(np.interp(timestamp_POM_oper,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)    
#OHCg_to_pom_oper = OHCg[oktimeg_pom_oper] 

fig,ax = plt.subplots(figsize=(12, 2.8))
plt.plot(timeg,Smean_dens_crit_glid,'-o',color='royalblue',label=inst_id.split('-')[0],linewidth=3)
plt.plot(time_GOFS,Smean_dens_crit_GOFS,'--s',color='indianred',label='GOFS 3.1')
plt.plot(timestamp_POM_oper,Smean_dens_crit_POM_oper,'-X',color='mediumpurple',label='POM Oper')
plt.plot(timestamp_POM_exp,Smean_dens_crit_POM_exp,'-^',color='teal',label='POM Exp')
plt.plot(timestamp_HYCOM_exp,Smean_dens_crit_HYCOM_exp,'-H',color='darkorange',label='HYCOM Exp')

t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([time_POM_oper[0],time_POM_oper[-1]])
plt.ylabel('($^oC$)',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(34.5,35.6,0.2)))# ng665,ng666
plt.plot(tDorian,np.arange(34.5,35.6,0.2),'--k')
plt.title('Mixed Layer Salinity',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + ' ' + inst_id + '_salt_ml'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Temp ML
    
oktimeg_gofs = np.round(np.interp(tstamp_model,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)
#OHCg_to31 = OHCg[oktimeg_gofs]

oktimeg_pom_oper = np.round(np.interp(timestamp_POM_oper,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)    
#OHCg_to_pom_oper = OHCg[oktimeg_pom_oper] 

fig,ax = plt.subplots(figsize=(12, 2.8))
plt.plot(timeg,Tmean_dens_crit_glid,'-o',color='royalblue',label=inst_id.split('-')[0],linewidth=3)
plt.plot(time_GOFS,Tmean_dens_crit_GOFS,'--s',color='indianred',label='GOFS 3.1')
plt.plot(timestamp_POM_oper,Tmean_dens_crit_POM_oper,'-X',color='mediumpurple',label='POM Oper')
plt.plot(timestamp_POM_exp,Tmean_dens_crit_POM_exp,'-^',color='teal',label='POM Exp')
plt.plot(timestamp_HYCOM_exp,Tmean_dens_crit_HYCOM_exp,'-^',color='darkorange',label='HYCOM Exp')
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([time_POM_oper[0],time_POM_oper[-1]])
plt.ylabel('($^oC$)',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(28,29.5,0.2)))# ng665,ng666
plt.plot(tDorian,np.arange(28,29.5,0.2),'--k')
plt.title('Mixed Layer Temperature',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + ' ' + inst_id + '_temp_ml'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% T100
    
oktimeg_gofs = np.round(np.interp(tstamp_model,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)
#OHCg_to31 = OHCg[oktimeg_gofs]

oktimeg_pom_oper = np.round(np.interp(timestamp_POM_oper,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)    
#OHCg_to_pom_oper = OHCg[oktimeg_pom_oper] 

fig,ax = plt.subplots(figsize=(12, 2.8))
plt.plot(timeg,T100_glid,'-o',color='royalblue',label=inst_id.split('-')[0],linewidth=3)
plt.plot(time_GOFS,T100_GOFS,'--s',color='indianred',label='GOFS 3.1')
plt.plot(timestamp_POM_oper,T100_POM_oper,'-X',color='mediumpurple',label='POM Oper')
plt.plot(timestamp_POM_exp,T100_POM_exp,'-^',color='teal',label='POM Exp')
plt.plot(timestamp_HYCOM_exp,T100_HYCOM_exp,'-^',color='darkorange',label='HYCOM Exp')
t0 = datetime(2019,8,25)
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([time_POM_oper[0],time_POM_oper[-1]])
plt.ylim([27.6,28.6])
plt.ylabel('($^oC$)',fontsize = 14)
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(27.6,28.7,0.1)))# ng665,ng666
plt.plot(tDorian,np.arange(27.6,28.7,0.1),'--k')
plt.title('T100',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + ' ' + inst_id + '_T100'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  


#%% Vertical profile temperature before and after Dorian
'''
tdorian = datetime(2019,8,28,18)
bef = timeg <= tdorian
aft = np.logical_and(timeg > tdorian, timeg <= datetime(2019,8,29,18))

plt.figure()
plt.plot(np.nanmean(tempg_gridded[:,bef],1),-depthg_gridded,'.-',color='indianred',label='18 hours Before')
plt.plot(np.nanmean(tempg_gridded[:,aft],1),-depthg_gridded,'.-',color='slateblue',label='1 day After')
plt.plot(tempg_gridded[:,bef],-depthg_gridded,'-',color='indianred',alpha=0.2)
plt.plot(tempg_gridded[:,aft],-depthg_gridded,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Profile before and after Dorian '+ inst_id.split('-')[0],size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Temperature ($^oC$)',size=14)
'''

tdorian = datetime(2019,8,28,18)
bef = timeg <= tdorian
aft = np.where(np.logical_and(timeg > tdorian, timeg <= datetime(2019,8,29,16)))[0]

plt.figure()
plt.plot(tempg_gridded[:,0],-depthg_gridded,'.-',color='indianred',label='18 hours Before')
plt.plot(tempg_gridded[:,aft[-1]],-depthg_gridded,'.-',color='slateblue',label='21 hour After')
plt.plot(np.arange(26,30,0.1),np.tile(-MLD_dens_crit_glid[0],len(np.arange(26,30,0.1))),'--',color='indianred')
plt.plot(np.arange(26,30,0.1),np.tile(-MLD_dens_crit_glid[aft[-1]],len(np.arange(26,30,0.1))),'--',color='slateblue')
plt.plot(tempg_gridded[:,bef],-depthg_gridded,'-',color='indianred',alpha=0.2)
plt.plot(tempg_gridded[:,aft],-depthg_gridded,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Profile before and after Dorian '+ inst_id.split('-')[0],size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Temperature ($^oC$)',size=14)

file = folder_fig + ' ' + inst_id + 'Temp_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

file = folder_fig + ' ' + inst_id + 'Temp_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Vertical profile temperature GOFS before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = time_GOFS <= tdorian
aft = np.where(np.logical_and(time_GOFS > tdorian, time_GOFS <= datetime(2019,8,29,16)))[0]

plt.figure()
plt.plot(target_temp_GOFS[:,0],-depth_GOFS,'.-',color='indianred',label='18 hours Before')
plt.plot(target_temp_GOFS[:,aft[-1]],-depth_GOFS,'.-',color='slateblue',label='21 hours After')
plt.plot(np.arange(26,30,0.1),np.tile(-MLD_dens_crit_GOFS[0],len(np.arange(26,30,0.1))),'--',color='indianred')
plt.plot(np.arange(26,30,0.1),np.tile(-MLD_dens_crit_GOFS[aft[-1]],len(np.arange(26,30,0.1))),'--',color='slateblue')
plt.plot(target_temp_GOFS[:,bef],-depth_GOFS,'-',color='indianred',alpha=0.2)
plt.plot(target_temp_GOFS[:,aft],-depth_GOFS,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Profile before and after Dorian GOFS',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Temperature ($^oC$)',size=14)

file = folder_fig + 'GOFS_Temp_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Vertical profile temperature POM oper before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timestamp_POM_oper <= mdates.date2num(tdorian)
aft = np.where(np.logical_and(timestamp_POM_oper > mdates.date2num(tdorian),\
                     timestamp_POM_oper <= mdates.date2num(datetime(2019,8,29,16))))[0]

plt.figure()
plt.plot(target_temp_POM_oper[:,0],target_depth_POM_oper[:,0],'.-',color='indianred',label='18 hours Before')
plt.plot(target_temp_POM_oper[:,aft[-1]],target_depth_POM_oper[:,aft[-1]],'.-',color='slateblue',label='21 hours After')
plt.plot(np.arange(26,30,0.1),np.tile(MLD_dens_crit_POM_oper[0],len(np.arange(26,30,0.1))),'--',color='indianred')
plt.plot(np.arange(26,30,0.1),np.tile(MLD_dens_crit_POM_oper[aft[-1]],len(np.arange(26,30,0.1))),'--',color='slateblue')
plt.plot(target_temp_POM_oper[:,bef],target_depth_POM_oper[:,bef],'-',color='indianred',alpha=0.2)
plt.plot(target_temp_POM_oper[:,aft],target_depth_POM_oper[:,bef],'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Profile before and after Dorian POM Oper',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Temperature ($^oC$)',size=14)

file = folder_fig + 'POM_oper_Temp_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Vertical profile temperature POM exp and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timestamp_POM_exp <= mdates.date2num(tdorian)
aft = np.where(np.logical_and(timestamp_POM_exp > mdates.date2num(tdorian),\
                     timestamp_POM_exp <= mdates.date2num(datetime(2019,8,29,16))))[0]

plt.figure()
plt.plot(target_temp_POM_exp[:,0],target_depth_POM_exp[:,0],'.-',color='indianred',label='18 hours Before')
plt.plot(target_temp_POM_exp[:,aft[-1]],target_depth_POM_exp[:,aft[-1]],'.-',color='slateblue',label='1 day After')
plt.plot(np.arange(26,30,0.1),np.tile(MLD_dens_crit_POM_exp[0],len(np.arange(26,30,0.1))),'--',color='indianred')
plt.plot(np.arange(26,30,0.1),np.tile(MLD_dens_crit_POM_exp[aft[-1]],len(np.arange(26,30,0.1))),'--',color='slateblue')
plt.plot(target_temp_POM_exp[:,bef],target_depth_POM_exp[:,bef],'-',color='indianred',alpha=0.2)
plt.plot(target_temp_POM_exp[:,aft],target_depth_POM_exp[:,bef],'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Profile before and after Dorian POM Exp',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Temperature ($^oC$)',size=14)

file = folder_fig + 'POM_exp_Temp_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Vertical profile temperature HYCOM exp before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timestamp_HYCOM_exp <= mdates.date2num(tdorian)
aft = np.where(np.logical_and(timestamp_HYCOM_exp > mdates.date2num(tdorian),\
                     timestamp_HYCOM_exp <= mdates.date2num(datetime(2019,8,29,16))))[0]

plt.figure()
plt.plot(target_temp_HYCOM_exp[:,0],-depth_HYCOM_exp,'.-',color='indianred',label='18 hours Before')
plt.plot(target_temp_HYCOM_exp[:,aft[-1]],-depth_HYCOM_exp,'.-',color='slateblue',label='1 day After')
plt.plot(np.arange(26,30,0.05),np.tile(-MLD_dens_crit_HYCOM_exp[0],len(np.arange(26,30,0.05))),'--',color='indianred')
plt.plot(np.arange(26,30,0.05),np.tile(-MLD_dens_crit_HYCOM_exp[aft[-1]],len(np.arange(26,30,0.05))),'--',color='slateblue')
plt.plot(target_temp_HYCOM_exp[:,bef],-depth_HYCOM_exp,'-',color='indianred',alpha=0.2)
plt.plot(target_temp_HYCOM_exp[:,aft],-depth_HYCOM_exp,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Profile before and after Dorian HYCOM Exp',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Temperature ($^oC$)',size=14)

file = folder_fig + 'HYCOM_exp_temp_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Vertical profile salinity before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timeg <= tdorian
aft = np.where(np.logical_and(timeg > tdorian, timeg <= datetime(2019,8,29,16)))[0]

plt.figure()
plt.plot(saltg_gridded[:,0],-depthg_gridded,'.-',color='indianred',label='18 hours Before')
plt.plot(saltg_gridded[:,aft[-1]],-depthg_gridded,'.-',color='slateblue',label='21 hours After')
plt.plot(np.arange(35,37,0.1),np.tile(-MLD_dens_crit_glid[0],len(np.arange(35,37,0.1))),'--',color='indianred')
plt.plot(np.arange(35,37,0.1),np.tile(-MLD_dens_crit_glid[aft[-1]],len(np.arange(35,37,0.1))),'--',color='slateblue')
plt.plot(saltg_gridded[:,bef],-depthg_gridded,'-',color='indianred',alpha=0.2)
plt.plot(saltg_gridded[:,aft],-depthg_gridded,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
#plt.xlim([34.5,37.5])
plt.xlim([35,37])
#plt.xlim([34,37])
plt.title('Profile before and after Dorian '+ inst_id.split('-')[0],size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Salinity',size=14)

file = folder_fig + ' ' + inst_id + 'Sal_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Vertical profile salinity GOFS before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = time_GOFS <= tdorian
aft = np.where(np.logical_and(time_GOFS > tdorian, time_GOFS <= datetime(2019,8,29,18)))[0]

plt.figure()
plt.plot(target_salt_GOFS[:,0],-depth_GOFS,'.-',color='indianred',label='18 hours Before')
plt.plot(target_salt_GOFS[:,aft[-1]],-depth_GOFS,'.-',color='slateblue',label='21 hours After')
plt.plot(np.arange(35,37,0.05),np.tile(-MLD_dens_crit_GOFS[0],len(np.arange(35,37,0.05))),'--',color='indianred')
plt.plot(np.arange(35,37,0.05),np.tile(-MLD_dens_crit_GOFS[aft[-1]],len(np.arange(35,37,0.05))),'--',color='slateblue')
plt.plot(target_salt_GOFS[:,bef],-depth_GOFS,'-',color='indianred',alpha=0.2)
plt.plot(target_salt_GOFS[:,aft],-depth_GOFS,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
#plt.xlim([34.5,37.5])
plt.xlim([35,37])
#plt.xlim([34,37])
plt.title('Profile before and after Dorian GOFS',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Salinity',size=14)

file = folder_fig + 'GOFS_Salt_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
 

#%% Vertical profile salinity POM oper before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timestamp_POM_oper <= mdates.date2num(tdorian)
aft = np.where(np.logical_and(timestamp_POM_oper > mdates.date2num(tdorian),\
                     timestamp_POM_oper <= mdates.date2num(datetime(2019,8,29,16))))[0]

plt.figure()
plt.plot(target_salt_POM_oper[:,0],target_depth_POM_oper[:,0],'.-',color='indianred',label='18 hours Before')
plt.plot(target_salt_POM_oper[:,aft[-1]],target_depth_POM_oper[:,aft[-1]],'.-',color='slateblue',label='21 hours After')
plt.plot(np.arange(35,37,0.05),np.tile(MLD_dens_crit_POM_oper[0],len(np.arange(35,37,0.05))),'--',color='indianred')
plt.plot(np.arange(35,37,0.05),np.tile(MLD_dens_crit_POM_oper[aft[-1]],len(np.arange(35,37,0.05))),'--',color='slateblue')
plt.plot(target_salt_POM_oper[:,bef],target_depth_POM_oper[:,bef],'-',color='indianred',alpha=0.2)
plt.plot(target_salt_POM_oper[:,aft],target_depth_POM_oper[:,bef],'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
#plt.xlim([34.5,37.5])
plt.xlim([35,37])
#plt.xlim([34,37])
plt.title('Profile before and after Dorian POM Oper',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Salinity',size=14)

file = folder_fig + 'POM_oper_Salt_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Vertical profile salinity POM exp before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timestamp_POM_exp <= mdates.date2num(tdorian)
aft = np.where(np.logical_and(timestamp_POM_exp > mdates.date2num(tdorian),\
                     timestamp_POM_exp <= mdates.date2num(datetime(2019,8,29,16))))[0]

plt.figure()
plt.plot(target_salt_POM_exp[:,0],target_depth_POM_exp[:,0],'.-',color='indianred',label='18 hours Before')
plt.plot(target_salt_POM_exp[:,aft[-1]],target_depth_POM_exp[:,aft[-1]],'.-',color='slateblue',label='21 hours After')
plt.plot(np.arange(35,37,0.05),np.tile(MLD_dens_crit_POM_exp[0],len(np.arange(35,37,0.05))),'--',color='indianred')
plt.plot(np.arange(35,37,0.05),np.tile(MLD_dens_crit_POM_exp[aft[-1]],len(np.arange(35,37,0.05))),'--',color='slateblue')
plt.plot(target_salt_POM_exp[:,bef],target_depth_POM_exp[:,bef],'-',color='indianred',alpha=0.2)
plt.plot(target_salt_POM_exp[:,aft],target_depth_POM_exp[:,aft],'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
#plt.xlim([34.5,37.5])
plt.xlim([35,37])
#plt.xlim([34,37])
plt.title('Profile before and after Dorian POM Exp',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Salinity',size=14)

file = folder_fig + 'POM_exp_Salt_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Vertical profile salinity HYCOM exp before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timestamp_HYCOM_exp <= mdates.date2num(tdorian)
aft = np.where(np.logical_and(timestamp_HYCOM_exp > mdates.date2num(tdorian),\
                     timestamp_HYCOM_exp <= mdates.date2num(datetime(2019,8,29,16))))[0]

plt.figure()
plt.plot(target_salt_HYCOM_exp[:,0],-depth_HYCOM_exp,'.-',color='indianred',label='18 hours Before')
plt.plot(target_salt_HYCOM_exp[:,aft[-1]],-depth_HYCOM_exp,'.-',color='slateblue',label='21 hours After')
plt.plot(np.arange(35,37,0.05),np.tile(-MLD_dens_crit_HYCOM_exp[0],len(np.arange(35,37,0.05))),'--',color='indianred')
plt.plot(np.arange(35,37,0.05),np.tile(-MLD_dens_crit_HYCOM_exp[aft[-1]],len(np.arange(35,37,0.05))),'--',color='slateblue')
plt.plot(target_salt_HYCOM_exp[:,bef],-depth_HYCOM_exp,'-',color='indianred',alpha=0.2)
plt.plot(target_salt_HYCOM_exp[:,aft],-depth_HYCOM_exp,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
#plt.xlim([34.5,37.5])
plt.xlim([35,37])
#plt.xlim([34,37])
plt.title('Profile before and after Dorian HYCOM Exp',size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Salinity',size=14)

file = folder_fig + ' ' + '_Sal_prof_bef_after_Dorian_HYCOM_exp'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Vertical profile density before and after Dorian

tdorian = datetime(2019,8,28,18)
bef = timeg <= tdorian
aft = np.logical_and(timeg > tdorian, timeg <= datetime(2019,8,29,18))

plt.figure()
plt.plot(np.nanmean(densg_gridded[:,bef],1),-depthg_gridded,'.-',color='indianred',label='18 hours Before')
plt.plot(np.nanmean(densg_gridded[:,aft],1),-depthg_gridded,'.-',color='slateblue',label='1 day After')
plt.plot(densg_gridded[:,bef],-depthg_gridded,'-',color='indianred',alpha=0.2)
plt.plot(densg_gridded[:,aft],-depthg_gridded,'-',color='slateblue',alpha=0.2)
plt.legend()
plt.ylim([-100,0])
#plt.xlim([26,30])
plt.title('Profile before and after Dorian '+ inst_id.split('-')[0],size=16)
plt.ylabel('Depth (m)',size=14)
plt.xlabel('Density ($kg/m^3$)')

file = folder_fig + ' ' + inst_id + 'dens_prof_bef_after_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Map  glider position

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(5, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)  
plt.plot(long,latg,'.-r')
plt.axis('scaled')  
#plt.yticks([])
#plt.xticks([])
plt.axis([-69,-64,16.5,21.5])
plt.text(np.mean(long)+0.1,np.mean(latg)+0.1,inst_id.split('-')[0],weight='bold',
                bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))

file = folder_fig + ' ' + inst_id + 'map_location'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%

fig,ax1 = plt.subplots(figsize=(6, 4))
plt.plot(timeg,MLD_temp_crit_glid,'.-k',label='MLD')
plt.legend(loc='upper left')
plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(30,50)))
plt.plot(tDorian,np.arange(30,50),'--k')
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

ax2 = ax1.twinx()
ax2.tick_params('y', colors='seagreen')
ax2.plot(timeg,Smean_dens_crit_glid,'.-',color='seagreen',label='Salt ML')
plt.legend(loc='upper center')
plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])

ax3 = ax1.twinx()
ax3.tick_params('y', colors='indianred')
ax3.plot(timeg,Tmean_dens_crit_glid,color='indianred',label='Temp ML')
plt.legend(loc='upper right')
plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])
xfmt = mdates.DateFormatter('%d \n %H')
ax1.xaxis.set_major_formatter(xfmt)

file = folder_fig + ' ' + inst_id + '_MLD_Temp_salt_ML_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%%

fig,ax1 = plt.subplots(figsize=(6, 4))
plt.plot(timeg,MLD_temp_crit_glid,'.-k',label='MLD')
plt.legend(loc='upper left')
#plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(30,50)))
plt.plot(tDorian,np.arange(30,50),'--k')
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

ax2 = ax1.twinx()
ax2.tick_params('y', colors='indianred')
ax2.plot(timeg,Tmean_dens_crit_glid,'.-',color='indianred',label='Temp ML')
plt.legend(loc='upper center')
#plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])
xfmt = mdates.DateFormatter('%d \n %b')
ax2.xaxis.set_major_formatter(xfmt)

file = folder_fig + ' ' + inst_id + '_MLD_Temp_ML_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Download wind speed and direction

url_NDBC = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/41043/41043h2019.nc'

wind_NDBC = xr.open_dataset(url_NDBC,decode_times=False)

tt = wind_NDBC['time']
time_NDBC = netCDF4.num2date(tt[:],tt.units)

wspd_NDBC = np.asarray(wind_NDBC['wind_spd'][:])[:,0,0]
wdir_NDBC = np.asarray(wind_NDBC['wind_dir'][:])[:,0,0] 

#%% Calculate rate of rotation of the wind

theta_wind = np.deg2rad(wdir_NDBC)
dtheta_wind = np.gradient(theta_wind)
dt = np.gradient(mdates.date2num(time_NDBC))*(60*24*60)
f = 0.524 * 10**(-4)#2*Omega*np.sin(lat) # Omega is rotation rate of earth
dthetadt_wind_over_f = dtheta_wind/(dt*f)

#%% 

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg,MLD_temp_crit_glid,'.-k',label='MLD')
plt.legend(loc='upper left')
plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(30,50)))
plt.plot(tDorian,np.arange(30,50),'--k')
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

ax2 = ax1.twinx()
ax2.plot(time_NDBC,wspd_NDBC,'.-',color='slateblue',alpha=0.5)
ax2.set_ylabel('Wind Speed (m/s)',color = 'slateblue',fontsize=14)
ax2.tick_params('y', colors='slateblue')
plt.xlim([datetime(2019,8,28),datetime(2019,8,29,18)])
plt.ylim([3,13])
xfmt = mdates.DateFormatter('%d \n %H')
ax2.xaxis.set_major_formatter(xfmt)
plt.grid(True)

#%% 

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg,MLD_temp_crit_glid,'.-k',label='MLD')
plt.legend(loc='upper left')
plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(30,50)))
plt.plot(tDorian,np.arange(30,50),'--k')
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

ax2 = ax1.twinx()
ax2.plot(time_NDBC,wdir_NDBC,'.-',color='seagreen',alpha=0.5)
ax2.set_ylabel('Wind Direction (degT)',color = 'seagreen',fontsize=14)
ax2.tick_params('y', colors='seagreen')
plt.xlim([datetime(2019,8,28),datetime(2019,8,29,18)])
plt.ylim([50,200])
xfmt = mdates.DateFormatter('%d \n %H')
ax2.xaxis.set_major_formatter(xfmt)
plt.grid(True)

#%%

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg,MLD_temp_crit_glid,'.-k',label='MLD')
plt.legend(loc='upper left')
plt.xlim([datetime(2019,8,28,0),datetime(2019,8,29,18)])
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(30,50)))
plt.plot(tDorian,np.arange(30,50),'--k')
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

ax2 = ax1.twinx()
ax2.tick_params('y', colors='darkblue')
ax2.plot(time_NDBC,dthetadt_wind_over_f,'.-',color='darkblue')
plt.xlim([datetime(2019,8,28),datetime(2019,8,29,18)])
plt.ylim([-20,20])
xfmt = mdates.DateFormatter('%d \n %H')
ax1.xaxis.set_major_formatter(xfmt)
plt.ylabel('$dtheta/dt * 1/f$',color='darkblue')

#%% Figure forecasted track POM operational, experiental and HYCOM exp

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=float(lead_time_pom_oper[-1]))
date_end = str(tend)

okt = np.logical_and(time_best_track >= tini,time_best_track <= tend)

# time forecasted track_exp
time_forec_track_pom_oper = np.asarray([tini + timedelta(hours = float(t)) for t in lead_time_pom_oper])
oktt = [np.where(t == time_forec_track_pom_oper)[0][0] for t in time_best_track[okt]]
    
lev = np.arange(-9000,9100,100)   
plt.figure()
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.plot(lon_forec_track_pom_oper[oktt], lat_forec_track_pom_oper[oktt],'X-',color='mediumorchid',\
         markeredgecolor='k',markersize=7,label='POM Oper')
plt.plot(lon_forec_track_pom_exp[oktt], lat_forec_track_pom_exp[oktt],'^-',color='teal',\
         markeredgecolor='k',markersize=7,label='POM Exp')
plt.plot(lon_forec_track_hycom_exp[oktt], lat_forec_track_hycom_exp[oktt],'H-',color='darkorange',\
         markeredgecolor='k',markersize=7,label='HYCOM Exp')
plt.plot(lon_best_track[okt], lat_best_track[okt],'o-',color='k',label='Best Track')   
plt.legend()
plt.title('Track Forecast Dorian '+ cycle,fontsize=18)

file = folder_fig + 'best_track_vs_forec_track_POM_HYCOM_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 