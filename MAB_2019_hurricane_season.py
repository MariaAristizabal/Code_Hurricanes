#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:02:35 2020

@author: root
"""

#%% User input

# Servers location
url_erddap = 'https://data.ioos.us/gliders/erddap'
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
url_doppio = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'

folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

date_ini = '2019/05/01/00'
date_end = '2019/11/30/00'
scatter_plot = 'no'

# MAB
lon_lim = [-80.0,-66.0]
lat_lim = [33.0,45.0]

# glider variable to retrieve
var_name_glider = 'temperature'
#var_glider = 'salinity'
delta_z = 0.4 # bin size in the vertical when gridding the variable vertical profile 
              # default value is 0.3  

# model variable name
model_name = 'GOFS 3.1'
var_name_model = 'water_temp'
#var_model = 'salinity'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import cmocean
import netCDF4
#from datetime import datetime, timedelta
import sys
import seawater as sw

sys.path
#sys.path.append('/home/aristizabal/glider_model_comparisons_Python/')
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import retrieve_dataset_id_erddap_server, read_glider_data_erddap_server
from process_glider_data import grid_glider_data
from glider_transect_model_com import get_glider_transect_from_GOFS

#%% 
def get_glider_transect_from_DOPPIO(url_doppio,timeg,long,latg):

    #  Read Doppio time, lat and lon
    print('Retrieving coordinates and time from Doppio ')
    
    doppio = xr.open_dataset(url_doppio,decode_times=False)
    
    latrhodoppio = np.asarray(doppio.variables['lat_rho'][:])
    lonrhodoppio = np.asarray(doppio.variables['lon_rho'][:])
    srhodoppio = np.asarray(doppio.variables['s_rho'][:])
    ttdoppio = doppio.variables['time'][:]
    tdoppio = netCDF4.num2date(ttdoppio[:],ttdoppio.attrs['units'])

    # Read Doppio S-coordinate parameters
    
    Vtransf = np.asarray(doppio.variables['Vtransform'])
    #Vstrect = np.asarray(doppio.variables['Vstretching'])
    Cs_r = np.asarray(doppio.variables['Cs_r'])
    #Cs_w = np.asarray(doppio.variables['Cs_w'])
    sc_r = np.asarray(doppio.variables['s_rho'])
    #sc_w = np.asarray(doppio.variables['s_w'])
    
    # depth
    h = np.asarray(doppio.variables['h'])
    # critical depth parameter
    hc = np.asarray(doppio.variables['hc'])
    
    igrid = 1

    # Narrowing time window of Doppio to coincide with glider time window
    
    tmin = mdates.num2date(mdates.date2num(timeg[0]))
    tmax = mdates.num2date(mdates.date2num(timeg[-1]))
    oktime_doppio = np.where(np.logical_and(mdates.date2num(tdoppio) >= mdates.date2num(tmin),\
                                     mdates.date2num(tdoppio) <= mdates.date2num(tmax)))
    timedoppio = tdoppio[oktime_doppio]        
    
    # Changing times to timestamp
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    tstamp_doppio = [mdates.date2num(timedoppio[i]) for i in np.arange(len(timedoppio))]
    
    # interpolating glider lon and lat to lat and lon on doppio time
    sublondoppio = np.interp(tstamp_doppio,tstamp_glider,long)
    sublatdoppio = np.interp(tstamp_doppio,tstamp_glider,latg)

    # getting the model grid positions for sublonm and sublatm
    oklatdoppio = np.empty((len(oktime_doppio[0])))
    oklatdoppio[:] = np.nan
    oklondoppio= np.empty((len(oktime_doppio[0])))
    oklondoppio[:] = np.nan
    for t,tt in enumerate(oktime_doppio[0]):
        
        # search in xi_rho direction 
        oklatmm = []
        oklonmm = []
        for pos_xi in np.arange(latrhodoppio.shape[1]):
            pos_eta = np.round(np.interp(sublatdoppio[t],latrhodoppio[:,pos_xi],np.arange(len(latrhodoppio[:,pos_xi])),\
                                         left=np.nan,right=np.nan))
            if np.isfinite(pos_eta):
                oklatmm.append((pos_eta).astype(int))
                oklonmm.append(pos_xi)
            
        pos = np.round(np.interp(sublondoppio[t],lonrhodoppio[oklatmm,oklonmm],np.arange(len(lonrhodoppio[oklatmm,oklonmm])))).astype(int)    
        oklatdoppio1 = oklatmm[pos]
        oklondoppio1 = oklonmm[pos] 
        
        #search in eta-rho direction
        oklatmm = []
        oklonmm = []
        for pos_eta in np.arange(latrhodoppio.shape[0]):
            pos_xi = np.round(np.interp(sublondoppio[t],lonrhodoppio[pos_eta,:],np.arange(len(lonrhodoppio[pos_eta,:])),\
                                        left=np.nan,right=np.nan))
            if np.isfinite(pos_xi):
                oklatmm.append(pos_eta)
                oklonmm.append(pos_xi.astype(int))
        
        pos_lat = np.round(np.interp(sublatdoppio[t],latrhodoppio[oklatmm,oklonmm],np.arange(len(latrhodoppio[oklatmm,oklonmm])))).astype(int)
        oklatdoppio2 = oklatmm[pos_lat]
        oklondoppio2 = oklonmm[pos_lat] 
        
        #check for minimum distance
        dist1 = np.sqrt((oklondoppio1-sublondoppio[t])**2 + (oklatdoppio1-sublatdoppio[t])**2) 
        dist2 = np.sqrt((oklondoppio2-sublondoppio[t])**2 + (oklatdoppio2-sublatdoppio[t])**2) 
        if dist1 >= dist2:
            oklatdoppio[t] = oklatdoppio1
            oklondoppio[t] = oklondoppio1
        else:
            oklatdoppio[t] = oklatdoppio2
            oklondoppio[t] = oklondoppio2
        
    oklatdoppio = oklatdoppio.astype(int)
    oklondoppio = oklondoppio.astype(int)
    
    # Getting glider transect from doppio
    print('Getting glider transect from Doppio')
    target_tempdoppio = np.empty((len(srhodoppio),len(oktime_doppio[0])))
    target_tempdoppio[:] = np.nan
    target_saltdoppio = np.empty((len(srhodoppio),len(oktime_doppio[0])))
    target_saltdoppio[:] = np.nan
    target_zdoppio = np.empty((len(srhodoppio),len(oktime_doppio[0])))
    target_zdoppio[:] = np.nan
    for i in range(len(oktime_doppio[0])):
        print(len(oktime_doppio[0]),' ',i)
        target_tempdoppio[:,i] = np.flip(doppio.variables['temp'][oktime_doppio[0][i],:,oklatdoppio[i],oklondoppio[i]])
        target_saltdoppio[:,i] = np.flip(doppio.variables['salt'][oktime_doppio[0][i],:,oklatdoppio[i],oklondoppio[i]])
        h = np.asarray(doppio.variables['h'][oklatdoppio[i],oklondoppio[i]])
        zeta = np.asarray(doppio.variables['zeta'][oktime_doppio[0][i],oklatdoppio[i],oklondoppio[i]])
        
        # Calculate doppio depth as a function of time
        if Vtransf ==1:
            if igrid == 1:
                for k in np.arange(sc_r.shape[0]):
                    z0 = (sc_r[k]-Cs_r[k])*hc + Cs_r[k]*h
                    target_zdoppio[k,i] = z0 + zeta * (1.0 + z0/h);
    
        if Vtransf == 2:
            if igrid == 1:
                for k in np.arange(sc_r.shape[0]):
                    z0 = (hc*sc_r[k] + Cs_r[k]*h) / (hc+h)
                    target_zdoppio[k,i] = zeta + (zeta+h)*z0
        
        target_zdoppio[:,i] = np.flip(target_zdoppio[:,i])
    
    # change time vector to matrix
    target_timedoppio = np.tile(timedoppio,(len(srhodoppio),1))
    
    return(target_tempdoppio,target_saltdoppio,target_zdoppio,target_timedoppio)

#%%
def get_glider_transect_from_RTOFS(folder_RTOFS,timeg,long,latg):
    
    from datetime import datetime, timedelta
    import glob
    import os

    ti_year = int(str(timeg[0]).split('-')[0]) 
    ti_month = int(str(timeg[0]).split('-')[1]) 
    ti_day = int(str(timeg[0]).split('-')[2].split('T')[0])
    tini = datetime(ti_year,ti_month,ti_day) 
    
    te_year = int(str(timeg[-1]).split('-')[0]) 
    te_month = int(str(timeg[-1]).split('-')[1]) 
    te_day = int(str(timeg[-1]).split('-')[2].split('T')[0])
    tend = datetime(te_year,te_month,te_day) 

    # Read RTOFS grid and time
    print('Retrieving coordinates from RTOFS')
    
    tt = tini
    if tt.month < 10:
        if tt.day < 10:
            file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + '0' + str(tt.month) + '0' + str(tt.day)
        else:
            file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + '0' + str(tt.month) + str(tt.day)
    else:
        if tt.day < 10:
            file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + str(tt.month) + '0' + str(tt.day)
        else:
            file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + str(tt.month) + str(tt.day)
        
    ncfiles = sorted(glob.glob(os.path.join(file_rtofs,'*.nc')))
        
    RTOFS = xr.open_dataset(ncfiles[0])
    lat_RTOFS = np.asarray(RTOFS.Latitude[:])
    lon_RTOFS = np.asarray(RTOFS.Longitude[:])
    depth_RTOFS = np.asarray(RTOFS.Depth[:])

    # Load RTOFS nc files
    
    print('Reading RTOFS nc files')
    
    nt_rtofs = ((tend-tini)*4).days
    target_temp_RTOFS = np.empty((len(depth_RTOFS),nt_rtofs))
    target_temp_RTOFS[:] = np.nan
    target_salt_RTOFS = np.empty((len(depth_RTOFS),nt_rtofs))
    target_salt_RTOFS[:] = np.nan
    time_RTOFS = []
     
    for dt in np.arange((tend-tini).days):
        tt = tini + timedelta(days = int(dt))
        print(tt)
        if tt.month < 10:
            if tt.day < 10:
                file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + '0' + str(tt.month) + '0' + str(tt.day)
            else:
                file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + '0' + str(tt.month) + str(tt.day)
        else:
            if tt.day < 10:
                file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + str(tt.month) + '0' + str(tt.day)
            else:
                file_rtofs = folder_RTOFS + 'rtofs.' + str(tt.year) + str(tt.month) + str(tt.day)
        
        ncfiles = sorted(glob.glob(os.path.join(file_rtofs,'*.nc')))
    
        for x,file in enumerate(ncfiles):
            print(x)
            RTOFS = xr.open_dataset(file)
            
            trtofs = np.asarray(RTOFS.MT[:])[0]
            time_RTOFS.append(trtofs)
            
            # Interpolating latg and longlider into RTOFS grid
            sublon_rtofs = np.interp(mdates.date2num(trtofs),mdates.date2num(timeg),long)
            sublat_rtofs = np.interp(mdates.date2num(trtofs),mdates.date2num(timeg),latg)
            oklon_rtofs = np.int(np.round(np.interp(sublon_rtofs,lon_RTOFS[0,:],np.arange(len(lon_RTOFS[0,:])))))
            oklat_rtofs = np.int(np.round(np.interp(sublat_rtofs,lat_RTOFS[:,0],np.arange(len(lat_RTOFS[:,0])))))
            
            count = int(dt*4 + x) 
            print(count)
            target_temp_RTOFS[:,count] = np.asarray(RTOFS['temperature'][0,:,oklat_rtofs,oklon_rtofs])
            target_salt_RTOFS[:,count] = np.asarray(RTOFS['salinity'][0,:,oklat_rtofs,oklon_rtofs])
    
    time_RTOFS = np.asarray(time_RTOFS)
    
    return(target_temp_RTOFS,target_salt_RTOFS,depth_RTOFS,time_RTOFS) 

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
            d10 = np.where(np.abs(depth) >= 10)[0][0]
        if depth.ndim == 2:
            d10 = np.where(np.abs(depth[:,t]) >= 10)[0][0]
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

#%% Calculate time series of potential Energy Anomaly over the top 100 m

def Potential_Energy_Anomaly(time,depth,density):
    g = 9.8 #m/s
    PEA = np.empty((len(time)))
    PEA[:] = np.nan
    for t,tstamp in enumerate(time):   
        print(t)
        if np.ndim(depth) == 2:
            dindex = np.fliplr(np.where(np.asarray(np.abs(depth[:,t])) <= 100))[0]
        else:
            dindex = np.fliplr(np.where(np.asarray(np.abs(depth)) <= 100))[0]
        if len(dindex) == 0:
            PEA[t] = np.nan
        else:
            if np.ndim(depth) == 2: 
                zz = np.asarray(np.abs(depth[dindex,t]))
            else:
                zz = np.asarray(np.abs(depth[dindex]))
            denss = np.asarray(density[dindex,t])
            ok = np.isfinite(denss)
            z = zz[ok]
            dens = denss[ok]
            if len(z)==0 or len(dens)==0 or np.min(zz) > 10 or np.max(zz) < 30:
                PEA[t] = np.nan
            else:
                if z[-1] - z[0] > 0:
                    # So PEA is < 0
                    #sign = -1
                    # Adding 0 to sigma integral is normalized
                    z = np.append(0,z)
                else:
                    # So PEA is < 0
                    #sign = 1
                    # Adding 0 to sigma integral is normalized
                    z = np.flipud(z)
                    z = np.append(0,z)
                    dens = np.flipud(dens)
    
                # adding density at depth = 0
                densit = np.interp(z,z[1:],dens)
                densit = np.flipud(densit)
                
                # defining sigma
                max_depth = np.nanmax(zz[ok])  
                sigma = -1*z/max_depth
                sigma = np.flipud(sigma)
                
                rhomean = np.trapz(densit,sigma,axis=0)
                drho = rhomean-densit
                torque = drho * sigma
                PEA[t] = g* max_depth * np.trapz(torque,sigma,axis=0) 
                #print(max_depth, ' ',PEA[t]) 
                
    return PEA 

#%%
def find_mean_temperature_below_picnocline(temp,dens,depth):
    drho = np.diff(dens,axis=0)
    dz = np.diff(depth,axis=0)
    if depth.ndim == 1:
        dz_matrix = np.tile(dz,(drho.shape[1],1)).T
    else:
        dz_matrix = dz
    
    drho_dz = np.abs(drho/dz_matrix)
    
    mean_temp_below_picn = np.empty((temp.shape[1]))
    mean_temp_below_picn[:] = np.nan
    for t in np.arange(temp.shape[1]):
        print(t)
        if np.where(np.isfinite(temp[:,t]))[0].shape[0] != 0:
            max_grad = np.where(drho_dz[:,t] == np.nanmax(drho_dz[:,t]))[0][-1]
            okd = np.isfinite(temp[:,t])
            if depth.ndim == 1:
                ddepth = depth
            else:
                ddepth = depth[:,t]
            
            max_depth = np.where(np.abs(ddepth) <= 100)[0][-1]
            if np.max(np.abs(depth[okd])) < 20:
                mean_temp_below_picn[t] = np.nan
            else:
                mean_temp_below_picn[t] = np.nanmean(temp[max_grad+1:max_depth+1,t]) 
        else:
            mean_temp_below_picn[t] = np.nan
            
    return mean_temp_below_picn

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

#%%
url_server = url_erddap
gliders = retrieve_dataset_id_erddap_server(url_server,lat_lim,lon_lim,date_ini,date_end)
print('The gliders found are ')
print(gliders)

#%% Blue-20190815

# reading glider data
dataset_id = gliders[0] 

date_ini = '2019/08/15/00'
date_end = '2019/08/20/00'

#%%
kwargs = dict(date_ini=date_ini,date_end=date_end)
scatter_plot = 'no'

#tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
#                                   lat_lim,lon_lim,scatter_plot)

tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                   lat_lim,lon_lim,scatter_plot,**kwargs)
    
#%% gridding glider data  
contour_plot = 'no' # default value is 'yes'
delta_z = 0.4     # default value is 0.3

var_name = 'temperature'
tempg_gridded, timegg, depthg_gridded = \
                    grid_glider_data(var_name,dataset_id,tempg,timeg,latg,long,depthg,delta_z,contour_plot)
    
var_name = 'salinity'
saltg_gridded, _, _  = \
                    grid_glider_data(var_name,dataset_id,saltg,timeg,latg,long,depthg,delta_z,contour_plot)          

# Calculate density
densg = sw.dens(saltg,tempg,depthg) 

var_name = 'density'
densg_gridded, _, _ = \
                    grid_glider_data(var_name,dataset_id,densg,timeg,latg,long,depthg,delta_z,contour_plot) 

#%% reading GOFS

# model variable name
model_name = 'GOFS 3.1'
var_name_model = 'water_temp'              
# Get temperature transect from model    
target_temp_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot='yes')
                  
var_name_model = 'salinity'              
# Get temperature transect from model    
target_salt_GOFS, _, _, _, _ = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        saltg,timeg,latg,long,depthg,contour_plot='yes')
                                        
# Calculate density GOFS 
target_dens_GOFS = sw.dens(target_salt_GOFS,target_temp_GOFS,np.tile(depth_GOFS,(len(time_GOFS),1)).T) 

#%% Reading Doppio
target_temp_doppio,target_salt_doppio,target_depth_doppio,time_doppio = \
    get_glider_transect_from_DOPPIO(url_doppio,timeg,long,latg)
    
# Calculate density Doppio 
target_dens_doppio = sw.dens(target_salt_doppio,target_temp_doppio,target_depth_doppio) 

#%% Reading RTOFS
target_temp_RTOFS,target_salt_RTOFS,depth_RTOFS,time_RTOFS = \
    get_glider_transect_from_RTOFS(folder_RTOFS,timeg,long,latg)
    
# Calculate density Doppio 
target_dens_RTOFS = sw.dens(target_salt_RTOFS,target_temp_RTOFS,np.tile(depth_RTOFS,(len(time_RTOFS),1)).T)     

#%% MLD
dt = 0.2
drho = 0.125

# glider data
MLD_temp_crit_glider, _, _, _, MLD_dens_crit_glider, Tmean_dens_crit_glider, Smean_dens_crit_glider, _ = \
MLD_temp_and_dens_criteria(dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)

# GOFS  
MLD_temp_crit_GOFS, _, _, _, MLD_dens_crit_GOFS, Tmean_dens_crit_GOFS, Smean_dens_crit_GOFS, _ = \
MLD_temp_and_dens_criteria(dt,drho,time_GOFS,depth_GOFS,target_temp_GOFS,target_salt_GOFS,target_dens_GOFS)  

# Doppio
MLD_temp_crit_doppio, _, _, _, MLD_dens_crit_doppio, Tmean_dens_crit_doppio, Smean_dens_crit_doppio, _ = \
MLD_temp_and_dens_criteria(dt,drho,time_doppio[0,:],target_depth_doppio,target_temp_doppio,target_salt_doppio,target_dens_doppio)

# RTOFS
MLD_temp_crit_RTOFS, _, _, _, MLD_dens_crit_RTOFS, Tmean_dens_crit_RTOFS, Smean_dens_crit_RTOFS, _ = \
MLD_temp_and_dens_criteria(dt,drho,time_RTOFS,depth_RTOFS,target_temp_RTOFS,target_salt_RTOFS,target_dens_RTOFS)

#%% Surface Ocean Heat Content

# glider
OHC_glider = OHC_surface(timeg,tempg_gridded,depthg_gridded,densg_gridded)

# GOFS
OHC_GOFS = OHC_surface(time_GOFS,target_temp_GOFS,depth_GOFS,target_dens_GOFS)

# Doppio
OHC_doppio = OHC_surface(time_doppio[0,:],target_temp_doppio,target_depth_doppio,target_dens_doppio)

# RTOFS
OHC_RTOFS = OHC_surface(time_RTOFS,target_temp_RTOFS,depth_RTOFS,target_dens_RTOFS)

#%% Calculate time series of potential Energy Anomaly over the top 100 m

# Glider
PEA_glider = Potential_Energy_Anomaly(timeg,depthg,densg)

# GOFS
PEA_GOFS = Potential_Energy_Anomaly(time_GOFS,depth_GOFS,target_dens_GOFS)

# Doppio
PEA_doppio = Potential_Energy_Anomaly(time_doppio[0,:],target_depth_doppio,target_dens_doppio)

# RTOFS
PEA_RTOFS = Potential_Energy_Anomaly(time_RTOFS,depth_RTOFS,target_dens_RTOFS)

#%% Calculate time series of temperature below picnocline

mean_temp_below_picn_glider = find_mean_temperature_below_picnocline(tempg_gridded,\
                            densg_gridded,depthg_gridded)
    
mean_temp_below_picn_GOFS = find_mean_temperature_below_picnocline(target_temp_GOFS,\
                            target_dens_GOFS,depth_GOFS)
    
mean_temp_below_picn_doppio = find_mean_temperature_below_picnocline(target_temp_doppio,\
                            target_dens_doppio,target_depth_doppio)
    
mean_temp_below_picn_RTOFS = find_mean_temperature_below_picnocline(target_temp_RTOFS,\
                            target_dens_RTOFS,depth_RTOFS)
  
#%% Plot glider trajectory

# Getting subdomain for plotting glider track on bathymetry
oklatbath = np.logical_and(bath_lat >= np.min(latg)-5,bath_lat <= np.max(latg)+5)
oklonbath = np.logical_and(bath_lon >= np.min(long)-5,bath_lon <= np.max(long)+5)

fig, ax = plt.subplots()    
bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

lev = np.arange(-9000,9100,100)
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
#plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
#plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
#plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.axis([np.min(long)-5,np.max(long)+5,np.min(latg)-5,np.max(latg)+5])
plt.plot(long,latg,'.-',color='orange',label='Glider track',\
     markersize=3)
plt.title('Glider track '+dataset_id,fontsize=20)
plt.axis('scaled')
plt.legend(loc='upper center',bbox_to_anchor=(1.1,1))

file = folder_fig +'trajectory_' + dataset_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% plot glider
    
color_map = cmocean.cm.thermal
okg = depthg_gridded <= np.max(depthg_gridded) 
okm = depth_GOFS <= np.max(depthg_gridded) 
min_val = np.int(np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])])))
max_val = np.int(np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])])))
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))

# plot
fig, ax = plt.subplots(figsize=(12, 2.0))     
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=color_map,**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded,[26],colors='k')

cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('Temperature',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

plt.title('Along Track ' + 'Temperature' + ' Profile ' + dataset_id)

file = folder_fig +'temp_transect_' + dataset_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% plot GOFS 3.1

fig, ax = plt.subplots(figsize=(12, 2.0))     
cs = plt.contourf(mdates.date2num(time_GOFS),-depth_GOFS,target_temp_GOFS,cmap=color_map,**kw)
plt.contour(mdates.date2num(time_GOFS),-depth_GOFS,target_temp_GOFS,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('Temperature',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Along Track ' + 'Temperature' + ' Profile ' + model_name)      

file = folder_fig +'temp_transect_' + model_name.split()[0] +'_'+ dataset_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% plot Doppio

fig, ax = plt.subplots(figsize=(12, 2.0))     
cs = plt.contourf(time_doppio,target_depth_doppio,target_temp_doppio,cmap=color_map,**kw)
plt.contour(time_doppio,target_depth_doppio,target_temp_doppio,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('Temperature',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Along Track ' + 'Temperature' + ' Profile Doppio')      

file = folder_fig +'temp_transect_' + 'Doppio'+'_'+ dataset_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% plot RTOFS

fig, ax = plt.subplots(figsize=(12, 2.0))     
cs = plt.contourf(time_RTOFS,-depth_RTOFS,target_temp_RTOFS,cmap=color_map,**kw)
plt.contour(time_RTOFS,-depth_RTOFS,target_temp_RTOFS,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('Temperature',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Along Track ' + 'Temperature' + ' Profile RTOFS')      

folder_fig = '/home/aristizabal/Figures/'
file = folder_fig +'temp_transect_' + 'RTOFS'+'_'+ dataset_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Temp ML
    
fig,ax = plt.subplots(figsize=(12, 2.8))
plt.plot(timeg,Tmean_dens_crit_glider,'-o',color='royalblue',label=dataset_id.split('-')[0],linewidth=3)
plt.plot(time_GOFS,Tmean_dens_crit_GOFS,'--s',color='indianred',label='GOFS 3.1')
plt.plot(time_doppio[0,:],Tmean_dens_crit_doppio,'-X',color='teal',label='Doppio')
plt.plot(time_RTOFS,Tmean_dens_crit_RTOFS,'-^',color='darkorange',label='RTOFS')
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel('($^oC$)',fontsize = 14)
plt.title('Mixed Layer Temperature',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))
plt.xlim([timeg[0],timeg[-1]])

file = folder_fig + dataset_id + '_temp_ml'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% OHC figure
    
fig,ax = plt.subplots(figsize=(12, 2.8))
plt.plot(timeg,OHC_glider*10**-7,'-o',color='royalblue',label=dataset_id.split('-')[0],linewidth=3)
plt.plot(time_GOFS,OHC_GOFS*10**-7,'--s',color='indianred',label='GOFS 3.1')
plt.plot(time_doppio[0,:],OHC_doppio*10**-7,'-X',color='teal',label='Doppio')
plt.plot(time_RTOFS,OHC_RTOFS*10**-7,'--s',color='darkorange',label='RTOFS')
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Ocean Heat Content',fontsize=16)
plt.ylabel('($KJ/cm^2$)',fontsize = 14)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))
plt.xlim([timeg[0],timeg[-1]])

file = folder_fig + dataset_id + '_OHC'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Figure Potential energy anomaly
fig,ax = plt.subplots(figsize=(12,2.8))
plt.plot(timeg,PEA_glider,'.-',color='royalblue',label=dataset_id.split('-')[0])
plt.plot(time_GOFS,PEA_GOFS,'.-',color='indianred',label='GOFS 3.1')
plt.plot(time_doppio[0,:],PEA_doppio,'-X',color='teal',label='Doppio')
plt.plot(time_RTOFS,PEA_RTOFS,'--s',color='darkorange',label='RTOFS')
plt.ylabel('($W/m^3$)',fontsize = 14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Potantial Energy Anomaly',fontsize=14)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))
plt.xlim([timeg[0],timeg[-1]])

file = folder_fig + dataset_id + '_PEA'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure mean tempearture below thermocline
fig,ax = plt.subplots(figsize=(12,2.8))
plt.plot(timeg,mean_temp_below_picn_glider ,'.-',color='royalblue',label=dataset_id.split('-')[0])
plt.plot(time_GOFS,mean_temp_below_picn_GOFS,'.-',color='indianred',label='GOFS 3.1')
plt.plot(time_doppio[0,:],mean_temp_below_picn_doppio,'-X',color='teal',label='Doppio')
plt.plot(time_RTOFS,mean_temp_below_picn_RTOFS,'--s',color='darkorange',label='RTOFS')
plt.ylabel('($^oc$)',fontsize = 14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Mean Temperature Below Pycnocline',fontsize=16)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))
plt.xlim([timeg[0],timeg[-1]])

file = folder_fig + dataset_id + '_mean_temp_below_pycn'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  
