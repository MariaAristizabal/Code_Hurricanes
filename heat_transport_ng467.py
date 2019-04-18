#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:08:24 2019

@author: aristizabal
"""

#%% User input

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng467_depth_aver_vel.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# Caribbean
lon_lim = [-69,-63];
lat_lim = [16,22];

#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import scipy.io as sio
from netCDF4 import Dataset

ng467 = sio.loadmat(mat_file)

#%% Functions to calculate density

def dens0(s, t):
    s, t = list(map(np.asanyarray, (s, t)))
    T68 = T68conv(t)
    # UNESCO 1983 Eqn.(13) p17.
    b = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
    c = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
    d = 4.8314e-4
    return (smow(t) + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) *
            T68) * T68) * s + (c[0] + (c[1] + c[2] * T68) * T68) * s * s ** 0.5 + d * s ** 2)

def smow(t):
    t = np.asanyarray(t)
    a = (999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9)
    T68 = T68conv(t)
    return (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) * T68) * T68)
    
def T68conv(T90):
    T90 = np.asanyarray(T90)
    return T90 * 1.00024

#%%

uglider = ng467['ug'][:,0]
vglider = ng467['vg'][:,0]
tempglider = ng467['tempg']
saltglider = ng467['saltg']
tstamp_glider = ng467['timeg'][:,0]
latglider = ng467['latg'][:,0]
longlider = ng467['long'][:,0]

tstamp_31 =  ng467['time31'][:,0]
oktime31 = ng467['oktime31'][:,0]
depth31 = ng467['depth31'][:,0]

u31 = ng467['target_u31']
v31 = ng467['target_v31']
temp31 = ng467['target_temp31']
salt31 = ng467['target_salt31']

#%% Changing timestamps to datenum

timeglid = []
tim31 = []
for i in np.arange(len(tstamp_glider)):
    timeglid.append(datetime.fromordinal(int(tstamp_glider[i])) + \
        timedelta(days=tstamp_glider[i]%1) - timedelta(days = 366))
    tim31.append(datetime.fromordinal(int(tstamp_31[i])) + \
        timedelta(days=tstamp_31[i]%1) - timedelta(days = 366))
    
timeglider = np.asarray(timeglid)
time31 = np.asarray(tim31)
tt31 = time31[oktime31]  

#%% Putting velocity vector at the same timestamp

Ug,oku = np.unique(uglider,return_index=True)
Vg = vglider[oku]
Xg = timeglider[oku]
Yg = np.zeros(len(Xg))

X31 = time31[oktime31]
Y31 = np.zeros(len(X31))

# Depth average velocoty in to 200 m
okd = depth31 < 200
U31 = np.mean(u31[okd,:],0)
V31 = np.mean(v31[okd,:],0)

ttg = np.asarray(tstamp_glider)[oku]
tt31 = np.asarray(tstamp_31)[oktime31]

U31_interp = np.interp(ttg,tt31,U31)
V31_interp = np.interp(ttg,tt31,V31)

#%% Heat transport calculation glider data

Cw = 3993 #J/(kg K)
rho_meang = np.nanmean(dens0(saltglider, tempglider),axis=0)
heat_transp_ug = Cw *rho_meang * np.nanmean(tempglider+273.15,axis=0) * uglider
heat_transp_vg = Cw *rho_meang * np.nanmean(tempglider+273.15,axis=0) * vglider

#%% Heat transport calculation model top 200 m

Cw = 3993 #J/(kg K)
okd = depth31 < 200
rho = dens0(salt31, temp31)
rho_mean31 = np.nanmean(rho[okd,:],axis=0)
heat_transp_u31 = Cw *rho_mean31 * np.nanmean(temp31[okd,:]+273.15,axis=0) * np.nanmean(u31[okd,:],axis=0)
heat_transp_v31 = Cw *rho_mean31 * np.nanmean(temp31[okd,:]+273.15,axis=0) * np.nanmean(v31[okd,:],axis=0)

#%% Heat transport calculation for along and cross channel direction 

alpha = np.radians(60)
velg_cross = np.cos(alpha)*uglider - np.sin(alpha)*vglider
velg_along = np.sin(alpha)*uglider + np.cos(alpha)*vglider
vel31_cross = np.cos(alpha)*u31 - np.sin(alpha)*v31
vel31_along = np.sin(alpha)*u31 + np.cos(alpha)*v31

# Heat transport calculation glider data
Cw = 3993 #J/(kg K)
rho_meang = np.nanmean(dens0(saltglider, tempglider),axis=0)
heat_transp_alongg = Cw *rho_meang * np.nanmean(tempglider+273.15,axis=0) * velg_along
heat_transp_crossg = Cw *rho_meang * np.nanmean(tempglider+273.15,axis=0) * velg_cross

# Heat transport calculation model top 200 m
Cw = 3993 #J/(kg K)
okd = depth31 < 200
rho = dens0(salt31, temp31)
rho_mean31 = np.nanmean(rho[okd,:],axis=0)
heat_transp_along31 = Cw *rho_mean31 * np.nanmean(temp31[okd,:]+273.15,axis=0) * np.nanmean(vel31_along[okd,:],axis=0)
heat_transp_cross31 = Cw *rho_mean31 * np.nanmean(temp31[okd,:]+273.15,axis=0) * np.nanmean(vel31_cross[okd,:],axis=0)

#%% Integrated heat transport

ok = np.isfinite(heat_transp_ug)
acc_heat_transp_ug = np.trapz(heat_transp_ug[ok],tstamp_glider[ok] * 24* 3600)

ok = np.isfinite(heat_transp_vg)
acc_heat_transp_vg = np.trapz(heat_transp_vg[ok],tstamp_glider[ok] * 24* 3600)

acc_heat_transp_u31 = np.trapz(heat_transp_u31,tstamp_31[oktime31] * 24* 3600)
acc_heat_transp_v31 = np.trapz(heat_transp_v31,tstamp_31[oktime31] * 24* 3600)

print('{:0.2E}'.format(acc_heat_transp_ug))
print('{:0.2E}'.format(acc_heat_transp_u31))
print('{:0.2E}'.format(acc_heat_transp_vg))
print('{:0.2E}'.format(acc_heat_transp_v31))

#%% Integrated heat transport along and cross

ok = np.isfinite(heat_transp_alongg)
acc_heat_transp_alongg = np.trapz(heat_transp_alongg[ok],tstamp_glider[ok] * 24* 3600)

ok = np.isfinite(heat_transp_crossg)
acc_heat_transp_crossg = np.trapz(heat_transp_crossg[ok],tstamp_glider[ok] * 24* 3600)

acc_heat_transp_along31 = np.trapz(heat_transp_along31,tstamp_31[oktime31] * 24* 3600)
acc_heat_transp_cross31 = np.trapz(heat_transp_cross31,tstamp_31[oktime31] * 24* 3600)

print('{:0.2E}'.format(acc_heat_transp_alongg))
print('{:0.2E}'.format(acc_heat_transp_along31))
print('{:0.2E}'.format(acc_heat_transp_crossg))
print('{:0.2E}'.format(acc_heat_transp_cross31))

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

#%% Figure heat transport East

fig, ax = plt.subplots(figsize=[10.5,3.5])
plt.plot(timeglider,heat_transp_ug,label=ng467['inst_id'][0].split('-')[0],color='b')
ax.fill_between(timeglider, 0, heat_transp_ug,alpha=0.4)
plt.plot(time31[oktime31],heat_transp_u31,label='GOFS3.1',color='r')
ax.fill_between(time31[oktime31], 0, heat_transp_u31,alpha=0.4)
plt.plot(timeglider,np.zeros(len(timeglider)),color='k',linestyle='--')
plt.legend(loc='upper left',fontsize=14)
plt.title('Heat Transport Top 200 m East Direction',size=20)
plt.ylim(-4*10**8,4*10**8)
plt.ylabel('$(\; \mathit{J/m^2 s} \:)$',size=16)

xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)

file = folder + 'heat_transport_East_' + ng467['inst_id'][0].split('-')[0]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure heat transport North

fig, ax = plt.subplots(figsize=[10.5,3.5])
plt.plot(timeglider,heat_transp_vg,label=ng467['inst_id'][0].split('-')[0],color='b')
ax.fill_between(timeglider, 0, heat_transp_vg,alpha=0.4)
plt.plot(time31[oktime31],heat_transp_v31,label='GOFS3.1',color='r')
ax.fill_between(time31[oktime31], 0, heat_transp_v31,alpha=0.4)
plt.plot(timeglider,np.zeros(len(timeglider)),color='k',linestyle='--')
plt.legend(loc='upper left',fontsize=14)
plt.title('Heat Transport Top 200 m North Direction',size=20)
plt.ylim(-4*10**8,4*10**8)
plt.ylabel('$(\; \mathit{J/m^2 s} \:)$',size=16)

xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)

file = folder + 'heat_transport_North_' + ng467['inst_id'][0].split('-')[0]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Map Caribbean with integrated heat transport

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev,cmap=plt.get_cmap('BrBG'))
#plt.plot(longlider,latglider,'.k')
plt.plot(np.mean(longlider),np.mean(latglider),'o',markersize = 15,\
                markeredgecolor='black', color='orange',markeredgewidth=2,\
                label = ng467['inst_id'][0].split('-')[0]) 

plt.quiver(np.mean(longlider),np.mean(latglider),acc_heat_transp_ug,0,\
           color='b',alpha=0.8,label=ng467['inst_id'][0].split('-')[0],scale=1.5*10**15)
plt.quiver(np.mean(longlider),np.mean(latglider),acc_heat_transp_u31,0,\
           color='r',alpha=0.8,label='GOFS 3.1',scale=1.5*10**15)
plt.legend()
plt.quiver(np.mean(longlider),np.mean(latglider),0,acc_heat_transp_v31,\
           color='r',alpha=0.8,scale=1.5*10**15)
plt.quiver(np.mean(longlider),np.mean(latglider),0,acc_heat_transp_vg,\
           color='b',alpha=0.8,scale=1.5*10**15)


plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Integrated Heat Transport \n During Hurricane Season 2018 ',size = 20)
plt.xlim([-66,-64])
plt.ylim([17,19])



file = 'map_Caribbean_heat_transp_'+ ng467['inst_id'][0].split('-')[0]       
plt.savefig(folder + file\
             ,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure heat transport along

fig, ax = plt.subplots(figsize=[10.5,3.5])
plt.plot(timeglider,heat_transp_alongg,label=ng467['inst_id'][0].split('-')[0],color='b')
ax.fill_between(timeglider, 0, heat_transp_alongg,alpha=0.4)
plt.plot(time31[oktime31],heat_transp_along31,label='GOFS3.1',color='r')
ax.fill_between(time31[oktime31], 0, heat_transp_along31,alpha=0.4)
plt.plot(timeglider,np.zeros(len(timeglider)),color='k',linestyle='--')
plt.legend(loc='upper left',fontsize=14)
plt.title('Heat Transport Top 200 m Along Channel Direction',size=20)
plt.ylim(-4*10**8,4*10**8)
plt.ylabel('$(\; \mathit{J/m^2 s} \:)$',size=16)

xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)

file = folder + 'heat_transport_along_channel_' + ng467['inst_id'][0].split('-')[0]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure heat transport cross channel

fig, ax = plt.subplots(figsize=[10.5,3.5])
plt.plot(timeglider,heat_transp_crossg,label=ng467['inst_id'][0].split('-')[0],color='b')
ax.fill_between(timeglider, 0, heat_transp_crossg,alpha=0.4)
plt.plot(time31[oktime31],heat_transp_cross31,label='GOFS3.1',color='r')
ax.fill_between(time31[oktime31], 0, heat_transp_cross31,alpha=0.4)
plt.plot(timeglider,np.zeros(len(timeglider)),color='k',linestyle='--')
plt.legend(loc='upper left',fontsize=14)
plt.title('Heat Transport Top 200 m Cross Channel Direction',size=20)
plt.ylim(-4*10**8,4*10**8)
plt.ylabel('$(\; \mathit{J/m^2 s} \:)$',size=16)

xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)

file = folder + 'heat_transport_cross_channel_' + ng467['inst_id'][0].split('-')[0]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Map Caribbean with integrated heat transport
angle_rot = 30
alpha = np.radians(angle_rot)
beta = np.radians(angle_rot-90)

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev,cmap=plt.get_cmap('BrBG'))
#plt.plot(longlider,latglider,'.k')
plt.plot(np.mean(longlider),np.mean(latglider),'o',markersize = 15,\
                markeredgecolor='black', color='orange',markeredgewidth=2,\
                label = ng467['inst_id'][0].split('-')[0]) 

plt.quiver(np.mean(longlider),np.mean(latglider),\
           acc_heat_transp_alongg*np.cos(alpha),acc_heat_transp_alongg*np.sin(alpha),\
           color='b',alpha=0.8,label=ng467['inst_id'][0].split('-')[0],scale=1.5*10**15)
plt.quiver(np.mean(longlider),np.mean(latglider),\
           acc_heat_transp_along31*np.cos(alpha),acc_heat_transp_along31*np.sin(alpha),\
           color='r',alpha=0.8,label='GOFS 3.1',scale=1.5*10**15)
plt.legend()
plt.quiver(np.mean(longlider),np.mean(latglider),\
           acc_heat_transp_cross31*np.cos(beta),acc_heat_transp_cross31*np.sin(beta),\
           color='r',alpha=0.8,scale=1.5*10**15)
plt.quiver(np.mean(longlider),np.mean(latglider),\
           acc_heat_transp_crossg*np.cos(beta),acc_heat_transp_crossg*np.sin(beta),\
           color='b',alpha=0.8,scale=1.5*10**15)


plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Integrated Heat Transport \n During Hurricane Season 2018 ',size = 20)
plt.xlim([-66,-64])
plt.ylim([17,19])



file = 'map_Caribbean_heat_transp2_'+ ng467['inst_id'][0].split('-')[0]       
plt.savefig(folder + file\
             ,bbox_inches = 'tight',pad_inches = 0.1) 