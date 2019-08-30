#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:18:50 2019

@author: aristizabal
"""

#%% read_glider_data_thredds_server

import sys
from matplotlib import pyplot as plt
import numpy as np
import cmocean
import matplotlib.dates as mdates

# Do not produce figures on screen
#plt.switch_backend('agg')

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%% SC668

from read_glider_data import read_glider_data_thredds_server
from process_glider_data import grid_glider_data_thredd  

url_glider = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
var_name = 'temperature'
#var = 'salinity'
date_ini = '2019/08/01/00' # year/month/day/hour
date_end = '2019/08/29/00' # year/month/day/hour
scatter_plot = 'yes'
kwargs = dict(date_ini=date_ini,date_end=date_end)

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var_name,scatter_plot,**kwargs)
             
             
#%%       
print('gridding')
contour_plot='no'    
depthg_gridded, varg_gridded, timegg = \
                    grid_glider_data_thredd(timeg,latg,long,depthg,varg,var_name,inst_id)                  
                    
#%%                    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import cmocean

    # sort time variable
    okt = np.argsort(timeg)
    timegg = timeg[okt]
    #latgg = latg[okt]
    #longg = long[okt]
    depthgg = depthg[okt,:]
    vargg = varg[okt,:]
    
    delta_z =0.3

    # Grid variables
    depthg_gridded = np.arange(0,np.nanmax(depthgg),delta_z)
    varg_gridded = np.empty((len(timegg),len(depthg_gridded)))
    varg_gridded[:] = np.nan
#%%
    for t,tt in enumerate(timegg):
        print(t)
        depthu,oku = np.unique(depthgg[t,:],return_index=True)
        varu = vargg[t,oku]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        varf = varu[okdd]
        ok = np.asarray(np.isfinite(varf))
        if np.sum(ok) < 3:
            varg_gridded[t,:] = np.nan
        else:
            okd = depthg_gridded < np.max(depthf[ok])
            varg_gridded[t,okd] = np.interp(depthg_gridded[okd],depthf[ok],varf[ok])                   

#%%
'''
#***************** Mixed layer depth

tempg_gridded = varg_gridded                    
d10 = np.where(depthg_gridded >= 10)[0][0]
dt = 0.2

MLD_dt = np.empty(len(timegg)) 
MLD_dt[:] = np.nan
for t,tt in enumerate(timegg):
    T10 = tempg_gridded[d10,t]
    delta_T = T10 - tempg_gridded[:,t] 
    ok_mld = np.where(delta_T > dt)
    if ok_mld[0].size == 0:
        MLD_dt[t] = np.nan
    else:
        MLD_dt[t] = depthg_gridded[ok_mld[0][0]]                    

#***********************
        
fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                                       np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timegg,-depthg_gridded,varg_gridded.T,levels=[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,varg_gridded.T,cmap=cmocean.cm.thermal,**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')

ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);        
       
file = folder + inst_id 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#*******************

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                                       np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timegg,-depthg_gridded,varg_gridded.T,levels=[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,varg_gridded.T,cmap=cmocean.cm.thermal,**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')

ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);        
        
plt.ylim(-200,0)
plt.plot(timegg,-MLD_dt,'.k') 
       
file = folder + inst_id + '_surface'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
                                
#*******************

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

nlevels = 30 - 21 + 1
kw = dict(levels = np.linspace(21,30,nlevels))
                                      
plt.contour(timegg,-depthg_gridded,varg_gridded.T,levels=[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,varg_gridded.T,cmap=cmocean.cm.thermal,**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')

ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);        
        
plt.ylim(-200,0)
plt.plot(timegg,-MLD_dt,'.k') 
       
file = folder + inst_id + '_detail'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#***************************************
                
#%% SG665
                    
from read_glider_data import read_glider_data_thredds_server
from process_glider_data import grid_glider_data_thredd  

url_glider = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG665-20190718T1155/SG665-20190718T1155.nc3.nc'
var_name = 'temperature'
#var = 'salinity'
date_ini = '2019/08/01/00' # year/month/day/hour
date_end = '2019/08/29/00' # year/month/day/hour
scatter_plot = 'no'
kwargs = dict(date_ini=date_ini,date_end=date_end)

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var_name,scatter_plot,**kwargs)
                          
contour_plot='no'    
depthg_gridded, varg_gridded, timegg = \
                    grid_glider_data_thredd(timeg,latg,long,depthg,varg,var_name,inst_id)                      

#************ Mixed layer depth
                    
tempg_gridded = varg_gridded                    
d10 = np.where(depthg_gridded >= 10)[0][0]
dt = 0.2

MLD_dt = np.empty(len(timegg)) 
MLD_dt[:] = np.nan
for t,tt in enumerate(timegg):
    T10 = tempg_gridded[d10,t]
    delta_T = T10 - tempg_gridded[:,t] 
    ok_mld = np.where(delta_T > dt)
    if ok_mld[0].size == 0:
        MLD_dt[t] = np.nan
    else:
        MLD_dt[t] = depthg_gridded[ok_mld[0][0]]                    

#************************

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                                       np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timegg,-depthg_gridded,varg_gridded.T,levels=[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,varg_gridded.T,cmap=cmocean.cm.thermal,**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')

ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);        
       
file = folder + inst_id 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#*********************

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                                       np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timegg,-depthg_gridded,varg_gridded.T,levels=[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,varg_gridded.T,cmap=cmocean.cm.thermal,**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')

ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);        
        
plt.ylim(-200,0)
plt.plot(timegg,-MLD_dt,'.k') 
       
file = folder + inst_id + '_surface'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
                                
#******************

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

nlevels = 30 - 21 + 1
kw = dict(levels = np.linspace(21,30,nlevels))
                                      
plt.contour(timegg,-depthg_gridded,varg_gridded.T,levels=[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,varg_gridded.T,cmap=cmocean.cm.thermal,**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')

ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);        
        
plt.ylim(-200,0)
plt.plot(timegg,-MLD_dt,'.k') 
       
file = folder + inst_id + '_detail'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%  SG666
                    
from read_glider_data import read_glider_data_thredds_server
from process_glider_data import grid_glider_data_thredd  

url_glider = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG666-20190718T1206/SG666-20190718T1206.nc3.nc'
var_name = 'temperature'
#var = 'salinity'
date_ini = '2019/08/01/00' # year/month/day/hour
date_end = '2019/08/29/00' # year/month/day/hour
scatter_plot = 'no'
kwargs = dict(date_ini=date_ini,date_end=date_end)

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var_name,scatter_plot,**kwargs)
             
contour_plot='no'    
depthg_gridded, varg_gridded, timegg = \
                    grid_glider_data_thredd(timeg,latg,long,depthg,varg,var_name,inst_id)                     

file = folder + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

'''
        