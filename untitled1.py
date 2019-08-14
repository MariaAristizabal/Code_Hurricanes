#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:35:17 2019

@author: aristizabal
"""
#%%
# getting the model grid positions for sublonm and sublatm
oklatdoppio = np.empty((len(oktime_doppio[0])))
oklatdoppio[:] = np.nan
oklondoppio= np.empty((len(oktime_doppio[0])))
oklondoppio[:] = np.nan
for t,tt in enumerate(oktime_doppio[0]):
    oklatmm = []
    oklonmm = []
    for pos_xi in np.arange(latrhodoppio.shape[1]):
        pos_eta = np.round(np.interp(sublatdoppio[t],latrhodoppio[:,pos_xi],np.arange(len(latrhodoppio[:,pos_xi])),\
                                     left=np.nan,right=np.nan))
        if np.isfinite(pos_eta):
            oklatmm.append((pos_eta).astype(int))
            oklonmm.append(pos_xi)
            
    pos = np.round(np.interp(sublondoppio[t],lonrhodoppio[oklatmm,oklonmm],np.arange(len(lonrhodoppio[oklatmm,oklonmm])))).astype(int)    
    oklatdoppio[t] = oklatmm[pos]
    oklondoppio[t] = oklonmm[pos]      
    
oklatdoppio = oklatdoppio.astype(int)
oklondoppio = oklondoppio.astype(int)

#%%

oklatdoppio1 = np.empty((len(oktime_doppio[0])))
oklatdoppio1[:] = np.nan
oklondoppio1 = np.empty((len(oktime_doppio[0])))
oklondoppio1[:] = np.nan

t=0
oklatmm = []
oklonmm = []
for pos_xi in np.arange(latrhodoppio.shape[1]):
    pos_eta = np.round(np.interp(sublatdoppio[t],latrhodoppio[:,pos_xi],np.arange(len(latrhodoppio[:,pos_xi])),\
                                 left=np.nan,right=np.nan))
    if np.isfinite(pos_eta):
        print('pos_xi ',pos_xi,' pos_eta',pos_eta)
        oklatmm.append((pos_eta).astype(int))
        oklonmm.append(pos_xi)
            
pos_lon = np.round(np.interp(sublondoppio[t],lonrhodoppio[oklatmm,oklonmm],np.arange(len(lonrhodoppio[oklatmm,oklonmm])))).astype(int)
   
oklatdoppio1[t] = oklatmm[pos_lon]
oklondoppio1[t] = oklonmm[pos_lon] 

oklatdoppio1 = oklatdoppio1.astype(int)
oklondoppio1 = oklondoppio1.astype(int)


#%%

oklatdoppio2 = np.empty((len(oktime_doppio[0])))
oklatdoppio2[:] = np.nan
oklondoppio2 = np.empty((len(oktime_doppio[0])))
oklondoppio2[:] = np.nan

t=0
oklatmm = []
oklonmm = []
for pos_eta in np.arange(latrhodoppio.shape[0]):
    pos_xi = np.round(np.interp(sublondoppio[t],lonrhodoppio[pos_eta,:],np.arange(len(lonrhodoppio[pos_eta,:])),\
                                 left=np.nan,right=np.nan))
    if np.isfinite(pos_xi):
        print('pos_xi ',pos_xi,' pos_eta',pos_eta)
        oklatmm.append(pos_eta)
        oklonmm.append(pos_xi.astype(int))
            
pos_lat = np.round(np.interp(sublatdoppio[t],latrhodoppio[oklatmm,oklonmm],np.arange(len(latrhodoppio[oklatmm,oklonmm])))).astype(int)
   
oklatdoppio2[t] = oklatmm[pos_lat]
oklondoppio2[t] = oklonmm[pos_lat] 

oklatdoppio2 = oklatdoppio2.astype(int)
oklondoppio2 = oklondoppio2.astype(int)


#%%
dist1 = np.sqrt((oklondoppio1[t]-sublondoppio[t])**2 + (oklatdoppio1[t]-sublatdoppio[t])**2) 
dist2 = np.sqrt((oklondoppio2[t]-sublondoppio[t])**2 + (oklatdoppio2[t]-sublatdoppio[t])**2) 




#%%

pos_xi = 41
pos_eta = 105
 
pos_lon = np.round(np.interp(sublondoppio[t],lonrhodoppio[oklatmm,oklonmm],np.arange(len(lonrhodoppio[oklatmm,oklonmm])))).astype(int)