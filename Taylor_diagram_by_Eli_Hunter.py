#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:07:00 2019

@author: root
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator,
                                                 DictFormatter)
import requests

#%% Create the ERDDAP request. The example below extracts data of obs_type=7 (Salinity) for year 2015.

urlfun=(
        'http://tds.marine.rutgers.edu/erddap/tabledap/DOPPIO_REANALYSIS_OBS.json?'
        'obs_type%2Cobs_provenance%2Ctime%2Clongitude%2Clatitude%2Cdepth%2Cobs_error%2Cobs_value%2Cmerc_value%2Chycom_value%2Cmodel_value'
        '&obs_type={}'
        '&time%3E={}'
        '&time%3C={}').format
starttime='2015-12-01T00:00:00Z'
endtime='2015-12-31T00:00:00Z'
url=urlfun('7',starttime,endtime)
print(url)

#%% Make the request and save it to a json OBJECT.

response=requests.get(url)
try:
    j=response.json()
except ValueError:
    print(response.text)
    print("Error accessing site")

j= j['table']

#%% Next we convert the JSON to a pandas dataframe, for easier processing.

df1 = pd.DataFrame([[d for d in x] for x in j['rows']],columns=[d for d in j['columnNames']])

#%% And calculate the statistics.

N=len(df1)-1  #For Unbiased estimmator.

xcorr=df1.corr()
stdevs=df1.std()
means=df1.mean()

cols=['CORRELATION','CRMSE','BIAS','MSTD','OSTD','RMSE']
    
tskill=np.empty((3,6))

#CORR
tskill[0,0]=xcorr['obs_value']['model_value']
tskill[1,0]=xcorr['obs_value']['hycom_value']
tskill[2,0]=xcorr['obs_value']['merc_value']
    
#CRMSE: centered root mean squared error normalized
tskill[0,1]=np.nansum(((df1.obs_value-means.obs_value)-(df1.model_value-means.model_value))**2)/(N*stdevs.obs_value*stdevs.model_value)
tskill[1,1]=np.nansum(((df1.obs_value-means.obs_value)-(df1.hycom_value-means.hycom_value))**2)/(N*stdevs.obs_value*stdevs.hycom_value)
tskill[2,1]=np.nansum(((df1.obs_value-means.obs_value)-(df1.merc_value-means.merc_value))**2)/(N*stdevs.obs_value*stdevs.merc_value)

#BIAS
tskill[0,2]=means.obs_value-means.model_value
tskill[1,2]=means.obs_value-means.hycom_value
tskill[2,2]=means.obs_value-means.merc_value

#MSTD
tskill[0,3]=stdevs.model_value
tskill[1,3]=stdevs.hycom_value
tskill[2,3]=stdevs.merc_value

#OSTD
tskill[0,4]=stdevs.obs_value
tskill[1,4]=stdevs.obs_value
tskill[2,4]=stdevs.obs_value

#CRMSE
tskill[0,5] = np.sqrt(np.nansum(((df1.obs_value-means.obs_value)-(df1.model_value-means.model_value))**2)/N)
tskill[1,5] = np.sqrt(np.nansum(((df1.obs_value-means.obs_value)-(df1.hycom_value-means.hycom_value))**2)/N)
tskill[2,5] = np.sqrt(np.nansum(((df1.obs_value-means.obs_value)-(df1.merc_value-means.merc_value))**2)/N)
    
skillscores=pd.DataFrame(tskill,
                        index=['Doppio','Hycom','Mercator'],
                        columns=cols)
print(skillscores)

#%% Create a plotting function. In this case for Taylor diagrams.

def taylor(scores):
    fig = plt.figure(1)
    tr = PolarAxes.PolarTransform()
    
    CCgrid= np.concatenate((np.arange(0,10,2)/10.,[0.9,0.95,0.99]))
    CCpolar=np.arccos(CCgrid)
    gf=FixedLocator(CCpolar)
    tf=DictFormatter(dict(zip(CCpolar, map(str,CCgrid))))
    
    STDgrid=np.arange(0,2.0,.5)
    gfs=FixedLocator(STDgrid)
    tfs=DictFormatter(dict(zip(STDgrid, map(str,STDgrid))))
    
    ra0, ra1 =0, np.pi/2
    cz0, cz1 = 0, 2
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
    ax1.axis["left"].label.set_text("Normalized Standard deviation")

    ax1.axis["right"].set_axis_direction("top")  
    ax1.axis["right"].toggle(ticklabels=True)
    ax1.axis["right"].major_ticklabels.set_axis_direction("left")
    
    ax1.axis["bottom"].set_visible(False) 
    ax1 = ax1.get_aux_axes(tr)
    
    rs,ts = np.meshgrid(np.linspace(0,2),np.linspace(0,np.pi/2))
    rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))
        
    contours = ax1.contour(ts, rs, rms, 3,colors='0.5')   
    plt.clabel(contours, inline=1, fontsize=10)
    plt.grid(linestyle=':',alpha=0.5) 
        
    
    for r in scores.iterrows():
        th=np.arccos(r[1].CORRELATION)
        rr=r[1].MSTD/r[1].OSTD
        
        ax1.plot(th,rr,'o',label=r[0])
        
    plt.legend(loc='upper right',bbox_to_anchor=[1.2,1.15])    
    plt.show()
    
#%%
    
taylor(skillscores)
    
#%% Create a plotting function. In this case for Taylor diagrams.

    fig = plt.figure()
    tr = PolarAxes.PolarTransform()
    
    CCgrid= np.concatenate((np.arange(0,10,2)/10.,[0.9,0.95,0.99]))
    CCpolar=np.arccos(CCgrid)
    gf=FixedLocator(CCpolar)
    tf=DictFormatter(dict(zip(CCpolar, map(str,CCgrid))))
    
    STDgrid=np.arange(0,np.round(skillscores.OSTD[0]+1,1),.5)
    gfs=FixedLocator(STDgrid)
    tfs=DictFormatter(dict(zip(STDgrid, map(str,STDgrid))))
    
    ra0, ra1 =0, np.pi/2
    cz0, cz1 = 0, np.round(skillscores.OSTD[0]+1,1)
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
    
    for r in skillscores.iterrows():
        theta=np.arccos(r[1].CORRELATION)
        rr=r[1].MSTD
        
        ax1.plot(theta,rr,'o',label=r[0])
    
    ax1.plot(0,r[1].OSTD,'o',label='Obs')    
    plt.legend(loc='upper right',bbox_to_anchor=[1.3,1.15])    
    plt.show()   
    
#%%
 
    crmse_doppio = np.sqrt(skillscores.OSTD[0]**2 + skillscores.MSTD[0]**2 \
                   - 2*skillscores.OSTD[0]*skillscores.MSTD[0]*skillscores.CORRELATION[0]) 
    
    crmse_hycom = np.sqrt(skillscores.OSTD[1]**2 + skillscores.MSTD[1]**2 \
                   - 2*skillscores.OSTD[1]*skillscores.MSTD[1]*skillscores.CORRELATION[1]) 
    
    crmse_mercat = np.sqrt(skillscores.OSTD[2]**2 + skillscores.MSTD[2]**2 \
                   - 2*skillscores.OSTD[2]*skillscores.MSTD[2]*skillscores.CORRELATION[2]) 
    
    rs,ts = np.meshgrid(np.linspace(0,2.4),np.linspace(0,np.pi))
    
    rms = np.sqrt(skillscores.OSTD[0]**2 + rs**2 - 2*rs*skillscores.OSTD[0]*np.cos(ts))
    
    contours = ax1.contour(ts, rs, rms,[0.2,0.5,1,2],colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10)
    plt.grid(linestyle=':',alpha=0.5) 
    c1 = ax1.contour(ts, rs, rms,[crmse_doppio],colors='steelblue')
    plt.clabel(c1, inline=1, fontsize=10)
    c1 = ax1.contour(ts, rs, rms,[crmse_hycom],colors='orange')
    #plt.clabel(c1, inline=1, fontsize=10)
    c1 = ax1.contour(ts, rs, rms,[crmse_mercat],colors='g')
    #plt.clabel(c1, inline=1, fontsize=10)
    
