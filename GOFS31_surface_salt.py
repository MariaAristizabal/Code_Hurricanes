#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:19:46 2018

@author: aristizabal
"""

#GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

#%%
  
import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

#%%
GOFS31 = Dataset(catalog31)

lat31 = GOFS31.variables['lat'][:]
lon31 = GOFS31.variables['lon'][:]
depth = GOFS31.variables['depth'][:]
time31 = GOFS31.variables['time']
time31 = netCDF4.num2date(time31[:],time31.units) 

#date_ini = datetime.datetime.strptime(dateini, '%Y/%m/%d/%H/%M') #Time already in UTC
#date_end = datetime.datetime.strptime(dateend, '%Y/%m/%d/%H/%M') #Time already in UTC

oklat31 = np.where(np.logical_and(lat31 > -5, lat31 < 40))
oklon31 = np.where(np.logical_and(lon31 > 260, lon31 < 360))

temp31 = GOFS31.variables['water_temp'][0,0,oklat31[0],oklon31[0]]
salt31 = GOFS31.variables['salinity'][0,0,oklat31[0],oklon31[0]]

#%% Figure Temperature

fig = plt.figure(num=None, figsize=(12, 8) )
m = Basemap(projection='merc',llcrnrlat=-5,urcrnrlat=40,llcrnrlon=260,urcrnrlon=360,resolution='c')
m.drawcoastlines()
x, y = m(*np.meshgrid(lon31[oklon31],lat31[oklat31]))
m.pcolormesh(x,y,temp31,shading='flat',cmap=plt.cm.jet)
m.colorbar(location='right')
m.fillcontinents(color='lightgrey',lake_color='lightblue')

#%% Figure Salinity

fig, ax = plt.subplots(figsize=(17, 5))
#fig = plt.figure(num=None, figsize=(12, 8) )
#ax = fig.add_subplot(111)
m = Basemap(projection='merc',llcrnrlat=-5,urcrnrlat=40,llcrnrlon=260,urcrnrlon=360,resolution='c')
m.drawcoastlines()
x, y = m(*np.meshgrid(lon31[oklon31],lat31[oklat31]))
m.pcolormesh(x,y,salt31,shading='flat',cmap=plt.cm.jet)
cbar = m.colorbar(location='right')
m.fillcontinents(color='lightgrey',lake_color='lightblue')
plt.clim(34,37)
cbar.set_label('psu', fontsize=20)
#cbar.ax.set_ylabel('Temperature ($^\circ$C)')
#cbar.ax.set_yticks(fontsize=16)
#cbar.set_yticklabel(fontsize=16)
#cbar.set_ticks(fontsize=16)
#cbar.set_ticklabels(fontsize=20)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Salinity/GOFS31_salinity_detail3")
plt.show()


#%%
fig = plt.figure(num=None, figsize=(12, 8) )
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='tan',lake_color='lightblue')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,60.),labels=[False,False,False,True],dashes=[2,2])
m.drawmapboundary(fill_color='lightblue')
plt.title("Mercator Projection")

#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m.shadedrelief()
plt.show()

#%%

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
c = ax.contourf(lon31[oklon31],lat31[oklat31],temp31)
cbar = fig.colorbar(c)
plt.axis([260,360,-5,40])
plt.axis('equal')