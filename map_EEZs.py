#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:51:47 2020

@author: aristizabal
"""

lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader

file_EEZs = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/World_EEZ_v11_20191118/eez_boundaries_v11.shp'

fig, ax = plt.subplots(figsize=(10, 5),subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
ax.add_feature(coast, edgecolor='black', facecolor='none')
ax.add_feature(cfeature.BORDERS)  # adds country borders  
ax.add_feature(cfeature.STATES)

shape_feature = cfeature.ShapelyFeature(Reader(file_EEZs).geometries(),
                               ccrs.PlateCarree(),edgecolor='grey',facecolor='none')
ax.add_feature(shape_feature,zorder=1)
plt.axis([-100,-10,0,50])
