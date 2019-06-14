#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:47:42 2019

@author: aristizabal
"""

python -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu \
--service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS \
--product-id global-analysis-forecast-phy-001-024 \
--longitude-min -110 --longitude-max -10 
--latitude-min 0 --latitude-max 45 
--date-min "2019-06-02 12:00:00" --date-max "2019-06-04 12:00:00" 
--depth-min 0.493 --depth-max 541.09 
--variable thetao --variable so 
--out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> 
--user <USERNAME> --pwd <PASSWORD>