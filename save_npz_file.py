#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:02:27 2019

@author: aristizabal
"""
#%%

np.savez('RTOFS_Michael.npz', time_matrixRTOFS=time_matrixRTOFS,target_zRTOFS=target_zRTOFS,\
         target_temp_RTOFS=target_temp_RTOFS)

#%%

np.savez('GOFS31_Michael.npz', time_matrixGOFS31=time_matrix_GOFS,target_zGOFS31=z_matrix_GOFS,\
         target_temp_GOFS31=target_temp_GOFS)

#%%

RTOFS_Michael = np.load('RTOFS_Michael.npz')

GOFS31_Michael = np.load('GOFS31_Michael.npz')
