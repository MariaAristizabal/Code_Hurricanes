function glider_transects_GOFS31_30_comparisons(url_glider,url_GOFS31,url_GOFS30,var,date_ini,date_end)

% The output of this function is three plots of:
% 1. a transect of the chosen variable from the glider data
% 2. a transect of the same varible following the glider track from GOFS 3.1 output
% 3. a transect of the same varible following the glider track from GOFS 3.0 output
%
% Inputs:
% url_glider = url address or directory on local computer where the netcdf 
%              file with the glider data resides
% url_GOFS31 = url address or directory on local computer where the netcdf 
%              file with GOFS 3.1 output resides
% url_GOFS30 = url address or directory on local computer where the nercdf 
%              file with GOFS 3.0 output resides
% var = variable to compare. Ex: 'water_temperature', 'salinity'. Make sure
%       to use the same name as defined in the netcdf file
% 
% Optional inputs
% date_ini = initial date the user wish to visualize the data. If empty,
%            then the default option is the beginning of the record
% date_end = final date the user wish to visualize the data. If empty,
%            then the default option is the end of the record

